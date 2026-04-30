from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINOv3 embeddings for MVTec.")
    parser.add_argument(
        "--mvtech-root",
        type=str,
        required=True,
        help="Path to the MVTec root directory.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model id.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="mvtech_dinov3",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face model/processor loaders.",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Run model in float16 on CUDA.",
    )
    return parser.parse_args()


def resolve_mvtech_root(user_root: str) -> Path:
    root = Path(user_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MVTec root does not exist: {root}")
    return root


def discover_images(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            parts = {x.lower() for x in p.parts}
            # MVTec stores pixel-level masks under ground_truth/ — skip them.
            if "ground_truth" in parts:
                continue
            files.append(p)
    files.sort()
    if not files:
        raise RuntimeError(f"No image files found under: {root}")
    return files


def infer_split_and_label(rel_path: Path) -> tuple[str, int]:
    parts = [p.lower() for p in rel_path.parts]
    split = "unknown"
    if "train" in parts:
        split = "train"
    elif "test" in parts:
        split = "test"

    # Common conventions across AD datasets.
    if "good" in parts or "normal" in parts:
        label = 0
    elif "anomaly" in parts:
        label = 1
    elif split == "train":
        label = 0
    elif split == "test":
        label = 1
    else:
        label = 0
    return split, label


@dataclass
class SampleMeta:
    abs_path: Path
    rel_path: str
    split: str
    label: int


class MvtechImageDataset(Dataset):
    def __init__(
        self,
        root: Path,
        samples: Sequence[SampleMeta],
        processor: AutoImageProcessor,
    ):
        self.root = root
        self.processor = processor
        self.meta = list(samples)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, object]:
        m = self.meta[idx]
        with Image.open(m.abs_path) as im:
            image = im.convert("RGB")
        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "path": m.rel_path,
            "split": m.split,
            "label": m.label,
        }


def collate_fn(batch: Iterable[dict[str, object]]) -> dict[str, object]:
    batch_list = list(batch)
    pixel_values = torch.stack([x["pixel_values"] for x in batch_list], dim=0)
    paths = [str(x["path"]) for x in batch_list]
    splits = [str(x["split"]) for x in batch_list]
    labels = np.asarray([int(x["label"]) for x in batch_list], dtype=np.int64)
    return {
        "pixel_values": pixel_values,
        "paths": paths,
        "splits": splits,
        "labels": labels,
    }


def extract_embeddings(
    root: Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
    device: str,
    trust_remote_code: bool,
    use_fp16: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    processor = AutoImageProcessor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.eval()
    model.to(device)

    files = discover_images(root)
    samples = []
    for f in files:
        rel = f.relative_to(root)
        split, label = infer_split_and_label(rel)
        samples.append(
            SampleMeta(abs_path=f, rel_path=rel.as_posix(), split=split, label=label)
        )

    ds = MvtechImageDataset(root=root, samples=samples, processor=processor)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_fn,
    )

    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []
    all_splits: list[str] = []
    all_labels: list[np.ndarray] = []

    amp_enabled = use_fp16 and device.startswith("cuda")
    amp_dtype = torch.float16
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    with torch.no_grad():
        for batch in tqdm(dl, desc="Extracting embeddings", unit="batch"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(pixel_values=pixel_values)
            # CLS token embedding for ViT-like models: [B, hidden_size]
            embeddings = outputs.pooler_output.float().cpu().numpy()
            all_embeddings.append(embeddings)
            all_paths.extend(batch["paths"])
            all_splits.extend(batch["splits"])
            all_labels.append(batch["labels"])

    embeddings_np = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)
    paths_np = np.asarray(all_paths, dtype=np.str_)
    splits_np = np.asarray(all_splits, dtype=np.str_)
    labels_np = np.concatenate(all_labels, axis=0).astype(np.int64, copy=False)

    return embeddings_np, paths_np, splits_np, labels_np


class MvtechEmbeddingDataset(Dataset):
    """
    Simple map-style dataset over saved embedding/index .npy files.
    """

    def __init__(
        self,
        root: str | Path,
        prefix: str = "mvtech_dinov3",
        mmap: bool = True,
        split: str | None = None,
    ):
        root = Path(root).expanduser().resolve()
        mmap_mode = "r" if mmap else None
        self.embeddings = np.load(root / f"{prefix}_embeddings.npy", mmap_mode=mmap_mode)
        self.paths = np.load(root / f"{prefix}_paths.npy", allow_pickle=False)
        self.splits = np.load(root / f"{prefix}_splits.npy", allow_pickle=False)
        self.labels = np.load(root / f"{prefix}_labels.npy", allow_pickle=False)

        if split is None:
            self.indices = np.arange(len(self.embeddings))
        else:
            self.indices = np.where(self.splits == split)[0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        real_idx = int(self.indices[idx])
        x = torch.from_numpy(np.asarray(self.embeddings[real_idx], dtype=np.float32))
        y = int(self.labels[real_idx])
        p = str(self.paths[real_idx])
        return x, y, p


def main() -> None:
    args = parse_args()
    root = resolve_mvtech_root(args.mvtech_root)

    embeddings, paths, splits, labels = extract_embeddings(
        root=root,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        use_fp16=args.use_fp16,
    )

    prefix = args.output_prefix
    np.save(root / f"{prefix}_embeddings.npy", embeddings)
    np.save(root / f"{prefix}_paths.npy", paths)
    np.save(root / f"{prefix}_splits.npy", splits)
    np.save(root / f"{prefix}_labels.npy", labels)

    print(f"Saved {len(embeddings)} embeddings to: {root}")
    print(f"Embedding shape: {embeddings.shape}, dtype={embeddings.dtype}")
    print(f"Train samples: {(splits == 'train').sum()}, test samples: {(splits == 'test').sum()}")
    print(f"Anomaly labels (1): {int(labels.sum())} / {len(labels)}")
    print("\nPyTorch usage example:")
    print(
        "from torch.utils.data import DataLoader\n"
        "from utils.extract_mvtech_dinov3_embeddings import MvtechEmbeddingDataset\n"
        f"ds = MvtechEmbeddingDataset(root=r'{root}', prefix='{prefix}', split='train')\n"
        "dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)\n"
    )


if __name__ == "__main__":
    main()
