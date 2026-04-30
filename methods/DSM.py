"""
Part of the code is adapted from the ResNet model (https://arxiv.org/abs/2106.11959)
provided at https://github.com/Yura52/rtdl and also adapted from https://github.com/rotot0/tab-ddpm
The model was modified to integrate Time Embedding.
On Diffusion Modeling for Anomaly Detection - Diffusion Time Estimation
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from typing import Callable, Union
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, norm

from models import ResNetDiffusion

ModuleType = Union[str, Callable[..., nn.Module]]

def tau_to_c(tau):
    z = norm.ppf(1 - tau / 2)
    return (z**2 - 3) / 24

def reorder_by_hist_peak(data, bins=50):
    """
    Transform 1D data so that histogram peak is centered and other bins
    are arranged outward in decreasing frequency order.
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    sorted_idx = np.argsort(-counts)

    new_positions = {}
    pos = 0
    direction = 1
    step = 1
    for idx in sorted_idx:
        new_positions[idx] = pos
        pos = direction * step
        direction *= -1
        if direction == 1:
            step += 1

    transformed = np.zeros_like(data)
    bin_indices = np.digitize(data, bin_edges) - 1
    for j in range(len(data)):
        idx = bin_indices[j]
        if idx < 0 or idx >= len(bin_centers):
            continue
        transformed[j] = new_positions[idx]

    return transformed


def compute_sigma_by_kurtosis(
    X, base_sigma=0.5, c=0.33, min_sigma=0.1, max_sigma=3.0, percentage=0.04, bins=200
):
    """
    Compute per-dimension sigma values based on kurtosis ratios,
    with histogram-peak recentering and outlier trimming.

    Args:
        X (np.ndarray): shape (n_samples, n_features)
        base_sigma (float): sigma assigned to the median-kurtosis dimension
        min_sigma (float): lower bound for sigma
        max_sigma (float): upper bound for sigma
        percentage (float): fraction of samples to trim (equally from both tails)
        bins (int): number of bins for histogram peak reordering

    Returns:
        sigmas: array of sigma values per dimension (bounded)
        kurts: array of kurtosis per dimension
    """
    n_original = X.shape[0]

    X_sorted = np.sort(X, axis=0)
    start = int(n_original * (percentage / 2))
    end = n_original - start
    X_clean = X_sorted[start:end]

    kurts = []
    for j in range(X_clean.shape[1]):
        dim_data = X_clean[:, j]

        uniques = set(np.unique(dim_data[~np.isnan(dim_data)]))
        if uniques.issubset({0, 1}) or uniques.issubset({0.0, 1.0}) or \
                uniques.issubset({-1, 0, 1}) or uniques.issubset({-1.0, 0.0, 1.0}):
            kurts.append(0.9 / base_sigma)
            continue

        dim_transformed = reorder_by_hist_peak(dim_data, bins=bins)
        dim_transformed = StandardScaler().fit_transform(dim_transformed.reshape(-1, 1)).ravel()
        k = kurtosis(dim_transformed, fisher=False, nan_policy="omit")
        kurts.append(k)

    kurts = np.nan_to_num(
        np.array(kurts),
        nan=1000.0,
        posinf=1000.0,
        neginf=0.0001,
    )

    sigmas = base_sigma * (1 + c * (kurts - 3.0))

    sigmas = np.clip(sigmas, min_sigma, max_sigma)
    
    return sigmas, kurts

from scipy.special import gammaln, gammaincinv
from scipy.optimize import brentq

def compute_sigma_by_kurtosis_ggd(
    X, base_sigma=0.5, tau=0.05, min_sigma=0.1, max_sigma=3.0, percentage=0.04, bins=200
):
    """
    Compute per-dimension sigma values based on kurtosis ratios,
    with histogram-peak recentering and outlier trimming.

    Args:
        X (np.ndarray): shape (n_samples, n_features)
        base_sigma (float): sigma assigned to the median-kurtosis dimension
        min_sigma (float): lower bound for sigma
        max_sigma (float): upper bound for sigma
        percentage (float): fraction of samples to trim (equally from both tails)
        bins (int): number of bins for histogram peak reordering

    Returns:
        sigmas: array of sigma values per dimension (bounded)
        kurts: array of kurtosis per dimension
    """
    n_original = X.shape[0]

    X_sorted = np.sort(X, axis=0)
    start = int(n_original * (percentage / 2))
    end = n_original - start
    X_clean = X_sorted[start:end]

    kurts = []
    for j in range(X_clean.shape[1]):
        dim_data = X_clean[:, j]

        uniques = set(np.unique(dim_data[~np.isnan(dim_data)]))
        if uniques.issubset({0, 1}) or uniques.issubset({0.0, 1.0}) or \
                uniques.issubset({-1, 0, 1}) or uniques.issubset({-1.0, 0.0, 1.0}):
            kurts.append(0.9 / base_sigma)
            continue

        dim_transformed = reorder_by_hist_peak(dim_data, bins=bins)
        dim_transformed = StandardScaler().fit_transform(dim_transformed.reshape(-1, 1)).ravel()
        k = kurtosis(dim_transformed, fisher=False, nan_policy="omit")
        kurts.append(k)

    kurts = np.nan_to_num(
        np.array(kurts),
        nan=1000.0,
        posinf=1000.0,
        neginf=0.0001,
    )
    
    def _ggd_log_kurtosis(beta):
        # log(Γ(5/β)·Γ(1/β) / Γ(3/β)²), computed in log-space to avoid overflow
        return gammaln(5/beta) + gammaln(1/beta) - 2*gammaln(3/beta)

    def _ggd_tail_radius(beta):
        log_a = 0.5 * (gammaln(1/beta) - gammaln(3/beta))  # log of unit-variance scale
        return np.exp(log_a) * gammaincinv(1/beta, 1 - tau) ** (1/beta)

    _R_gaussian = _ggd_tail_radius(2.0)  # reference tail radius at kappa=3 (beta=2)

    def _kappa_to_sigma(kappa):
        kappa = float(np.clip(kappa, 1.802, 200.0))  # GGD is defined for kappa >= 1.8
        beta_hat = brentq(
            lambda b: _ggd_log_kurtosis(b) - np.log(kappa),
            a=0.02, b=2000.0, xtol=1e-6,
        )
        return base_sigma * _ggd_tail_radius(beta_hat) / _R_gaussian

    sigmas = np.array([_kappa_to_sigma(k) for k in kurts])
    sigmas = np.clip(sigmas, min_sigma, max_sigma)
    
    return sigmas, kurts

class DSM_base:
    def __init__(self, seed=0, sigma=0.5, epochs=300, batch_size=128,
                 lr=5e-4, weight_decay=5e-5, device=None, drop_last=True, verbose=False):
        self.seed = seed
        self.sigma = sigma
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_last = drop_last
        self.model = None
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.verbose = verbose
        
    def loss_fn(self, score_model, x):
        raise NotImplementedError

    def fit(self, X_train, y_train=None, project_name="default", logger=None):
        # make sure batch size is not larger than dataset size to avoid BatchNorm issues
        if self.batch_size > len(X_train):
            if self.verbose:
                print(f"Warning: batch_size {self.batch_size} is larger than dataset size {len(X_train)}. "
                      f"Reducing batch_size to {len(X_train)} to avoid BatchNorm issues.")
            self.batch_size = len(X_train)
        train_loader = DataLoader(
            torch.from_numpy(X_train).float(),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
        )

        if self.model is None:
            params = {
                "d_main": 512,
                "n_blocks": 6,
                "d_hidden": 512,
                "dropout_first": 0.2,
                "dropout_second": 0.1,
            }
            self.model = ResNetDiffusion(X_train.shape[-1], 0, params, self.sigma).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_losses = []
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []

            for x in train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                loss = self.loss_fn(self.model, x)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            train_losses.append(np.mean(train_loss))

            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch}, Train Loss: {np.mean(train_loss)}")
                if skipped_singleton_batches > 0:
                    print(f"Skipped {skipped_singleton_batches} singleton batch(es) for BatchNorm stability")
                if logger:
                    logger({"epoch": epoch, "train_loss": np.mean(train_loss)})

        return self

    @torch.no_grad()
    def predict_score(self, X):
        test_loader = DataLoader(
            torch.from_numpy(X).float(), batch_size=1000, shuffle=False, drop_last=False
        )
        preds = []
        self.model.eval()
        for x in test_loader:
            x = x.to(self.device)
            out = self.model(x)
            pred = torch.norm(out, dim=-1)
            preds.append(pred.detach().cpu().numpy())

        return np.concatenate(preds, axis=0)


class DSM(DSM_base):
    """DSM with a single noisy sample per input and a fixed scalar sigma."""

    def __init__(self, seed=0, sigma=0.5, epochs=300, batch_size=128,
                 lr=5e-4, weight_decay=5e-5, device=None, model_name="DSM", drop_last=True, verbose=False):
        super().__init__(seed=seed, sigma=sigma, epochs=epochs, batch_size=batch_size,
                         lr=lr, weight_decay=weight_decay, device=device, drop_last=drop_last, verbose=verbose)

    def loss_fn(self, score_model, x):
        noise = torch.randn_like(x) * self.sigma
        x_tilde = x + noise
        score_pred = score_model(x_tilde)
        target = (x - x_tilde) / self.sigma ** 2
        return 0.5 * ((score_pred - target) ** 2).sum(dim=1).mean()


class KDSM(DSM_base):
    """DSM with per-dimension sigma derived from kurtosis and multiple noisy samples."""

    def __init__(self, seed=0, sigma=0.5, epochs=500, batch_size=128, c = 0.33, tau = None,
             lr=5e-4, weight_decay=5e-5, device=None, model_name="KDSM", drop_last=False, verbose=False, loss_type=None):
        super().__init__(seed=seed, sigma=sigma, epochs=epochs, batch_size=batch_size,
                 lr=lr, weight_decay=weight_decay, device=device, drop_last=drop_last, verbose=verbose)
        self.kurt_sigma = sigma  # will be replaced by a tensor in fit()
        self.loss_type = loss_type

        if tau is None:
            self.c = c
        else:
            self.c = tau_to_c(tau)

    def fit(self, X_train, y_train=None, project_name="default", logger=None):
        sigma_vec_np, _ = compute_sigma_by_kurtosis(np.asarray(X_train), base_sigma=self.sigma, c=self.c)
        self.sigma = float(np.mean(sigma_vec_np))
        self.kurt_sigma = torch.tensor(sigma_vec_np, dtype=torch.float32, device=self.device)
        print("Computed per-feature sigmas: ", self.kurt_sigma.mean().item())
        return super().fit(X_train, y_train=y_train, project_name=project_name, logger=logger)

    def loss_fn(self, score_model, x, num_noisy_samples=3):
        batch_size, dim = x.shape
        sigma = self.kurt_sigma

        x_expanded = x.unsqueeze(1).repeat(1, num_noisy_samples, 1).reshape(-1, dim)
        noise = torch.randn_like(x_expanded) * sigma
        x_tilde = x_expanded + noise

        score_pred = score_model(x_tilde)

        x_repeated = x.unsqueeze(1).repeat(1, num_noisy_samples, 1).reshape(-1, dim)
        
        if self.loss_type == "squared":
            target = (x_repeated - x_tilde) / sigma ** 2
        else:
            target = -(x_tilde - x_repeated) / sigma

        return 0.5 * ((score_pred - target) ** 2).sum(dim=1).mean()

class KDSM_GGD(DSM_base):
    """DSM with per-dimension sigma derived from kurtosis and multiple noisy samples."""

    def __init__(self, seed=0, sigma=0.5, epochs=500, batch_size=128, tau = 0.33,
             lr=5e-4, weight_decay=5e-5, device=None, model_name="KDSM", drop_last=False, verbose=False):
        super().__init__(seed=seed, sigma=sigma, epochs=epochs, batch_size=batch_size,
                 lr=lr, weight_decay=weight_decay, device=device, drop_last=drop_last, verbose=verbose)
        self.kurt_sigma = sigma  # will be replaced by a tensor in fit()
        self.tau = tau

    def fit(self, X_train, y_train=None, project_name="default", logger=None):
        sigma_vec_np, _ = compute_sigma_by_kurtosis_ggd(np.asarray(X_train), base_sigma=self.sigma, tau=self.tau)
        self.sigma = float(np.mean(sigma_vec_np))
        self.kurt_sigma = torch.tensor(sigma_vec_np, dtype=torch.float32, device=self.device)
        print("Computed per-feature sigmas: ", self.kurt_sigma)
        return super().fit(X_train, y_train=y_train, project_name=project_name, logger=logger)

    def loss_fn(self, score_model, x, num_noisy_samples=3):
        batch_size, dim = x.shape
        sigma = self.kurt_sigma

        x_expanded = x.unsqueeze(1).repeat(1, num_noisy_samples, 1).reshape(-1, dim)
        noise = torch.randn_like(x_expanded) * sigma
        x_tilde = x_expanded + noise

        score_pred = score_model(x_tilde)

        x_repeated = x.unsqueeze(1).repeat(1, num_noisy_samples, 1).reshape(-1, dim)
        target = -(x_tilde - x_repeated) / sigma

        return 0.5 * ((score_pred - target) ** 2).sum(dim=1).mean()
    

import copy as _copy

def _make_model_and_optimizer(cls_instance, d_in):
    """Shared helper: initialise ResNetDiffusion + Adam if not yet done."""
    if cls_instance.model is None:
        params = {"d_main": 512, "n_blocks": 6, "d_hidden": 512,
                  "dropout_first": 0.2, "dropout_second": 0.1}
        cls_instance.model = ResNetDiffusion(d_in, 0, params, cls_instance.sigma).to(cls_instance.device)
    return Adam(cls_instance.model.parameters(),
                lr=cls_instance.lr, weight_decay=cls_instance.weight_decay)


class DSM_EMATeacher(DSM_base):
    """DSM: EMA teacher filters low-density batch samples before backprop."""

    def __init__(self, seed=0, sigma=0.5, epochs=300, batch_size=128,
                 ema_decay=0.999, filter_pct=80,
                 lr=5e-4, weight_decay=5e-5, device=None,
                 model_name="DSM_EMATeacher", drop_last=True, verbose=False):
        super().__init__(seed=seed, sigma=sigma, epochs=epochs, batch_size=batch_size,
                         lr=lr, weight_decay=weight_decay, device=device,
                         drop_last=drop_last, verbose=verbose)
        self.ema_decay = ema_decay
        # Accept either 85 or 0.85 for backward compatibility.
        self.filter_pct = filter_pct * 100 if filter_pct <= 1 else filter_pct

    def fit(self, X_train, y_train=None, project_name="default", logger=None):
        if self.batch_size > len(X_train):
            self.batch_size = len(X_train)
        X_tensor = torch.from_numpy(X_train).float()
        optimizer = _make_model_and_optimizer(self, X_train.shape[-1])
        ema_model = _copy.deepcopy(self.model).eval()

        for epoch in range(self.epochs):
            loader = DataLoader(X_tensor, batch_size=self.batch_size,
                                shuffle=True, drop_last=self.drop_last)
            self.model.train()
            train_loss = []
            for x in loader:
                x = x.to(self.device)
                with torch.no_grad():
                    ema_norms = torch.norm(ema_model(x), dim=-1)
                    threshold = torch.quantile(ema_norms, self.filter_pct / 100.0)
                    mask = ema_norms <= threshold
                x_filtered = x[mask] if mask.sum() > 1 else x

                optimizer.zero_grad()
                loss = self.loss_fn(self.model, x_filtered)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

                for ep, p in zip(ema_model.parameters(), self.model.parameters()):
                    ep.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch}, Loss: {np.mean(train_loss):.4f}")
        return self

    def loss_fn(self, score_model, x):
        noise = torch.randn_like(x) * self.sigma
        x_tilde = x + noise
        score_pred = score_model(x_tilde)
        target = (x - x_tilde) / self.sigma ** 2
        return 0.5 * ((score_pred - target) ** 2).sum(dim=1).mean()


class KDSM_EMATeacher(DSM_base):
    """KDSM: EMA teacher filters low-density batch samples before backprop."""

    def __init__(self, seed=0, sigma=0.5, epochs=500, batch_size=128, c=0.33, tau=None,
                 ema_decay=0.999, filter_pct=80,
                 lr=5e-4, weight_decay=5e-5, device=None,
                 model_name="KDSM_EMATeacher", drop_last=False, verbose=False):
        super().__init__(seed=seed, sigma=sigma, epochs=epochs, batch_size=batch_size,
                         lr=lr, weight_decay=weight_decay, device=device,
                         drop_last=drop_last, verbose=verbose)
        self.kurt_sigma = sigma
        self.ema_decay = ema_decay
        # Accept either 85 or 0.85 for backward compatibility.
        self.filter_pct = filter_pct * 100 if filter_pct <= 1 else filter_pct
        self.c = c if tau is None else tau_to_c(tau)

    def fit(self, X_train, y_train=None, project_name="default", logger=None):
        if self.batch_size > len(X_train):
            self.batch_size = len(X_train)

        sigma_vec_np, _ = compute_sigma_by_kurtosis(np.asarray(X_train), base_sigma=self.sigma, c=self.c)
        self.sigma = float(np.mean(sigma_vec_np))
        self.kurt_sigma = torch.tensor(sigma_vec_np, dtype=torch.float32, device=self.device)

        X_tensor = torch.from_numpy(X_train).float()
        optimizer = _make_model_and_optimizer(self, X_train.shape[-1])
        ema_model = _copy.deepcopy(self.model).eval()

        for epoch in range(self.epochs):
            loader = DataLoader(X_tensor, batch_size=self.batch_size,
                                shuffle=True, drop_last=self.drop_last)
            self.model.train()
            train_loss = []
            for x in loader:
                x = x.to(self.device)
                with torch.no_grad():
                    ema_norms = torch.norm(ema_model(x), dim=-1)
                    threshold = torch.quantile(ema_norms, self.filter_pct / 100.0)
                    mask = ema_norms <= threshold
                x_filtered = x[mask] if mask.sum() > 1 else x

                optimizer.zero_grad()
                loss = self.loss_fn(self.model, x_filtered)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

                for ep, p in zip(ema_model.parameters(), self.model.parameters()):
                    ep.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch}, Loss: {np.mean(train_loss):.4f}")
        return self

    def loss_fn(self, score_model, x, num_noisy_samples=3):
        batch_size, dim = x.shape
        sigma = self.kurt_sigma
        x_expanded = x.unsqueeze(1).repeat(1, num_noisy_samples, 1).reshape(-1, dim)
        noise = torch.randn_like(x_expanded) * sigma
        x_tilde = x_expanded + noise
        score_pred = score_model(x_tilde)
        x_repeated = x.unsqueeze(1).repeat(1, num_noisy_samples, 1).reshape(-1, dim)
        target = -(x_tilde - x_repeated) / sigma
        return 0.5 * ((score_pred - target) ** 2).sum(dim=1).mean()