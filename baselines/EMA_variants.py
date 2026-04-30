"""
EMA Teacher variants of DDAE, DTECategorical, and MSM.

Each wraps the original model's training loop with an EMA shadow copy.
Before each gradient step the EMA model scores the current batch; samples
above `filter_pct` are dropped so the online model only trains on the
high-density part of the distribution.

The EMA model changes slowly (ema_decay ≈ 0.999), giving a stable,
lag-smoothed density signal that avoids the feedback instability that
plagued live-score-norm truncation.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset

from baselines.DDAE import DDAE, DiffusionDenoiser
from baselines.DTE import DTECategorical, MLP, binning
from baselines.MSM.MSM import MSM, TabResNet, train_gmm


# ── DDAE EMA Teacher ──────────────────────────────────────────────────────────
# EMA score per sample: reconstruction error ‖x₀ - ema_denoiser(xₜ, t)‖.
# Anomalous x₀ produce high reconstruction error even at the same t as
# normal samples, because the EMA denoiser has only learned to invert noise
# for the high-density region.

class DDAE_EMATeacher(DDAE):
    """DDAE with EMA teacher that filters high-reconstruction-error samples."""

    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.02,
                 scheduler='linear', epochs=100, batch_size=64, learning_rate=1e-3,
                 device=None, eval_epochs=10, seed=0, model_name="DDAE_EMATeacher",
                 ema_decay=0.999, filter_pct=80):
        super().__init__(num_timesteps=num_timesteps, beta_start=beta_start,
                         beta_end=beta_end, scheduler=scheduler, epochs=epochs,
                         batch_size=batch_size, learning_rate=learning_rate,
                         device=device, eval_epochs=eval_epochs, seed=seed,
                         model_name=model_name)
        self.ema_decay = ema_decay
        self.filter_pct = filter_pct

    def fit(self, x_train, y_train=None, x_test=None, y_test=None):
        if self.denoiser is None:
            self.denoiser = DiffusionDenoiser(
                input_dim=x_train.shape[1],
                hidden_dims=[512, 512],
                time_emb_dim=4,
                time_emb_type='sinusoidal',
                activation='lrelu',
            ).to(self.device)

        ema_denoiser = copy.deepcopy(self.denoiser).eval()
        optimizer = Adam(self.denoiser.parameters(), lr=self.learning_rate)
        dataloader = DataLoader(torch.from_numpy(x_train).float(),
                                batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.denoiser.train()
            total_loss = 0

            for x_0 in dataloader:
                x_0 = x_0.to(self.device)
                t = torch.randint(1, self.T, (x_0.size(0),), device=self.device).long()
                x_t, _ = self.scheduler.q_sample(x_0, t)

                # EMA teacher: reconstruction error at (x_t, t)
                with torch.no_grad():
                    x_0_hat_ema = ema_denoiser(x_t, t.float())
                    ema_scores = torch.norm(x_0 - x_0_hat_ema, dim=1)
                    threshold = torch.quantile(ema_scores, self.filter_pct / 100.0)
                    mask = ema_scores <= threshold

                if mask.sum() > 1:
                    x_0_f, t_f = x_0[mask], t[mask]
                    x_t_f, _ = self.scheduler.q_sample(x_0_f, t_f)
                else:
                    x_0_f, t_f, x_t_f = x_0, t, x_t

                x_0_hat = self.denoiser(x_t_f, t_f.float())
                loss = nn.MSELoss()(x_0_hat, x_0_f)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for ep, p in zip(ema_denoiser.parameters(), self.denoiser.parameters()):
                    ep.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % (self.eval_epochs or self.epochs) == 0 and x_test is not None:
                from baselines.DDAE import evaluate_anomaly_detection
                metrics = evaluate_anomaly_detection(self.predict_score(x_test), y_test)
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f} | PR-AUC: {metrics['PR-AUC']:.4f}")

        return self


# ── DTECategorical EMA Teacher ────────────────────────────────────────────────
# EMA score per sample: expected bin = Σ_k k·softmax(ema_model(x_noisy))_k.
# At a given noise level t, anomalous samples should look "harder" to the
# model (higher predicted bin) because their noisy version at t resembles
# a normal sample at a much higher t.

class DTECategorical_EMATeacher(DTECategorical):
    """DTECategorical with EMA teacher that filters high-predicted-bin samples."""

    def __init__(self, seed=0, model_name="DTE_EMATeacher",
                 hidden_size=[256, 512, 256], epochs=400, batch_size=64,
                 lr=1e-4, weight_decay=5e-4, T=400, num_bins=7, device=None,
                 ema_decay=0.999, filter_pct=80):
        super().__init__(seed=seed, model_name=model_name, hidden_size=hidden_size,
                         epochs=epochs, batch_size=batch_size, lr=lr,
                         weight_decay=weight_decay, T=T, num_bins=num_bins,
                         device=device)
        self.ema_decay = ema_decay
        self.filter_pct = filter_pct

    def fit(self, X_train, y_train=None, X_test=None, y_test=None, verbose=False):
        if self.model is None:
            self.model = MLP([X_train.shape[-1]] + self.hidden_size,
                             num_bins=self.num_bins).to(self.device)

        ema_model = copy.deepcopy(self.model).eval()
        optimizer = Adam(self.model.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        train_loader = DataLoader(torch.from_numpy(X_train).float(),
                                  batch_size=self.batch_size, shuffle=True,
                                  drop_last=False)
        bin_weights = torch.arange(self.num_bins, device=self.device).float()

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = []

            for x in train_loader:
                x = x.to(self.device)
                t = torch.randint(0, self.T, (x.shape[0],), device=self.device).long()
                x_noisy = self.forward_noise(x, t)

                # EMA teacher: expected bin — high value means model sees sample as harder
                with torch.no_grad():
                    probs_ema = ema_model(x_noisy)          # (B, num_bins)
                    expected_bin = (probs_ema * bin_weights).sum(dim=1)
                    threshold = torch.quantile(expected_bin, self.filter_pct / 100.0)
                    mask = expected_bin <= threshold

                if mask.sum() > 1:
                    x_f = x[mask]
                    t_f = t[mask]
                else:
                    x_f, t_f = x, t

                loss = self.compute_loss(x_f, t_f)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

                for ep, p in zip(ema_model.parameters(), self.model.parameters()):
                    ep.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {np.mean(epoch_loss):.4f}")

        return self


# ── MSM EMA Teacher ───────────────────────────────────────────────────────────
# EMA score per sample: score norm at t=0 (smallest σ in the noise schedule).
# The smallest-σ score is the most sensitive to local density; high norm at t=0
# means the clean sample is in a low-density region.
# Note: MSM already stores an `ema_rate` for its own internal EMA (unused in
# the original code). We introduce a separate shadow model for teacher filtering.

class MSM_EMATeacher(MSM):
    """MSM with EMA teacher that filters high-score-norm (low-density) samples."""

    def __init__(self, seed=0, model_name="MSM_EMATeacher", dataset="Not_specified",
                 epochs=1000, batch_size=128, lr=3e-4, weight_decay=0.0, device=None,
                 sigma_min=1e-1, sigma_max=1.0, num_scales=20,
                 ndims=512, time_embedding_size=128, layers=12, dropout=0.0,
                 act="gelu", embedding_type="fourier", ema_rate=0.999, verbose=False,
                 optimizer="AdamW", beta1=0.9, beta2=0.999, grad_clip=1.0,
                 ema_decay_teacher=0.999, filter_pct=80):
        super().__init__(seed=seed, model_name=model_name, dataset=dataset,
                         epochs=epochs, batch_size=batch_size, lr=lr,
                         weight_decay=weight_decay, device=device,
                         sigma_min=sigma_min, sigma_max=sigma_max,
                         num_scales=num_scales, ndims=ndims,
                         time_embedding_size=time_embedding_size, layers=layers,
                         dropout=dropout, act=act, embedding_type=embedding_type,
                         ema_rate=ema_rate, verbose=verbose,
                         optimizer=optimizer, beta1=beta1, beta2=beta2,
                         grad_clip=grad_clip)
        self.ema_decay_teacher = ema_decay_teacher
        self.filter_pct = filter_pct

    def fit(self, X_train, Y_train):
        train_loader = DataLoader(torch.from_numpy(X_train).float(),
                                  batch_size=self.batch_size, shuffle=True,
                                  drop_last=True)

        if self.model is None:
            self.model = TabResNet(
                input_dims=X_train.shape[-1],
                ndims=self.ndims,
                act=self.act,
                time_embedding_size=self.time_embedding_size,
                layers=self.layers,
                dropout=self.dropout,
                embedding_type=self.embedding_type,
            ).to(self.device)

        if self.optimizer_type == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=self.lr,
                              weight_decay=self.weight_decay,
                              betas=(self.beta1, self.beta2))
        else:
            optimizer = Adam(self.model.parameters(), lr=self.lr,
                             weight_decay=self.weight_decay,
                             betas=(self.beta1, self.beta2))

        ema_teacher = copy.deepcopy(self.model).eval()

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        for epoch in range(self.epochs):
            self.model.train()
            fold_loss = []

            for x in train_loader:
                x = x.to(self.device)

                # EMA teacher: score norm at t=0 (smallest sigma, most density-sensitive)
                with torch.no_grad():
                    t_ref = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
                    ema_score = ema_teacher(x, t_ref.float())
                    ema_norms = torch.norm(ema_score, dim=-1)
                    threshold = torch.quantile(ema_norms, self.filter_pct / 100.0)
                    mask = ema_norms <= threshold

                x_filtered = x[mask] if mask.sum() > 1 else x

                optimizer.zero_grad()
                loss = self.training_step(x_filtered)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                fold_loss.append(loss.item())

                for ep, p in zip(ema_teacher.parameters(), self.model.parameters()):
                    ep.data.mul_(self.ema_decay_teacher).add_(p.data,
                                                              alpha=1.0 - self.ema_decay_teacher)

            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch}, Loss: {np.mean(fold_loss):.6f}")

        train_norm = self.get_score_norm(X_train)
        self.best_gmm_clf = train_gmm(train_norm)

        return self
