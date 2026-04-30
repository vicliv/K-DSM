"""
Part of the code is adapted from the ResNet model (https://arxiv.org/abs/2106.11959)
provided at https://github.com/Yura52/rtdl and also adapted from https://github.com/rotot0/tab-ddpm
The model was modified to integrate Time Embedding.
On Diffusion Modeling for Anomaly Detection - Diffusion Time Estimation
"""
import os

import torch.nn.functional as F
from torch import nn
import torch
import sklearn.metrics as skm
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy
import os
import tqdm
import numpy as np
import functools

from adbench.myutils import Utils
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from .tab_resnet import TabResNet

from torch import Tensor

utils = Utils()
ModuleType = Union[str, Callable[..., nn.Module]]

import numpy as np
from sklearn.neighbors import NearestNeighbors

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_gmm(
    X_train, components_range=range(2, 21, 2), verbose=False
):

    def scorer(gmm, X, y=None):
        return np.quantile(gmm.score_samples(X), 0.1)

    gmm_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("GMM", GaussianMixture(reg_covar=1e-3)),
        ]
    )

    param_grid = dict(

        GMM__n_components=components_range,
        GMM__max_iter=[1000],
        GMM__covariance_type=["full"],
        GMM__init_params = ['kmeans'],
    )  # Full always performs best

    grid = GridSearchCV(
        estimator=gmm_clf,
        param_grid=param_grid,
        cv=5 if len(list(components_range)) > 1 else 2,
        n_jobs=1,
        verbose=1,
        scoring=scorer,
        error_score="raise"
    )

    grid_result = grid.fit(X_train)

    return grid.best_estimator_

class MSM():
    def __init__(self, seed=0, model_name="MSM", dataset="Not_specified", epochs=1000, batch_size=128, 
                 lr=3e-4, weight_decay=0.0, device=None, 
                 # Model parameters
                 sigma_min=1e-1, sigma_max=1.0, num_scales=20,
                 ndims=512, time_embedding_size=128, layers=12, dropout=0.0,
                 act="gelu", embedding_type="fourier", ema_rate=0.999, verbose=False,
                 # Optimizer parameters
                 optimizer="AdamW", beta1=0.9, beta2=0.999, grad_clip=1.0):
        """
        Initialize the MSM (Multi-Scale Matching) model.
        
        Args:
            seed: Random seed for reproducibility
            model_name: Name of the model
            dataset: Dataset name
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            device: Device to run on (cuda/cpu)
            noise_scale: List of noise scales (legacy parameter)
            
            # Model parameters
            sigma_min: Minimum noise
            sigma_max: Maximum noise level
            num_scales: Number of noise scales
            ndims: Number of hidden dimensions
            time_embedding_size: Size of time embeddings
            layers: Number of layers in the network
            dropout: Dropout rate
            act: Activation function
            embedding_type: Type of time embedding
            ema_rate: Exponential moving average rate
            
            # Optimizer parameters
            optimizer: Optimizer type ("AdamW" or "Adam")
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
            grad_clip: Gradient clipping threshold
        """
        self.seed = seed
        self.model_name = model_name
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        self.ndims = ndims
        self.time_embedding_size = time_embedding_size
        self.layers = layers
        self.dropout = dropout
        self.act = act
        self.embedding_type = embedding_type
        self.ema_rate = ema_rate
        self.verbose = verbose
        
        # Optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip
        
        # Initialize model
        self.model = None
        
        # Generate noise scales (geometric progression from sigma_max to sigma_min)
        self.sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.sigma_max), np.log(self.sigma_min), self.num_scales)),
            dtype=torch.float32
            ,device=self.device
        )
        
        
        
    def score_fn(self, x, t):
        """Assumes that the model is predicting the `logit noise`
        Following the log concrete gradient, we need to rescale
        the logits to turn them into scores
        """
        score = self.model(x, t.float())
        return torch.as_tensor(score, device=x.device)
    
    @torch.inference_mode()
    def scorer(self, x_batch, denoise_step=False):
        N = self.num_scales
        score_norms = torch.zeros(x_batch.shape[0], N, device=x_batch.device)
        # Single denoising step
        if denoise_step:
            vec_t = torch.ones(x_batch.shape[0], device=x_batch.device) * (N - 1)
            score = self.score_fn(x_batch, vec_t)
            score *= torch.linalg.norm(score, dim=1, keepdim=True) ** -1
            
            x_batch += score

        for idx in range(N):
            vec_t = (
                torch.ones((x_batch.shape[0],), device=x_batch.device, dtype=torch.long)
                * idx
            )
            
            score = self.score_fn(x_batch, vec_t)
            score_norms[:, idx] = torch.linalg.norm(
                score.reshape(score.shape[0], -1), dim=1
            )
        return score_norms

    
    def training_step(self, train_batch) -> torch.Tensor:

        idxs = torch.randint(
            self.num_scales,
            size=(train_batch.shape[0],),
            device=self.device,
            dtype=torch.long,
            requires_grad=False,
        )

        loss = self.single_loss_step(train_batch, idxs)

        return loss
    
    def validation_step(self, val_batch) -> torch.Tensor:
        x_batch= val_batch

        val_loss = 0.0
        val_err = 0.0

        t = torch.ones(
            size=(x_batch.shape[0],),
            device=self.device,
            dtype=torch.long,
            requires_grad=False,
        )
        L = self.num_scales

        for idx in range(0, self.num_scales, 3):
            loss = self.single_loss_step(x_batch, t * idx)
            val_loss += loss
        
        denom=len(range(0, self.num_scales, 3))
        val_err /= denom

        return val_loss
    
    def single_loss_step(self, x_batch, timestep_idxs):
        """
        Compute denoising score matching loss for a single step.
        
        Args:
            x_batch: Clean input data
            timestep_idxs: Indices of timesteps/noise levels
            return_error: If True, also return relative error for monitoring
            
        Returns:
            loss: Mean loss value
            rel_err (optional): Relative error between predicted and target scores
        """
        # Get noise level for this timestep
        sigma = self.sigmas[timestep_idxs][:, None]
        
        # Add noise to data
        noise = torch.randn_like(x_batch) * sigma
        x_noisy = x_batch + noise
        
        # Predict the score (gradient of log density)
        scores = self.score_fn(x_noisy, timestep_idxs)
        
        # Compute denoising score matching loss
        batch_sz = scores.shape[0]
        target = -noise / (sigma ** 2)
        loss = (scores - target) ** 2
        loss = 0.5 * torch.sum(loss.reshape(batch_sz, -1), dim=-1)
        # loss = torch.mean(loss.reshape(batch_sz, -1), dim=-1)
        # loss *= sigma[:, 0] ** 2
        
        return torch.mean(loss)
        
    
    def fit(self, X_train,Y_train):
        """
        Fit the MSM model using the training_step and validation_step methods.
        """
        train_loader = DataLoader(
            torch.from_numpy(X_train).float(), 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True
        )

        if self.model is None:  # Initialize the model only once
            self.model = TabResNet(
                input_dims=X_train.shape[-1],
                ndims=self.ndims,
                act=self.act,
                time_embedding_size=self.time_embedding_size,
                layers=self.layers,
                dropout=self.dropout,
                embedding_type=self.embedding_type
            ).to(self.device)

        if self.optimizer_type == "AdamW":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.beta1, self.beta2)
            )
        elif self.optimizer_type == "Adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.beta1, self.beta2)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Initialize tracking variables
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        train_losses = []
        
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            fold_train_loss = []
            
            for x in train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                
                # Use the training_step method
                loss = self.training_step(x)
                
                loss.backward()
                optimizer.step()
                fold_train_loss.append(loss.item())
            
            avg_train_loss = np.mean(fold_train_loss)
            train_losses.append(avg_train_loss)

            # Logging and evaluation
            if self.verbose and (self.epoch % 5 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}")   
        
        train_norm = self.get_score_norm(X_train)
        
        self.best_gmm_clf = train_gmm(train_norm)
        
        return self
    
    def get_score_norm(self, X, denoise=False, seed=0, batch_size=256):
        """
        Compute score norms for dataset.
        
        Args:
            X: Input data (numpy array)
            denoise: Whether to apply denoising step in scorer
            seed: Random seed (unused in current implementation)
            batch_size: Batch size for processing
            
        Returns:
            score_norms: Score norms for the data (numpy array)
        """
        
        dataset = TensorDataset(
            torch.from_numpy(X).float()
        )
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        score_norms = []
        
        with torch.cuda.device(0):
            for (x_batch,) in loader:  # Note the comma - unpacks the tuple
                s = (
                    self.scorer(x_batch.to(self.device), denoise_step=denoise)
                    .cpu()
                    .numpy()
                )
                score_norms.append(s)
            
            score_norms = np.concatenate(score_norms)
        
        return score_norms

    @torch.no_grad()
    def predict_score(self, X):
        
        test_norm = self.get_score_norm(X)
        
        gmm_test_score = self.best_gmm_clf.score_samples(test_norm)
        
        return -gmm_test_score
    
    
    
