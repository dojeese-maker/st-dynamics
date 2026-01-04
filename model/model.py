"""
Main ST-Dynamics model with strict separation of training and inference.

This module implements the core STDynamicsModel class that performs
representation learning for spatial transcriptomics temporal dynamics.

Critical principles:
- Strict separation between fit() and infer() methods
- No direct supervision from time labels during training
- Time information used ONLY for temporal consistency constraints
- Post-hoc latent time inference is completely separate from training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import logging
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings

from .encoder import create_encoder, Decoder
from .losses import STDynamicsLoss
from ..data.dataset import SpatialTranscriptomicsDataset
from ..config import Config


class STDynamicsModel(nn.Module):
    """
    ST-Dynamics model for spatial transcriptomics temporal dynamics analysis.
    
    This model learns latent representations from gene expression and spatial
    coordinates, incorporating temporal consistency and spatial smoothness
    constraints WITHOUT direct time supervision.
    
    Architecture:
    - Encoder: gene expression → latent embedding
    - Decoder: latent embedding → reconstructed gene expression (optional)
    - Loss: reconstruction + temporal consistency + spatial smoothness
    
    Key Methods:
    - fit(): Train model parameters (NO access to future timepoints)
    - infer_latent(): Generate latent embeddings (frozen parameters)
    - save/load: Model persistence
    """
    
    def __init__(
        self,
        config: Config,
        encoder_type: str = "mlp",
        use_decoder: bool = True,
        device: Optional[str] = None
    ):
        super(STDynamicsModel, self).__init__()
        
        self.config = config
        self.encoder_type = encoder_type
        self.use_decoder = use_decoder
        self.device = device or config.device
        self.is_fitted = False
        
        # Initialize model components
        self.encoder = create_encoder(config, encoder_type)
        self.decoder = Decoder(config) if use_decoder else None
        
        # Loss function
        self.criterion = STDynamicsLoss(config)
        
        # Optimizer (initialized in fit())
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'loss_components': []
        }
        
        # Move to device
        self.to(self.device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_reconstruction: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, n_genes)
            Gene expression input
        return_reconstruction : bool, optional
            Whether to compute reconstruction (uses decoder)
            
        Returns:
        --------
        outputs : dict
            Dictionary containing 'z' (latent embedding) and optionally 
            'x_recon', 'mu', 'logvar' depending on model configuration
        """
        
        outputs = {}
        
        # Encode to latent space
        if self.encoder_type == "variational":
            z, mu, logvar = self.encoder(x)
            outputs['z'] = z
            outputs['mu'] = mu
            outputs['logvar'] = logvar
        else:
            z = self.encoder(x)
            outputs['z'] = z
        
        # Decode (if requested and decoder available)
        if return_reconstruction is None:
            return_reconstruction = self.use_decoder and self.training
        
        if return_reconstruction and self.decoder is not None:
            x_recon = self.decoder(z)
            outputs['x_recon'] = x_recon
        
        return outputs
    
    def _prepare_batch_data(
        self, 
        batch: Dict[str, torch.Tensor],
        dataset: SpatialTranscriptomicsDataset
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Prepare batch data for training/inference.
        
        Separates inputs and targets, and prepares temporal pairs
        for consistency constraints.
        """
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Prepare inputs
        inputs = {
            'expression': batch['expression'],
            'coords': batch['coords'],
            'temporal_pairs': []  # Will be filled if temporal data available
        }
        
        # Get temporal pairs for consistency loss (if training)
        if self.training and 'time' in batch:
            temporal_pairs = dataset.get_temporal_pairs()
            
            # Convert to tensors and collect embeddings
            paired_embeddings = []
            for indices_t, indices_t_plus_1 in temporal_pairs:
                if len(indices_t) > 0 and len(indices_t_plus_1) > 0:
                    # Get expressions for temporal pairs
                    expr_t = dataset.X_processed[indices_t]
                    expr_t1 = dataset.X_processed[indices_t_plus_1]
                    
                    # Convert to tensors
                    expr_t = torch.FloatTensor(expr_t).to(self.device)
                    expr_t1 = torch.FloatTensor(expr_t1).to(self.device)
                    
                    # Get embeddings
                    with torch.no_grad() if not self.training else torch.enable_grad():
                        z_t = self.encoder(expr_t)
                        z_t1 = self.encoder(expr_t1)
                        
                        if self.encoder_type == "variational":
                            z_t = z_t[0]  # Take mean for variational encoder
                            z_t1 = z_t1[0]
                    
                    paired_embeddings.append((z_t, z_t1))
            
            inputs['temporal_pairs'] = paired_embeddings
        
        # Prepare outputs (same as inputs for autoencoder training)
        outputs = inputs.copy()
        
        return inputs, outputs
    
    def fit(
        self,
        train_dataset: SpatialTranscriptomicsDataset,
        val_dataset: Optional[SpatialTranscriptomicsDataset] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        early_stopping: bool = True,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train the ST-Dynamics model.
        
        This method performs representation learning using ONLY:
        - Gene expression reconstruction loss
        - Temporal consistency constraints (NO direct time supervision)
        - Spatial smoothness regularization
        
        Parameters:
        -----------
        train_dataset : SpatialTranscriptomicsDataset
            Training dataset
        val_dataset : SpatialTranscriptomicsDataset, optional
            Validation dataset for early stopping
        batch_size : int, optional
            Batch size (uses config default if None)
        max_epochs : int, optional
            Maximum training epochs (uses config default if None)
        learning_rate : float, optional
            Learning rate (uses config default if None)
        early_stopping : bool
            Whether to use early stopping based on validation loss
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        history : dict
            Training history containing losses and metrics
        """
        
        # Set training parameters
        batch_size = batch_size or self.config.batch_size
        max_epochs = max_epochs or self.config.max_epochs
        learning_rate = learning_rate or self.config.learning_rate
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device == 'cuda' else False
            )
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.early_stopping_patience // 2,
            verbose=verbose
        )
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        self.train()  # Set to training mode
        
        for epoch in range(max_epochs):
            # Training epoch
            train_loss, train_components = self._train_epoch(
                train_loader, train_dataset, epoch, verbose
            )
            
            # Validation epoch
            val_loss, val_components = None, None
            if val_loader is not None:
                val_loss, val_components = self._validate_epoch(
                    val_loader, val_dataset, epoch, verbose
                )
            
            # Update learning rate
            if val_loss is not None:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)
            
            # Store history
            self.training_history['epochs'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
            self.training_history['loss_components'].append(train_components)
            
            # Early stopping check
            if early_stopping and val_loss is not None:
                if val_loss < best_val_loss - self.config.early_stopping_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # Progress reporting
            if verbose and epoch % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{max_epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f if val_loss else 'N/A'}, "
                      f"LR: {lr:.2e}")
        
        # Restore best model if early stopping was used
        if early_stopping and best_model_state is not None:
            self.load_state_dict(best_model_state)
            if verbose:
                print(f"Restored best model with validation loss: {best_val_loss:.6f}")
        
        self.is_fitted = True
        
        return self.training_history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        train_dataset: SpatialTranscriptomicsDataset,
        epoch: int,
        verbose: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        
        self.train()
        total_loss = 0.0
        total_components = {}
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}") if verbose else train_loader
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Prepare batch data
            inputs, _ = self._prepare_batch_data(batch, train_dataset)
            
            # Forward pass
            outputs = self.forward(inputs['expression'], return_reconstruction=True)
            
            # Compute loss
            loss, loss_components = self.criterion(outputs, inputs)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_components.items():
                if key not in total_components:
                    total_components[key] = 0.0
                total_components[key] += value.item()
            
            n_batches += 1
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({'loss': loss.item()})
        
        # Average losses
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        val_dataset: SpatialTranscriptomicsDataset,
        epoch: int,
        verbose: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        
        self.eval()
        total_loss = 0.0
        total_components = {}
        n_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}") if verbose else val_loader
            
            for batch in pbar:
                # Prepare batch data
                inputs, _ = self._prepare_batch_data(batch, val_dataset)
                
                # Forward pass
                outputs = self.forward(inputs['expression'], return_reconstruction=True)
                
                # Compute loss
                loss, loss_components = self.criterion(outputs, inputs)
                
                # Accumulate losses
                total_loss += loss.item()
                for key, value in loss_components.items():
                    if key not in total_components:
                        total_components[key] = 0.0
                    total_components[key] += value.item()
                
                n_batches += 1
                
                # Update progress bar
                if verbose:
                    pbar.set_postfix({'val_loss': loss.item()})
        
        # Average losses
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def infer_latent(
        self,
        dataset: SpatialTranscriptomicsDataset,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Infer latent embeddings for given data.
        
        This method performs ONLY forward inference with frozen parameters.
        No training or parameter updates occur.
        
        Parameters:
        -----------
        dataset : SpatialTranscriptomicsDataset
            Dataset to generate embeddings for
        batch_size : int, optional
            Batch size for inference
            
        Returns:
        --------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Latent embeddings
        """
        
        if not self.is_fitted:
            warnings.warn("Model has not been fitted. Results may be unreliable.")
        
        batch_size = batch_size or self.config.batch_size
        
        # Create data loader
        inference_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Inference
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(inference_loader, desc="Inferring latent embeddings"):
                x = batch['expression'].to(self.device)
                
                # Forward pass (no reconstruction needed for inference)
                outputs = self.forward(x, return_reconstruction=False)
                z = outputs['z']
                
                all_embeddings.append(z.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        return embeddings
    
    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save model state and configuration.
        
        Parameters:
        -----------
        save_path : str or Path
            Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare state dictionary
        state = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'encoder_type': self.encoder_type,
            'use_decoder': self.use_decoder,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history
        }
        
        # Save
        torch.save(state, save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: Union[str, Path], device: Optional[str] = None) -> 'STDynamicsModel':
        """
        Load model from saved state.
        
        Parameters:
        -----------
        load_path : str or Path
            Path to load model from
        device : str, optional
            Device to load model to
            
        Returns:
        --------
        model : STDynamicsModel
            Loaded model
        """
        
        # Load state
        state = torch.load(load_path, map_location=device)
        
        # Create model
        model = cls(
            config=state['config'],
            encoder_type=state['encoder_type'],
            use_decoder=state['use_decoder'],
            device=device
        )
        
        # Load state dict
        model.load_state_dict(state['model_state_dict'])
        model.is_fitted = state['is_fitted']
        model.training_history = state['training_history']
        
        return model
    
    def get_model_summary(self) -> Dict[str, any]:
        """Get summary information about the model."""
        
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'encoder_type': self.encoder_type,
            'use_decoder': self.use_decoder,
            'total_parameters': n_params,
            'trainable_parameters': n_trainable,
            'device': self.device,
            'is_fitted': self.is_fitted,
            'config': vars(self.config)
        }