"""
Training Module for Music Generation

This module handles:
- Dataset preparation for PyTorch
- Training loop with validation
- Checkpointing and early stopping
- Training history and visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime


class MusicDataset(Dataset):
    """
    PyTorch Dataset for music sequences.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences, shape (n_samples, sequence_length)
            y: Target outputs, shape (n_samples,)
        """
        # Convert to PyTorch tensors
        self.X = torch.LongTensor(X)  # LongTensor for indices
        self.y = torch.LongTensor(y)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index."""
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    HOW IT WORKS:
    1. Track validation loss after each epoch
    2. If loss doesn't improve for 'patience' epochs → stop training
    3. Optionally restore best model weights
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement! Reset counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class MusicTrainer:
    """
    Handles training of the music generation model.
    
    Features:
    - Training with validation
    - Learning rate scheduling
    - Checkpointing (save best model)
    - Early stopping
    - Training history tracking
    - Loss visualization
    
    Usage:
        trainer = MusicTrainer(model, device)
        history = trainer.train(train_loader, val_loader, epochs=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        checkpoint_dir: str = "outputs/models"
    ):
        """
        Initialize trainer.
        
        Args:
            model: The neural network model
            device: Device to train on (CPU/GPU)
            learning_rate: Initial learning rate
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',         # Reduce when loss stops decreasing
            factor=0.5,         # Multiply LR by 0.5
            patience=5,         # Wait 5 epochs before reducing
        )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()  # Set model to training mode (enables dropout)
        total_loss = 0.0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_X, batch_y in pbar:
            # Move data to device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Zero gradients (IMPORTANT: PyTorch accumulates gradients)
            self.optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.model(batch_X)
            
            # Calculate loss
            loss = self.criterion(output, batch_y)
            
            # Backward pass (calculate gradients)
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients in RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()  # Set model to evaluation mode (disables dropout)
        total_loss = 0.0
        
        # No gradient calculation during validation (saves memory)
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output, _ = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            epochs: Maximum number of epochs
            early_stopping_patience: Epochs without improvement before stopping
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training")
        print(f"{'='*60}")
        print(f"   Epochs: {epochs}")
        print(f"   Device: {self.device}")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"{'='*60}\n")
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch results
            print(f"Epoch {epoch+1:3d}/{epochs} │ "
                  f"Train Loss: {train_loss:.4f} │ "
                  f"Val Loss: {val_loss:.4f} │ "
                  f"LR: {current_lr:.6f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss)
                print(f"         └── 💾 New best model saved!")
            
            # Early stopping check
            if early_stopping(val_loss):
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pt', epoch, val_loss)
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_history()
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"   Best Validation Loss: {best_val_loss:.4f}")
        print(f"   Model saved to: {self.checkpoint_dir}")
        print(f"{'='*60}")
        
        return self.history
    
    def save_checkpoint(
        self, 
        filename: str, 
        epoch: int, 
        val_loss: float
    ) -> None:
        """
        Save model checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embedding_dim': self.model.embedding_dim,
                'lstm_hidden_size': self.model.lstm_hidden_size,
                'num_lstm_layers': self.model.num_lstm_layers,
                'dropout': self.model.dropout
            }
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_history(self) -> None:
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        # Plot losses
        ax1.plot(epochs_range, self.history['train_loss'], 
                 label='Training Loss', color='blue', linewidth=2)
        ax1.plot(epochs_range, self.history['val_loss'], 
                 label='Validation Loss', color='red', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot learning rate
        ax2.plot(epochs_range, self.history['learning_rate'], 
                 color='green', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved to {plot_path}")


def prepare_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation data loaders.
    
    Args:
        X: Input sequences
        y: Target outputs
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = MusicDataset(X, y)
    
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle training data each epoch
        num_workers=0,      # Set >0 for parallel loading (if not Windows)
        pin_memory=True     # Faster CPU→GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # No need to shuffle validation
        num_workers=0,
        pin_memory=True
    )
    
    print(f"📦 Data Loaders Created:")
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


# Test when run directly
if __name__ == "__main__":
    print("Testing training module...\n")
    
    # Create fake data for testing
    vocab_size = 100
    seq_length = 50
    n_samples = 1000
    
    X = np.random.randint(0, vocab_size, (n_samples, seq_length))
    y = np.random.randint(0, vocab_size, (n_samples,))
    
    # Import model
    from src.model import create_model
    
    # Create model
    model, device = create_model(vocab_size)
    
    # Prepare data
    train_loader, val_loader = prepare_data_loaders(X, y, batch_size=32)
    
    # Create trainer
    trainer = MusicTrainer(model, device)
    
    # Train for a few epochs to test
    print("\n🧪 Running quick training test (3 epochs)...\n")
    history = trainer.train(train_loader, val_loader, epochs=3)
    
    print("\n✅ Training module test complete!")