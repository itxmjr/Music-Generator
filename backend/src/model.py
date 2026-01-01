"""
LSTM Model for Music Generation

This module contains:
- MusicLSTM: The neural network architecture
- Model creation and configuration utilities

Architecture:
    Embedding → LSTM → LSTM → Dense → Output
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MusicLSTM(nn.Module):
    """
    LSTM-based neural network for music generation.
    
    Usage:
        model = MusicLSTM(vocab_size=500, embedding_dim=256)
        output = model(input_sequence)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize the LSTM model.
        
        Args:
            vocab_size: Number of unique notes/chords in vocabulary
            embedding_dim: Dimension of note embeddings
            lstm_hidden_size: Number of LSTM units per layer
            num_lstm_layers: Number of stacked LSTM layers
            dropout: Dropout probability for regularization
        """
        super(MusicLSTM, self).__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=None  # No padding token for now
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self.dense = nn.Linear(lstm_hidden_size, 256)
        
        self.output = nn.Linear(256, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Initialize weights using Xavier initialization.
        
        Default random initialization can lead to:
        - Vanishing gradients (weights too small)
        - Exploding gradients (weights too large)
        
        Xavier initialization keeps gradients in a good range.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights: orthogonal initialization works well
                    nn.init.orthogonal_(param)
                else:
                    # Other weights: Xavier uniform
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
               Contains indices of notes
            hidden: Optional tuple of (h_0, c_0) for LSTM
                   If None, initialized to zeros
        
        Returns:
            Tuple of:
            - output: Logits of shape (batch_size, vocab_size)
            - hidden: Tuple of (h_n, c_n) for next iteration
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # Embedding: (batch, seq_len) → (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch, seq_len, embedding_dim) → (batch, seq_len, hidden_size)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Take only the last output (we're predicting the next note)
        # lstm_out[:, -1, :] shape: (batch, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Dropout for regularization
        dropped = self.dropout_layer(last_output)
        
        # Dense layer with ReLU activation
        dense_out = torch.relu(self.dense(dropped))
        dense_out = self.dropout_layer(dense_out)
        
        # Output layer (no activation - we'll use CrossEntropyLoss which includes softmax)
        logits = self.output(dense_out)
        
        return logits, hidden
    
    def _init_hidden(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state and cell state for LSTM.
        
        Args:
            batch_size: Number of samples in batch
            device: Device to create tensors on (CPU/GPU)
        
        Returns:
            Tuple of (h_0, c_0) initialized to zeros
        """
        # Shape: (num_layers, batch_size, hidden_size)
        h_0 = torch.zeros(
            self.num_lstm_layers, 
            batch_size, 
            self.lstm_hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.num_lstm_layers, 
            batch_size, 
            self.lstm_hidden_size,
            device=device
        )
        
        return (h_0, c_0)
    
    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """
        Get a summary of the model architecture.
        
        Returns:
            Formatted string with model details
        """
        summary_str = f"""
╔══════════════════════════════════════════════════════════════╗
║                    MUSIC LSTM MODEL SUMMARY                  ║
╠══════════════════════════════════════════════════════════════╣
║  Vocabulary Size:     {self.vocab_size:>8}                               ║
║  Embedding Dimension: {self.embedding_dim:>8}                               ║
║  LSTM Hidden Size:    {self.lstm_hidden_size:>8}                               ║
║  LSTM Layers:         {self.num_lstm_layers:>8}                               ║
║  Dropout:             {self.dropout:>8.2f}                               ║
╠══════════════════════════════════════════════════════════════╣
║  Total Parameters:    {self.get_num_parameters():>8,}                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary_str


def create_model(
    vocab_size: int,
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[MusicLSTM, torch.device]:
    """
    Factory function to create and initialize a model.
    
    Args:
        vocab_size: Number of unique notes
        device: Optional device (auto-detects if None)
        **kwargs: Additional arguments for MusicLSTM
    
    Returns:
        Tuple of (model, device)
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = MusicLSTM(vocab_size=vocab_size, **kwargs)
    
    # Move to device
    model = model.to(device)
    
    # Print summary
    print(model.summary())
    print(f"🖥️  Using device: {device}")
    
    return model, device


# Test when run directly
if __name__ == "__main__":
    print("Testing MusicLSTM model...\n")
    
    # Create a model with sample vocab size
    vocab_size = 500
    model, device = create_model(vocab_size)
    
    # Create sample input (batch of 4, sequence length 100)
    sample_input = torch.randint(0, vocab_size, (4, 100)).to(device)
    
    # Forward pass
    output, hidden = model(sample_input)
    
    print(f"\n📊 Test Results:")
    print(f"   Input shape:  {sample_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Hidden state shape: h={hidden[0].shape}, c={hidden[1].shape}")
    
    # Verify output shape
    assert output.shape == (4, vocab_size), "Output shape mismatch!"
    print(f"\n✅ All tests passed!")