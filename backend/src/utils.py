"""
Utility Functions for Music Generation Project

This module contains helper functions used across the project:
- File handling
- Configuration management
- Logging utilities
- Common operations
"""

import json
import random
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    
    Setting seeds ensures same results each run (for debugging/comparison).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For complete reproducibility (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Get the best available device (GPU if available, else CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"🖥️  Using CPU")
    
    return device


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Config saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    return config


def get_timestamp() -> str:
    """
    Get current timestamp as formatted string.
    
    Returns:
        Timestamp string (YYYY-MM-DD_HH-MM-SS)
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


def format_number(num: int) -> str:
    """
    Format large numbers with commas.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string (e.g., 1,234,567)
    """
    return f"{num:,}"


def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        char: Character for border
        width: Total width
    """
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


# Quick test when run directly
if __name__ == "__main__":
    print("Testing utility functions...\n")
    
    # Test set_seed
    set_seed(42)
    
    # Test get_device
    device = get_device()
    
    # Test timestamp
    print(f"\n⏰ Timestamp: {get_timestamp()}")
    
    # Test section printing
    print_section("TEST SECTION")
    
    print("\n✅ All utilities working!")