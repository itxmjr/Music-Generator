"""
Preprocessor Module for Music Generation

This module handles:
- Building vocabulary from notes
- Converting notes to numerical sequences
- Creating training sequences (X, y pairs)
- Saving/loading vocabulary for inference
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter


class MusicPreprocessor:
    """
    Preprocesses musical notes for neural network training.
    
    Usage:
        preprocessor = MusicPreprocessor(sequence_length=100)
        X, y = preprocessor.prepare_sequences(notes)
    """
    
    def __init__(self, sequence_length: int = 100):
        """
        Initialize the preprocessor.
        
        Args:
            sequence_length: Number of notes to use as input for prediction
                           (longer = more context but slower training)
        """
        self.sequence_length = sequence_length
        
        self.note_to_int: Dict[str, int] = {}
        self.int_to_note: Dict[int, str] = {}
        self.vocab_size: int = 0
        
        self._vocab_built = False
    
    def build_vocabulary(self, notes: List[str], min_count: int = 1) -> None:
        """
        Build vocabulary from a list of notes.
        
        Args:
            notes: List of note/chord strings
            min_count: Minimum occurrences to include in vocabulary
                      (helps filter out rare/erroneous notes)
        """
        # Count occurrences of each note
        note_counts = Counter(notes)
        
        # Filter by minimum count and sort for reproducibility
        filtered_notes = sorted([
            note for note, count in note_counts.items() 
            if count >= min_count
        ])
        
        # Create mappings
        self.note_to_int = {note: idx for idx, note in enumerate(filtered_notes)}
        self.int_to_note = {idx: note for note, idx in self.note_to_int.items()}
        self.vocab_size = len(filtered_notes)
        self._vocab_built = True
        
        # Print statistics
        total_unique = len(note_counts)
        filtered_out = total_unique - self.vocab_size
        
        print(f"📊 Vocabulary Statistics:")
        print(f"   Total unique notes/chords: {total_unique}")
        print(f"   Filtered out (count < {min_count}): {filtered_out}")
        print(f"   Final vocabulary size: {self.vocab_size}")
        print(f"   Sample mappings: {dict(list(self.note_to_int.items())[:5])}")
    
    def notes_to_integers(self, notes: List[str]) -> List[int]:
        """
        Convert a list of notes to integers.
        
        Args:
            notes: List of note strings
            
        Returns:
            List of integers
        """
        if not self._vocab_built:
            raise ValueError("Vocabulary not built! Call build_vocabulary() first.")
        
        integers = []
        unknown_notes = set()
        
        for note in notes:
            if note in self.note_to_int:
                integers.append(self.note_to_int[note])
            else:
                unknown_notes.add(note)
        
        if unknown_notes:
            print(f"⚠️  Skipped {len(unknown_notes)} unknown notes")
        
        return integers
    
    def integers_to_notes(self, integers: List[int]) -> List[str]:
        """
        Convert integers back to notes.
        
        Args:
            integers: List of integer indices
            
        Returns:
            List of note strings
        """
        if not self._vocab_built:
            raise ValueError("Vocabulary not built! Call build_vocabulary() first.")
        
        return [self.int_to_note[i] for i in integers if i in self.int_to_note]
    
    def prepare_sequences(
        self, 
        notes: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for training.
        
        Args:
            notes: List of note strings
            
        Returns:
            Tuple of (X, y) where:
            - X: Input sequences, shape (n_samples, sequence_length)
            - y: Output labels, shape (n_samples,)
        """
        if not self._vocab_built:
            raise ValueError("Vocabulary not built! Call build_vocabulary() first.")
        
        note_integers = self.notes_to_integers(notes)
        
        # Check if we have enough notes
        if len(note_integers) <= self.sequence_length:
            raise ValueError(
                f"Not enough notes ({len(note_integers)}) "
                f"for sequence length {self.sequence_length}"
            )
        
        X = []
        y = []
        
        # Create sliding window sequences
        for i in range(len(note_integers) - self.sequence_length):
            # Input: sequence_length notes
            sequence_in = note_integers[i:i + self.sequence_length]
            # Output: the next note
            sequence_out = note_integers[i + self.sequence_length]
            
            X.append(sequence_in)
            y.append(sequence_out)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n🎼 Sequence Statistics:")
        print(f"   Number of sequences: {len(X)}")
        print(f"   Input shape: {X.shape}")
        print(f"   Output shape: {y.shape}")
        print(f"   Sequence length: {self.sequence_length}")
        
        return X, y
    
    def save_vocabulary(self, filepath: str) -> None:
        """
        Save vocabulary to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        if not self._vocab_built:
            raise ValueError("Vocabulary not built! Nothing to save.")
        
        vocab_data = {
            "note_to_int": self.note_to_int,
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"💾 Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """
        Load vocabulary from JSON file.
        
        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.note_to_int = vocab_data["note_to_int"]
        self.int_to_note = {int(v): k for k, v in self.note_to_int.items()}
        self.sequence_length = vocab_data["sequence_length"]
        self.vocab_size = vocab_data["vocab_size"]
        self._vocab_built = True
        
        print(f"📂 Vocabulary loaded: {self.vocab_size} notes")


# Quick test when run directly
if __name__ == "__main__":
    sample_notes = ['C4', 'E4', 'G4', 'C4', 'E4', 'G4', 'B4', 'C5'] * 50
    
    preprocessor = MusicPreprocessor(sequence_length=10)
    
    preprocessor.build_vocabulary(sample_notes)
    
    X, y = preprocessor.prepare_sequences(sample_notes)
    
    print(f"\nSample input sequence (as integers): {X[0]}")
    print(f"Sample input sequence (as notes): {preprocessor.integers_to_notes(X[0].tolist())}")
    print(f"Target output: {y[0]} ({preprocessor.int_to_note[y[0]]})")