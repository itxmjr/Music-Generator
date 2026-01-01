"""
Music Generation Module

This module handles:
- Loading trained model
- Generating new note sequences
- Converting sequences to MIDI files
- Temperature-based sampling for creativity control
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from music21 import stream, note, chord, instrument, tempo
import random

from src.model import MusicLSTM
from src.preprocessor import MusicPreprocessor


class MoodConfig:
    """Configuration for different musical moods."""
    
    MOODS = {
        "energetic": {"bpm": 140, "temperature": 1.2},
        "melancholic": {"bpm": 60, "temperature": 0.6},
        "ambient": {"bpm": 80, "temperature": 0.9},
        "classical": {"bpm": 90, "temperature": 0.7},
        "synthwave": {"bpm": 120, "temperature": 1.0},
        "jazz": {"bpm": 100, "temperature": 1.1},
        "rock": {"bpm": 130, "temperature": 1.1},
        "lofi": {"bpm": 70, "temperature": 0.8},
        "pop": {"bpm": 110, "temperature": 0.95},
        "hiphop": {"bpm": 95, "temperature": 1.0},
        "electro": {"bpm": 128, "temperature": 1.1},
        "blues": {"bpm": 70, "temperature": 1.0},
    }
    
    @classmethod
    def get_config(cls, mood: str) -> dict:
        """Get configuration for a mood, with defaults."""
        return cls.MOODS.get(mood.lower(), {"bpm": 100, "temperature": 1.0})


class MusicGenerator:
    """
    Generates new music using a trained LSTM model.
    
    The generator uses the trained model to predict notes one at a time,
    building up a complete musical sequence.
    
    Usage:
        generator = MusicGenerator('outputs/models/best_model.pt', 
                                   'outputs/models/vocabulary.json')
        notes = generator.generate(length=500)
        generator.save_midi(notes, 'outputs/generated/song.mid')
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the generator.
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary JSON file
            device: Device to run generation on (auto-detects if None)
        """
        # Auto-detect device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load vocabulary
        self.preprocessor = MusicPreprocessor()
        self.preprocessor.load_vocabulary(vocab_path)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
        print(f"🎵 Generator initialized!")
        print(f"   Device: {self.device}")
        print(f"   Vocabulary size: {self.preprocessor.vocab_size}")
    
    def _load_model(self, model_path: str) -> MusicLSTM:
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded MusicLSTM model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration from checkpoint
        config = checkpoint['model_config']
        
        # Create model with same configuration
        model = MusicLSTM(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            lstm_hidden_size=config['lstm_hidden_size'],
            num_lstm_layers=config['num_lstm_layers'],
            dropout=config['dropout']
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"📂 Model loaded from {model_path}")
        print(f"   Trained for {checkpoint['epoch'] + 1} epochs")
        print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
        
        return model
    
    def generate(
        self,
        length: int = 500,
        temperature: float = 1.0,
        seed_sequence: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate a sequence of notes.
        
        Args:
            length: Number of notes to generate
            temperature: Controls randomness (0.5=conservative, 1.5=creative)
            seed_sequence: Optional starting notes (uses random if None)
            
        Returns:
            List of generated note strings
            
        TEMPERATURE EXPLAINED:
        ----------------------
        Temperature adjusts the probability distribution:
        
        Low (0.2-0.5):  
        - Sharpens distribution (high probs higher, low probs lower)
        - More predictable, repetitive output
        - Safer, more "correct" sounding
        
        Medium (0.8-1.0):
        - Original distribution
        - Balanced creativity and structure
        
        High (1.2-2.0):
        - Flattens distribution (all options more equal)
        - More random, surprising output
        - More creative but might be chaotic
        """
        print(f"\n🎼 Generating {length} notes (temperature={temperature})...")
        
        # Get seed sequence
        if seed_sequence is None:
            seed_sequence = self._get_random_seed()
        
        # Convert seed to integers
        current_sequence = self.preprocessor.notes_to_integers(seed_sequence)
        
        # Ensure we have enough notes for input
        seq_length = self.preprocessor.sequence_length
        if len(current_sequence) < seq_length:
            raise ValueError(
                f"Seed sequence too short ({len(current_sequence)}), "
                f"need at least {seq_length}"
            )
        
        # Take last seq_length notes as starting point
        current_sequence = current_sequence[-seq_length:]
        
        # Generate notes one at a time
        generated_indices = []
        
        with torch.no_grad():  # No gradients needed for generation
            for i in range(length):
                # Prepare input tensor
                input_tensor = torch.LongTensor([current_sequence]).to(self.device)
                
                # Get prediction
                output, _ = self.model(input_tensor)
                
                # Apply temperature and convert to probabilities
                probabilities = F.softmax(output[0] / temperature, dim=0)
                
                # Sample from distribution
                next_index = self._sample_from_distribution(probabilities)
                
                # Store generated note
                generated_indices.append(next_index)
                
                # Update sequence for next prediction
                current_sequence = current_sequence[1:] + [next_index]
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"   Generated {i + 1}/{length} notes...")
        
        # Convert indices back to notes
        generated_notes = self.preprocessor.integers_to_notes(generated_indices)
        
        print(f"✅ Generated {len(generated_notes)} notes!")
        
        return generated_notes
    
    def _sample_from_distribution(
        self, 
        probabilities: torch.Tensor
    ) -> int:
        """
        Sample a note index from probability distribution.
        
        Uses multinomial sampling for randomness.
        
        Args:
            probabilities: Tensor of probabilities for each note
            
        Returns:
            Sampled note index
        """
        # Multinomial sampling: randomly select based on probabilities
        index = torch.multinomial(probabilities, 1).item()
        return index
    
    def _get_random_seed(self) -> List[str]:
        """
        Get a random seed sequence from vocabulary.
        
        Returns:
            List of random notes for seeding generation
        """
        vocab = list(self.preprocessor.note_to_int.keys())
        seed_length = self.preprocessor.sequence_length
        
        # Random selection of notes
        seed = random.choices(vocab, k=seed_length)
        
        print(f"   Using random seed: {seed[:5]}... ({seed_length} notes)")
        return seed
    
    def save_midi(
        self,
        notes: List[str],
        output_path: str,
        bpm: int = 120
    ) -> None:
        """
        Convert generated notes to a MIDI file.
        
        Args:
            notes: List of note strings (e.g., ['C4', 'E4.G4.B4', 'D5'])
            output_path: Path to save MIDI file
            bpm: Tempo in beats per minute
            
        HOW MIDI CONVERSION WORKS:
        --------------------------
        1. Create a music21 Stream (like a musical timeline)
        2. Add each note/chord at the right position
        3. Set instrument and tempo
        4. Export to MIDI file
        """
        print(f"\n🎹 Converting to MIDI...")
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a music21 stream
        midi_stream = stream.Stream()
        
        # Set instrument to Piano
        midi_stream.append(instrument.Piano())
        
        # Set tempo
        midi_stream.append(tempo.MetronomeMark(number=bpm))
        
        # Track position in the stream
        offset = 0.0
        note_duration = 0.5  # Quarter note duration
        
        for pattern in notes:
            # Check if it's a chord (contains dots)
            if '.' in pattern:
                # It's a chord - split by dots
                chord_notes = pattern.split('.')
                
                try:
                    # Create chord object
                    new_chord = chord.Chord(chord_notes)
                    new_chord.quarterLength = note_duration
                    new_chord.offset = offset
                    midi_stream.append(new_chord)
                except Exception as e:
                    # Skip invalid chords
                    pass
            else:
                # It's a single note
                try:
                    new_note = note.Note(pattern)
                    new_note.quarterLength = note_duration
                    new_note.offset = offset
                    midi_stream.append(new_note)
                except Exception as e:
                    # Skip invalid notes
                    pass
            
            offset += note_duration
        
        # Write to MIDI file
        midi_stream.write('midi', fp=str(output_path))
        
        print(f"💾 MIDI saved to: {output_path}")
        print(f"   Duration: ~{len(notes) * note_duration / 2:.1f} seconds")
        print(f"   Tempo: {bpm} BPM")
    
    def generate_and_save(
        self,
        output_path: str,
        length: int = 500,
        temperature: float = 1.0,
        bpm: int = 120
    ) -> List[str]:
        """
        Generate music and save directly to MIDI.
        
        Convenience method that combines generate() and save_midi().
        
        Args:
            output_path: Path to save MIDI file
            length: Number of notes to generate
            temperature: Creativity control
            bpm: Tempo in beats per minute
            
        Returns:
            List of generated notes
        """
        notes = self.generate(length=length, temperature=temperature)
        self.save_midi(notes, output_path, bpm=bpm)
        return notes


def generate_samples(
    model_path: str = "outputs/models/best_model.pt",
    vocab_path: str = "outputs/models/vocabulary.json",
    output_dir: str = "outputs/generated",
    num_samples: int = 3,
    length: int = 500
) -> None:
    """
    Generate multiple music samples with different temperatures.
    
    Creates samples at different temperature levels to showcase
    the effect of temperature on creativity.
    
    Args:
        model_path: Path to trained model
        vocab_path: Path to vocabulary file
        output_dir: Directory to save generated files
        num_samples: Number of samples per temperature
        length: Notes per sample
    """
    # Create generator
    generator = MusicGenerator(model_path, vocab_path)
    
    # Different temperatures to try
    temperatures = [0.5, 0.8, 1.0, 1.2]
    
    print("\n" + "="*60)
    print("🎵 GENERATING MUSIC SAMPLES")
    print("="*60)
    
    for temp in temperatures:
        for i in range(num_samples):
            filename = f"generated_temp{temp}_sample{i+1}.mid"
            output_path = Path(output_dir) / filename
            
            print(f"\n--- Temperature: {temp}, Sample: {i+1} ---")
            generator.generate_and_save(
                output_path=str(output_path),
                length=length,
                temperature=temp,
                bpm=120
            )
    
    print("\n" + "="*60)
    print(f"✅ All samples saved to {output_dir}/")
    print("="*60)


# Test/run when executed directly
if __name__ == "__main__":
    import sys
    
    # Check if model exists
    model_path = Path("outputs/models/best_model.pt")
    vocab_path = Path("outputs/models/vocabulary.json")
    
    if not model_path.exists():
        print("❌ No trained model found!")
        print("   Please run training first: python main.py")
        sys.exit(1)
    
    if not vocab_path.exists():
        print("❌ No vocabulary file found!")
        print("   Please run training first: python main.py")
        sys.exit(1)
    
    # Generate samples
    generate_samples(
        num_samples=2,
        length=300
    )