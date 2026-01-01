"""
Data Loader Module for MIDI Processing

This module handles:
- Finding MIDI files in directories
- Parsing MIDI files using music21
- Extracting notes and chords
"""

from pathlib import Path
from typing import List, Optional, Union
from music21 import converter, instrument, note, chord
from tqdm import tqdm  # Progress bars


class MidiDataLoader:
    """
    Loads and parses MIDI files from a directory.
    
    Usage:
        loader = MidiDataLoader("data/midi")
        notes = loader.load_all_notes()
    """
    
    MIDI_EXTENSIONS = {".mid", ".midi"}
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing MIDI files
        """
        self.data_dir = Path(data_dir)
        
        # Validate directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        
        # Find all MIDI files (including subdirectories)
        self.midi_files = self._find_midi_files()
        print(f"📁 Found {len(self.midi_files)} MIDI files in {self.data_dir}")
    
    def _find_midi_files(self) -> List[Path]:
        """
        Find all MIDI files recursively.
        
        Returns:
            List of paths to MIDI files
        """
        midi_files = []
        
        for ext in self.MIDI_EXTENSIONS:
            # rglob = recursive glob (searches subdirectories)
            midi_files.extend(self.data_dir.rglob(f"*{ext}"))
        
        return sorted(midi_files)  # Sorted for reproducibility
    
    def parse_midi_file(self, file_path: Path) -> Optional[List[str]]:
        """
        Extract notes and chords from a single MIDI file.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            List of note/chord strings, or None if parsing fails
        """
        try:
            # Parse the MIDI file
            midi_data = converter.parse(file_path)
            
            notes_list = []
            
            # Try to partition by instrument
            parts = instrument.partitionByInstrument(midi_data)
            
            if parts:
                # Multi-instrument: take the first part (usually melody)
                notes_to_parse = parts.parts[0].recurse()
            else:
                # Single track: use all notes
                notes_to_parse = midi_data.flat.notes
            
            # Extract notes and chords
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    # Single note: store pitch string (e.g., 'C4')
                    notes_list.append(str(element.pitch))
                    
                elif isinstance(element, chord.Chord):
                    # Chord: join pitches with dots (e.g., 'C4.E4.G4')
                    chord_str = '.'.join(str(p) for p in element.pitches)
                    notes_list.append(chord_str)
            
            return notes_list if notes_list else None
            
        except Exception as e:
            # Log the error but don't crash
            print(f"⚠️  Failed to parse {file_path.name}: {e}")
            return None
    
    def load_all_notes(self, max_files: Optional[int] = None) -> List[str]:
        """
        Load notes from all MIDI files.
        
        Args:
            max_files: Optional limit on number of files to process
                      (useful for testing with smaller dataset)
        
        Returns:
            Combined list of all notes and chords
        """
        all_notes = []
        
        files_to_process = self.midi_files[:max_files] if max_files else self.midi_files
        
        # tqdm gives us a progress bar
        print(f"🎵 Processing {len(files_to_process)} MIDI files...")
        
        successful = 0
        failed = 0
        
        for file_path in tqdm(files_to_process, desc="Loading MIDI"):
            notes = self.parse_midi_file(file_path)
            
            if notes:
                all_notes.extend(notes)
                successful += 1
            else:
                failed += 1
        
        # Summary statistics
        print(f"\n✅ Successfully processed: {successful} files")
        print(f"❌ Failed to process: {failed} files")
        print(f"🎹 Total notes/chords extracted: {len(all_notes)}")
        
        return all_notes
    
    def get_sample_notes(self, n_files: int = 5) -> List[str]:
        """
        Load notes from a small sample of files.
        
        Args:
            n_files: Number of files to sample
            
        Returns:
            Notes from sampled files
        """
        return self.load_all_notes(max_files=n_files)


if __name__ == "__main__":
    loader = MidiDataLoader("data/midi")
    sample = loader.get_sample_notes(n_files=3)
    print(f"\nSample notes: {sample[:20]}")