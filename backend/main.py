"""
Music Generator - Main Entry Point

Usage:
    python main.py train      # Train the model
    python main.py generate   # Generate music
"""

import sys
from pathlib import Path

from src.dataloader import MidiDataLoader
from src.preprocessor import MusicPreprocessor
from src.model import create_model
from src.train import MusicTrainer, prepare_data_loaders
from src.generate import MusicGenerator
from src.utils import set_seed, print_section


# ============== CONFIGURATION ==============
# Adjust these parameters as needed

CONFIG = {
    # Data settings
    "data_dir": "data/midi",
    "max_files": 10,           # Number of MIDI files to use
    
    # Model settings
    "sequence_length": 100,     # Notes to look back
    
    # Training settings
    "batch_size": 64,
    "epochs": 2,
    "learning_rate": 0.001,
    "patience": 15,             # Early stopping after this many epochs
    
    # Generation settings
    "generate_length": 500,     # Notes to generate
    "temperatures": [0.5, 0.8, 1.0, 1.2],
    "num_samples": 1,         # Samples per temperature
    "bpm": 120,
    
    # Paths
    "model_dir": "outputs/models",
    "generated_dir": "outputs/generated",
}


def train() -> None:
    """Train the music generation model."""
    
    set_seed(42)
    
    # Step 1: Load Data
    print_section("STEP 1: Loading MIDI Data")
    loader = MidiDataLoader(CONFIG["data_dir"])
    notes = loader.load_all_notes(max_files=CONFIG["max_files"])
    
    if len(notes) < CONFIG["sequence_length"] + 1:
        print(f"❌ Not enough notes ({len(notes)})")
        sys.exit(1)
    
    # Step 2: Preprocess
    print_section("STEP 2: Preprocessing Data")
    preprocessor = MusicPreprocessor(sequence_length=CONFIG["sequence_length"])
    preprocessor.build_vocabulary(notes, min_count=2)
    X, y = preprocessor.prepare_sequences(notes)
    
    # Save vocabulary
    vocab_path = Path(CONFIG["model_dir"]) / "vocabulary.json"
    preprocessor.save_vocabulary(str(vocab_path))
    
    # Step 3: Create Model
    print_section("STEP 3: Creating Model")
    model, device = create_model(vocab_size=preprocessor.vocab_size)
    
    # Step 4: Prepare Data
    print_section("STEP 4: Preparing Data Loaders")
    train_loader, val_loader = prepare_data_loaders(
        X, y, 
        batch_size=CONFIG["batch_size"]
    )
    
    # Step 5: Train
    print_section("STEP 5: Training Model")
    trainer = MusicTrainer(
        model, 
        device, 
        learning_rate=CONFIG["learning_rate"],
        checkpoint_dir=CONFIG["model_dir"]
    )
    trainer.train(
        train_loader, 
        val_loader, 
        epochs=CONFIG["epochs"],
        early_stopping_patience=CONFIG["patience"]
    )
    
    print_section("✅ Training Complete!")
    print(f"   Model saved to: {CONFIG['model_dir']}/best_model.pt")
    print(f"\n   Next step: python main.py generate")


def generate() -> None:
    """Generate music using trained model."""
    
    model_path = Path(CONFIG["model_dir"]) / "best_model.pt"
    vocab_path = Path(CONFIG["model_dir"]) / "vocabulary.json"
    
    # Check if model exists
    if not model_path.exists():
        print("❌ No trained model found!")
        print("   Run training first: python main.py train")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(CONFIG["generated_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    print_section("Initializing Generator")
    generator = MusicGenerator(str(model_path), str(vocab_path))
    
    # Generate samples
    print_section("Generating Music")
    
    for temp in CONFIG["temperatures"]:
        for i in range(CONFIG["num_samples"]):
            filename = f"generated_temp{temp}_sample{i+1}.mid"
            output_path = output_dir / filename
            
            print(f"\n🎵 Generating: {filename}")
            generator.generate_and_save(
                output_path=str(output_path),
                length=CONFIG["generate_length"],
                temperature=temp,
                bpm=CONFIG["bpm"]
            )
    
    print_section("✅ Generation Complete!")
    print(f"   Files saved to: {CONFIG['generated_dir']}/")
    print(f"\n   To play MIDI files:")
    print(f"   - Use VLC or any MIDI player")
    print(f"   - Or upload to: https://signal.vercel.app/")


def show_help() -> None:
    """Show usage instructions."""
    
    help_text = """
╔══════════════════════════════════════════════════════════════════════╗
║                    🎵 MUSIC GENERATION WITH AI 🎵                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  USAGE:                                                              ║
║  ──────                                                              ║
║    python main.py train      Train the model on MIDI data            ║
║    python main.py generate   Generate new music                      ║
║                                                                      ║
║  WORKFLOW:                                                           ║
║  ─────────                                                           ║
║    1. Place MIDI files in data/midi/                                 ║
║    2. Run: python main.py train                                      ║
║    3. Run: python main.py generate                                   ║
║    4. Find generated MIDI files in outputs/generated/                ║
║                                                                      ║
║  CONFIGURATION:                                                      ║
║  ──────────────                                                      ║
║    Edit CONFIG dictionary in main.py to adjust:                      ║
║    - max_files: Number of training files                             ║
║    - epochs: Training iterations                                     ║
║    - temperatures: Creativity levels for generation                  ║
║    - generate_length: Number of notes to generate                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(help_text)


def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "train":
        train()
    elif command == "generate":
        generate()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    main()