from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import time

from src.generate import MusicGenerator, MoodConfig

# --- Configuration ---
MODEL_PATH = "outputs/models/best_model.pt"
VOCAB_PATH = "outputs/models/vocabulary.json"
OUTPUT_DIR = Path("outputs/generated")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize App ---
app = FastAPI(
    title="AI Music Generator API",
    description="A deep learning powered MIDI music generator.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production (e.g., Vercel URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if model exists
if not Path(MODEL_PATH).exists() or not Path(VOCAB_PATH).exists():
    logger.error("Model files not found! Please run 'python main.py train' first.")

# Initialize the generator globally to load model once
try:
    generator = MusicGenerator(MODEL_PATH, VOCAB_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    generator = None

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Routes ---

@app.get("/")
async def read_root():
    """API Root - Welcome message."""
    return {
        "message": "AI Music Generator API is running",
        "docs": "/docs",
        "health": "/health",
        "generate": "/generate"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and HF Spaces."""
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "timestamp": time.time()
    }

@app.get("/generate")
async def generate_music(
    mood: str = "energetic",
    temperature: float = None,
    bpm: int = None
):
    """
    Generate music based on mood or custom parameters.
    Returns a MIDI file download.
    """
    if not generator:
        raise HTTPException(status_code=503, detail="Model not initialized. Please check server logs.")
    
    try:
        # Get mood configuration
        config = MoodConfig.get_config(mood)
        
        # Override with custom parameters if provided
        gen_temperature = temperature if temperature is not None else config["temperature"]
        gen_bpm = bpm if bpm is not None else config["bpm"]
        
        # Validation
        gen_temperature = max(0.1, min(2.0, gen_temperature))
        gen_bpm = max(40, min(240, gen_bpm))
        
        logger.info(f"Generating for mood '{mood}' (applied: temp={gen_temperature}, bpm={gen_bpm})")
        
        # Create unique filename
        filename = f"generated_{mood}_{int(time.time())}.mid"
        output_path = OUTPUT_DIR / filename
        
        # Generate music
        generator.generate_and_save(
            output_path=str(output_path),
            length=300,  # Reasonable length for web generation
            temperature=gen_temperature,
            bpm=gen_bpm
        )
        
        if not output_path.exists():
             raise HTTPException(status_code=500, detail="Failed to create MIDI file")
             
        # Return file download
        return FileResponse(
            path=output_path,
            filename=filename,
            media_type="audio/midi"
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Server check or other logic can go here

