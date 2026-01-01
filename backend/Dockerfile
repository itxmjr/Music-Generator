# Use Python 3.10 slim
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Install system dependencies for music21 and MIDI
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libjack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user and set working directory
RUN useradd -m -u 1000 user
WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy only requirements first for caching
COPY pyproject.toml .
# If uv.lock exists, copy it too
COPY uv.lock* ./

# Install dependencies using uv
RUN uv pip install --system --no-cache -r pyproject.toml || \
    pip install --no-cache-dir torch numpy music21 tqdm matplotlib fastapi uvicorn python-multipart

# Copy project files
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p outputs/models outputs/generated data/midi \
    && chown -R user:user /app

# Switch to non-root user
USER user

# Expose HF Spaces port
EXPOSE 7860

# Run the application
# We use uvicorn to serve the FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
