# 🎵 Music Generator

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15+-000000.svg?style=for-the-badge&logo=nextdotjs&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

A powerful AI-based music composition tool featuring a robust FastAPI backend and a modern Next.js frontend. This project uses LSTM neural networks to create unique musical pieces and offers a sleek, mood-based generation UI.

## ✨ Features

- **Mood-Based Generation**: Select from moods like "Energetic", "Melancholic", or "Cyberpunk" to shape the music.
- **Modern Web Interface**: Beautiful, glassmorphic UI built with Next.js, TailwindCSS, and Shadcn UI.
- **Deep Learning Core**: Utilizes robust LSTM neural networks to understand and predict musical patterns.
- **Real-Time Generation**: Powered by FastAPI for quick, asynchronous music creation.
- **MIDI Export**: Instantly download your creations as standard MIDI files.

## 🚀 Demo

*(Add a screenshot of your new UI here)*

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/), [PyTorch](https://pytorch.org/), [music21](https://web.mit.edu/music21/).
- **Frontend**: [Next.js](https://nextjs.org/), [TailwindCSS](https://tailwindcss.com/), [Shadcn UI](https://ui.shadcn.com/), [Tone.js](https://tonejs.github.io/).
- **Deployment**: [Docker](https://www.docker.com/), [Hugging Face Spaces](https://huggingface.co/spaces).

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/itxmjr/Music-Generator.git
cd Music-Generator
```

### 2. Backend Setup
The backend handles the AI model and music generation.
```bash
cd backend
# Using uv (recommended)
uv sync
# Or using pip
pip install -r requirements.txt
```

### 3. Frontend Setup
The modern interface built with Next.js.
```bash
cd ../frontend
npm install
# or
bun install
```

## 🎮 Usage

### 🌐 Running the Application

1. **Start the Backend**:
   ```bash
   cd backend
   uv run uvicorn main:app --host 0.0.0.0 --port 8000
   ```
2. **Start the Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
3. Open [http://localhost:3000](http://localhost:3000) (Next.js) and enjoy!

### 💻 CLI Usage (Backend)
- **Train the Model**: `python main.py train`
- **Generate Music**: `python main.py generate`

## 📂 Project Structure

```
Music-Generator/
├── backend/             # FastAPI Backend
│   ├── data/            # Training datasets
│   ├── src/             # Core logic (model, train, generate)
│   ├── main.py          # Entry point
│   └── outputs/         # Saved models & generations
├── frontend/            # Next.js Frontend (React 19)
│   ├── app/             # Next.js App Router
│   ├── components/      # UI Components
│   └── public/          # Static assets
├── neon-muse-ai/        # Legacy Vite Frontend (Deprecated)
└── README.md            # Root documentation
```
## 🤝 Contributing

Contributions are welcome!

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by M Jawad ur Rehman.</sub>
</div>