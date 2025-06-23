# Magenta RT with ShadCN UI Setup

This project combines Magenta RT (real-time music generation) with a modern ShadCN UI frontend.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA-compatible GPU (optional, for better performance)

### 1. Python Virtual Environment with JAX CUDA

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install JAX with CUDA support
pip install jax[cuda12] jaxlib

# Install additional dependencies
pip install fastapi uvicorn websockets soundfile
pip install absl-py chex flax gin-config numpy resampy tqdm
```

### 2. Backend Setup

```bash
# Start the FastAPI backend
cd backend
source ../venv/bin/activate
python main.py
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
# Install dependencies and start the frontend
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 🎵 Features

### Backend (FastAPI)
- **REST API** for music generation
- **Style embedding** from text prompts
- **Real-time audio generation** using Magenta RT
- **WebSocket support** (ready for streaming)
- **CORS enabled** for frontend communication
- **Health monitoring** and error handling
- **State management** for continuous generation

### Frontend (Next.js + ShadCN UI) - Real-time Interface
- **Multiple style prompts** with individual weight sliders
- **Real-time parameter adjustment** (temperature, top-k, guidance)
- **Weighted style mixing** - blend multiple styles in real-time
- **Start/Stop/Reset controls** for streaming generation
- **Beautiful ShadCN components** with Tailwind CSS
- **Live status monitoring** with connection indicators
- **Modern responsive design** with gradient backgrounds

## 🎛️ API Endpoints

### Health Check
```bash
GET /api/health
```

### Style Embedding
```bash
POST /api/embed-style
{
  "text_or_audio": "upbeat jazz with saxophone",
  "weight": 1.0
}
```

### Generate Music Chunk
```bash
POST /api/generate-chunk
{
  "style_embedding": [...],
  "temperature": 1.1,
  "topk": 40,
  "guidance_weight": 5.0
}
```

### WebSocket (Real-time)
```bash
WS /ws/generate
```

## 🎨 UI Components

The frontend uses ShadCN UI components:
- **Card** - Layout containers
- **Button** - Interactive controls
- **Input/Textarea** - Text input
- **Slider** - Parameter controls
- **Progress** - Generation progress
- **Badge** - Status indicators

## 🔧 Configuration

### Backend Configuration
- **Host**: `0.0.0.0:8000`
- **CORS**: Enabled for `localhost:3000`
- **Mock Mode**: Currently using MockMagentaRT for faster startup
- **GPU Support**: Automatically detected

### Frontend Configuration
- **API Base URL**: `http://localhost:8000`
- **Auto-reconnect**: Health checks every 30 seconds
- **Audio Format**: WAV (base64 encoded)

## 🚀 Production Deployment

### Backend
1. Replace MockMagentaRT with full MagentaRT system
2. Install complete dependencies: `pip install -e .[gpu]`
3. Configure production WSGI server (gunicorn)
4. Set up proper CORS origins
5. Add authentication if needed

### Frontend
1. Build for production: `npm run build`
2. Deploy to Vercel, Netlify, or similar
3. Update API base URL for production
4. Configure environment variables

## 🎵 Usage - Real-time Music Generation

1. **Start both backend and frontend**
2. **Open** `http://localhost:3000` in your browser
3. **Set up style prompts**:
   - Enter multiple style descriptions (e.g., "synthwave", "flamenco guitar")
   - Adjust individual weight sliders for each prompt
   - Mix styles by setting multiple weights > 0
4. **Adjust sampling parameters**:
   - **Temperature**: Controls chaos/creativity (0.0-4.0)
   - **Top-K**: Vocabulary filtering (0-1024)
   - **Guidance**: Style adherence strength (0.0-10.0)
5. **Start real-time generation**:
   - Click "Start" to begin streaming
   - Adjust sliders in real-time to change the music
   - Watch the live status indicators
6. **Control playback**:
   - Use "Stop" to halt generation
   - Use "Reset" to clear state and start fresh

## 🔍 Troubleshooting

### JAX CUDA Issues
- Ensure CUDA drivers are installed
- Check GPU memory availability
- Try CPU-only mode if GPU fails

### Backend Connection Issues
- Verify backend is running on port 8000
- Check CORS configuration
- Ensure no firewall blocking

### Frontend Build Issues
- Clear node_modules and reinstall
- Check Node.js version compatibility
- Verify ShadCN UI installation

## 📁 Project Structure

```
magenta-realtime/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   └── page.tsx     # Main page
│   │   ├── components/
│   │   │   ├── ui/          # ShadCN components
│   │   │   └── MagentaRTStudio.tsx
│   │   └── hooks/
│   │       └── useAudioGeneration.ts
│   ├── package.json
│   └── tailwind.config.js
├── venv/                    # Python virtual environment
├── magenta_rt/             # Magenta RT source code
└── SETUP.md                # This file
```

## 🎯 Next Steps

1. **Integrate full Magenta RT** (replace mock system)
2. **Add WebSocket streaming** for real-time generation
3. **Implement audio visualization** (waveform, spectrum)
4. **Add user authentication** and session management
5. **Create audio library** for saving/loading generations
6. **Add more style controls** (tempo, key, instruments)
7. **Implement collaborative features** (sharing, remixing)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project combines:
- **Magenta RT**: Apache 2.0 License (code) + CC BY 4.0 (models)
- **ShadCN UI**: MIT License
- **Custom Code**: MIT License

---

**Enjoy creating music with AI! 🎵✨**
