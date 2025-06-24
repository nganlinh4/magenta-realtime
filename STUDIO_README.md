# 🎵 Magenta RT Studio

A real-time AI music generation studio built with Magenta RT, FastAPI, and Next.js with ShadCN UI.

## ✨ Features

- **Real-time AI Music Generation** using Google's Magenta RT
- **Interactive Web Interface** with real-time sliders and controls
- **Style Embedding** from text prompts or audio files
- **Chunk-by-chunk Generation** with seamless crossfading
- **Modern UI** built with Next.js and ShadCN components
- **FastAPI Backend** with WebSocket support for real-time streaming

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm run setup
```

### 2. Start Development Servers
```bash
npm run dev
```

This will automatically:
- ✅ Check Magenta RT requirements
- ✅ Set up Google Cloud credentials (if needed)
- ✅ Configure the real AI music system
- ✅ Start both frontend and backend servers

### 3. Open Your Browser
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/health

## 🎛️ Alternative Commands

- **Quick start** (skip Magenta RT setup): `npm run dev:quick`
- **Setup Magenta RT only**: `npm run setup:magenta-rt`

## 🔧 Configuration

The system automatically creates a `.env` file with these settings:

```bash
# Magenta RT Configuration
MAGENTA_RT_DEVICE=gpu          # Device to use (gpu or tpu:v2-8)
MAGENTA_RT_MODEL=large         # Model size (base or large)

# Mock mode has been completely removed
```

## 🎵 AI Music Generation

### Real AI Music (Only Option)
- **Requirements**: Google Cloud credentials, GPU/TPU, compatible JAX/XLA versions
- **Setup**: Run `npm run setup:magenta-rt` and follow the prompts
- **Result**: High-quality AI-generated music
- **Note**: System will fail to start if requirements are not met (no fallback)

## 🛠️ System Requirements

### For Real AI Music:
- **GPU**: CUDA-compatible GPU with sufficient memory
- **Credentials**: Google Cloud authentication
- **Storage**: Several GB for model downloads
- **Memory**: 8GB+ RAM recommended

### For Development/Testing:
- **CPU**: Any modern processor
- **Memory**: 4GB+ RAM
- **Storage**: 1GB for dependencies

## 📊 Health Check

Check system status at: http://localhost:8000/api/health

Response includes:
- `status`: "healthy" or "degraded"
- `magenta_rt_loaded`: true/false
- `gpu_available`: true/false
- `system_type`: "real" or "none"
- `jax_version`: JAX version info

## 🔍 Troubleshooting

### System fails to start
1. Check JAX/XLA version compatibility
2. Run `npm run setup:magenta-rt`
3. Follow Google Cloud authentication prompts
4. Ensure sufficient GPU memory
5. Restart with `npm run dev`

### GPU not detected
- Install CUDA drivers
- Check: `python -c "import jax; print(jax.devices())"`

### Out of memory errors
- Try smaller model: `MAGENTA_RT_MODEL=base`
- Close other GPU applications
- Use machine with more GPU memory

### Network/download issues
- Check internet connection
- Verify Google Cloud credentials
- Models are large (several GB)

## 📁 Project Structure

```
magenta-realtime/
├── frontend/              # Next.js frontend with ShadCN UI
├── backend/              # FastAPI backend
├── magenta_rt/           # Magenta RT source code
├── setup-magenta-rt.sh   # Automatic setup script
├── .env                  # Configuration (auto-generated)
└── package.json          # NPM scripts
```

## 🎯 Usage

1. **Start the application**: `npm run dev`
2. **Enter a style prompt**: e.g., "jazz piano", "electronic dance"
3. **Adjust parameters**: Temperature, guidance, etc.
4. **Click "Start"**: Begin real-time generation
5. **Listen**: Enjoy AI-generated music!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with real Magenta RT system
5. Submit a pull request

## 📄 License

This project combines:
- **Codebase**: Apache 2.0 License
- **Magenta RT Models**: Creative Commons Attribution 4.0

See individual license files for details.

## 🆘 Support

- **Setup Issues**: Run `npm run setup:magenta-rt`
- **API Issues**: Check `http://localhost:8000/api/health`
- **Frontend Issues**: Check browser console
- **Model Issues**: See `REAL_MAGENTA_RT_SETUP.md`

---

Built with ❤️ using Magenta RT, FastAPI, Next.js, and ShadCN UI
