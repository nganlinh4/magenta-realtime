# Setting Up Real Magenta RT (AI Music Generation)

Currently, your Magenta RT backend is running in **mock mode**, which generates fake audio instead of real AI-generated music. This document explains how to enable the real Magenta RT system.

## Current Status

✅ **Mock System Active**: The backend is running and generating fake audio  
❌ **Real AI System**: Not active due to missing requirements

## Why You're Hearing Mock Audio

The real Magenta RT system requires:
1. **Google Cloud credentials** for downloading AI models
2. **GPU with sufficient memory** (recommended)
3. **Network connectivity** for model downloads

## How to Enable Real AI Music Generation

### Option 1: Set Up Google Cloud Credentials (Recommended)

1. **Install Google Cloud CLI**:
   ```bash
   # On Ubuntu/Debian
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   
   # On macOS
   brew install google-cloud-sdk
   ```

2. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth application-default login
   ```
   This will open a browser window for authentication.

3. **Restart the backend**:
   ```bash
   # Stop the current backend (Ctrl+C)
   # Then restart
   npm run dev
   ```

### Option 2: Use Pre-downloaded Models (Advanced)

If you have access to pre-downloaded Magenta RT model files:

1. Set the checkpoint directory:
   ```bash
   export MAGENTA_RT_CHECKPOINT_DIR="/path/to/your/models"
   ```

2. Restart the backend

### Option 3: Use Different Model Size

If you have memory constraints, try the smaller model:

```bash
export MAGENTA_RT_MODEL="base"  # instead of "large"
npm run dev
```

## Environment Variables

You can configure the Magenta RT system using these environment variables:

- `MAGENTA_RT_DEVICE`: Device to use (`gpu` or `tpu:v2-8`)
- `MAGENTA_RT_MODEL`: Model size (`base` or `large`)
- `MAGENTA_RT_CHECKPOINT_DIR`: Custom checkpoint directory

Example:
```bash
export MAGENTA_RT_DEVICE="gpu"
export MAGENTA_RT_MODEL="large"
npm run dev
```

## Troubleshooting

### "Your default credentials were not found"
- Follow Option 1 above to set up Google Cloud credentials

### "Unsupported device: cpu"
- The real Magenta RT system requires GPU or TPU
- Make sure you have a CUDA-compatible GPU
- Check that JAX can detect your GPU: `python -c "import jax; print(jax.devices())"`

### Out of Memory Errors
- Try using the smaller model: `export MAGENTA_RT_MODEL="base"`
- Close other GPU-intensive applications
- Consider using a machine with more GPU memory

### Network/Download Issues
- Ensure stable internet connection
- Check firewall settings
- Models are large (several GB) and may take time to download

## Verifying Real System is Active

Once properly configured, you should see:

```
INFO:main:✅ REAL Magenta RT system initialized successfully!
INFO:main:🎵 You will now hear AI-generated music!
```

Instead of:

```
INFO:main:Mock Magenta RT system initialized successfully
```

## Health Check

You can check the system status at: http://localhost:8000/api/health

The `system_type` field will show:
- `"mock"`: Fake audio generation
- `"real"`: Real AI music generation
- `"none"`: System failed to initialize

## Need Help?

If you continue to have issues:
1. Check the backend logs for specific error messages
2. Ensure your system meets the hardware requirements
3. Verify your Google Cloud setup is correct
4. Consider starting with the smaller "base" model first
