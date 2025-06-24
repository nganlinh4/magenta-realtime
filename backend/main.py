#!/usr/bin/env python3
"""
FastAPI backend for Magenta RT music generation.
Provides REST API and WebSocket endpoints for real-time music generation.
"""

import asyncio
import base64
import io
import json
import logging
import os
from typing import Optional, Dict, Any, List
import traceback

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading .env file
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Magenta RT API",
    description="Real-time music generation API using Magenta RT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the Magenta RT system
magenta_rt_system = None
current_state = None

# Pydantic models for API requests/responses
class StyleEmbedRequest(BaseModel):
    text_or_audio: str
    weight: float = 1.0

class GenerateChunkRequest(BaseModel):
    style_embedding: Optional[List[float]] = None
    seed: Optional[int] = None
    temperature: float = 1.1
    topk: int = 40
    guidance_weight: float = 5.0

class GenerateChunkResponse(BaseModel):
    audio_data: str  # base64 encoded audio
    sample_rate: int
    chunk_index: int
    success: bool
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    magenta_rt_loaded: bool
    gpu_available: bool
    jax_version: str
    system_type: str  # "real" or "none"

def initialize_magenta_rt():
    """Initialize the Magenta RT system."""
    global magenta_rt_system

    try:
        logger.info("🚀 Initializing Magenta RT system...")
        logger.info("📥 This may take several minutes to download models and initialize...")

        # Import the real Magenta RT system
        from magenta_rt.system import MagentaRT

        # Configuration from environment variables
        device = os.getenv('MAGENTA_RT_DEVICE', 'gpu')  # 'gpu' or 'tpu:v2-8' (cpu not supported)
        model_tag = os.getenv('MAGENTA_RT_MODEL', 'base')  # 'base' or 'large'

        logger.info(f"🎯 Loading Magenta RT model: {model_tag} on device: {device}")

        # Use HuggingFace for easier model downloads (no Google Cloud required)
        os.environ['MAGENTA_RT_ASSET_SOURCE'] = 'hf'
        logger.info("🤗 Using HuggingFace for model downloads (no Google Cloud required)")

        # Initialize the real Magenta RT system
        magenta_rt_system = MagentaRT(
            tag=model_tag,
            device=device,
            lazy=False,  # Load immediately for better startup feedback
            skip_cache=False  # Use cache for faster subsequent loads
        )

        logger.info("✅ Magenta RT system initialized successfully!")
        logger.info("🎵 Ready to generate AI music!")

    except ImportError as e:
        logger.error(f"❌ Failed to import Magenta RT: {e}")
        logger.error("💡 Make sure magenta-rt is properly installed:")
        logger.error("   pip install 'git+https://github.com/magenta/magenta-realtime#egg=magenta_rt[gpu]'")
        raise RuntimeError(f"Magenta RT import failed: {e}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Magenta RT: {e}")
        logger.error("💡 This might be due to:")
        logger.error("   - Incompatible JAX/XLA versions (try updating)")
        logger.error("   - Insufficient GPU memory")
        logger.error("   - Network issues during model download")
        logger.error("   - Missing model files")
        logger.error(f"   - Error details: {str(e)}")
        raise RuntimeError(f"Magenta RT initialization failed: {e}")

def audio_to_base64(audio_samples: np.ndarray, sample_rate: int) -> str:
    """Convert audio samples to base64 encoded WAV."""
    try:
        import soundfile as sf
        
        # Create a BytesIO buffer
        buffer = io.BytesIO()
        
        # Write audio to buffer as WAV
        sf.write(buffer, audio_samples, sample_rate, format='WAV')
        
        # Get the bytes and encode as base64
        buffer.seek(0)
        audio_bytes = buffer.read()
        return base64.b64encode(audio_bytes).decode('utf-8')
        
    except ImportError:
        # Fallback: create a simple WAV header manually
        logger.warning("soundfile not available, using basic WAV encoding")
        
        # Normalize audio to 16-bit range
        if audio_samples.dtype != np.int16:
            audio_samples = (audio_samples * 32767).astype(np.int16)
        
        # Simple WAV header creation
        audio_bytes = audio_samples.tobytes()
        return base64.b64encode(audio_bytes).decode('utf-8')

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    logger.info("Starting Magenta RT API server...")
    initialize_magenta_rt()

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        import jax
        jax_version = jax.__version__
        gpu_available = len([d for d in jax.devices() if d.platform == 'gpu']) > 0
    except ImportError:
        jax_version = "not available"
        gpu_available = False

    # Determine system type
    if magenta_rt_system is None:
        system_type = "none"
    else:
        system_type = "real"

    return HealthResponse(
        status="healthy" if magenta_rt_system is not None else "degraded",
        magenta_rt_loaded=magenta_rt_system is not None,
        gpu_available=gpu_available,
        jax_version=jax_version,
        system_type=system_type
    )

@app.post("/api/embed-style")
async def embed_style(request: StyleEmbedRequest):
    """Embed a text or audio style prompt."""
    if magenta_rt_system is None:
        raise HTTPException(status_code=503, detail="Magenta RT system not available")
    
    try:
        # Embed the style
        style_embedding = magenta_rt_system.embed_style(request.text_or_audio)
        
        # Convert to list for JSON serialization
        if isinstance(style_embedding, np.ndarray):
            style_embedding = style_embedding.tolist()
        
        return {
            "style_embedding": style_embedding,
            "weight": request.weight,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Style embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Style embedding failed: {str(e)}")

@app.post("/api/generate-chunk", response_model=GenerateChunkResponse)
async def generate_chunk(request: GenerateChunkRequest):
    """Generate a single chunk of audio."""
    global current_state

    if magenta_rt_system is None:
        raise HTTPException(status_code=503, detail="Magenta RT system not available")

    try:
        # Convert style embedding back to numpy array if provided
        style = None
        if request.style_embedding:
            style = np.array(request.style_embedding, dtype=np.float32)

        # Generate chunk with parameters
        audio_chunk, new_state = magenta_rt_system.generate_chunk(
            state=current_state,
            style=style,
            seed=request.seed,
            temperature=request.temperature,
            topk=request.topk,
            guidance_weight=request.guidance_weight
        )

        # Update state
        current_state = new_state

        # Convert audio to base64
        audio_data = audio_to_base64(audio_chunk.samples, audio_chunk.sample_rate)

        return GenerateChunkResponse(
            audio_data=audio_data,
            sample_rate=audio_chunk.sample_rate,
            chunk_index=current_state.chunk_index if current_state else 0,
            success=True
        )

    except Exception as e:
        logger.error(f"Chunk generation failed: {e}")
        return GenerateChunkResponse(
            audio_data="",
            sample_rate=48000,
            chunk_index=0,
            success=False,
            error=str(e)
        )

@app.post("/api/reset-state")
async def reset_state():
    """Reset the generation state."""
    global current_state
    current_state = None
    return {"success": True, "message": "Generation state reset"}

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time music generation."""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "generate":
                # Generate chunk based on message parameters
                request_data = message.get("data", {})
                
                try:
                    # Create request object
                    request = GenerateChunkRequest(**request_data)
                    
                    # Generate chunk (reuse the REST endpoint logic)
                    response = await generate_chunk(request)
                    
                    # Send response back
                    await websocket.send_text(json.dumps({
                        "type": "chunk",
                        "data": response.dict()
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": str(e)}
                    }))
            
            elif message.get("type") == "reset":
                # Reset the generation state
                global current_state
                current_state = None
                await websocket.send_text(json.dumps({
                    "type": "reset_complete",
                    "data": {"success": True}
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
