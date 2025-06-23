#!/usr/bin/env python3
"""
Complete setup test for Magenta RT with ShadCN UI.
Tests the backend API and verifies the system is working correctly.
"""

import requests
import json
import base64
import time
import sys

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {data['status']}")
            print(f"   Magenta RT loaded: {data['magenta_rt_loaded']}")
            print(f"   GPU available: {data['gpu_available']}")
            print(f"   JAX version: {data['jax_version']}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_style_embedding():
    """Test style embedding endpoint."""
    print("\n🎨 Testing style embedding...")
    try:
        payload = {
            "text_or_audio": "upbeat jazz with saxophone",
            "weight": 1.0
        }
        response = requests.post(f"{API_BASE_URL}/api/embed-style", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                embedding = data.get('style_embedding', [])
                print(f"✅ Style embedding successful!")
                print(f"   Embedding dimension: {len(embedding)}")
                print(f"   Sample values: {embedding[:5]}...")
                return embedding
            else:
                print(f"❌ Style embedding failed: {data.get('error', 'Unknown error')}")
                return None
        else:
            print(f"❌ Style embedding failed with status {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Style embedding failed: {e}")
        return None

def test_music_generation(style_embedding=None):
    """Test music generation endpoint."""
    print("\n🎵 Testing music generation...")
    try:
        payload = {
            "temperature": 1.1,
            "topk": 40,
            "guidance_weight": 5.0
        }
        if style_embedding:
            payload["style_embedding"] = style_embedding
            
        response = requests.post(f"{API_BASE_URL}/api/generate-chunk", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                audio_data = data.get('audio_data', '')
                sample_rate = data.get('sample_rate', 0)
                chunk_index = data.get('chunk_index', 0)
                print(f"✅ Music generation successful!")
                print(f"   Sample rate: {sample_rate} Hz")
                print(f"   Chunk index: {chunk_index}")
                print(f"   Audio data length: {len(audio_data)} characters")
                
                # Try to decode the base64 audio data
                try:
                    audio_bytes = base64.b64decode(audio_data)
                    print(f"   Decoded audio size: {len(audio_bytes)} bytes")
                except Exception as decode_error:
                    print(f"   ⚠️  Audio decode warning: {decode_error}")
                
                return True
            else:
                print(f"❌ Music generation failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Music generation failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Music generation failed: {e}")
        return False

def test_frontend_connection():
    """Test if frontend is accessible."""
    print("\n🌐 Testing frontend connection...")
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible!")
            return True
        else:
            print(f"❌ Frontend returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend connection failed: {e}")
        print("   Make sure to run 'npm run dev' in the frontend directory")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Magenta RT + ShadCN UI Setup Test")
    print("=" * 50)
    
    # Test backend
    health_ok = test_health_check()
    if not health_ok:
        print("\n❌ Backend is not running or not healthy!")
        print("   Make sure to run 'python backend/main.py'")
        sys.exit(1)
    
    # Test style embedding
    style_embedding = test_style_embedding()
    
    # Test music generation
    generation_ok = test_music_generation(style_embedding)
    
    # Test frontend
    frontend_ok = test_frontend_connection()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Backend Health: {'✅' if health_ok else '❌'}")
    print(f"   Style Embedding: {'✅' if style_embedding else '❌'}")
    print(f"   Music Generation: {'✅' if generation_ok else '❌'}")
    print(f"   Frontend Access: {'✅' if frontend_ok else '❌'}")
    
    if health_ok and style_embedding and generation_ok:
        print("\n🎉 All backend tests passed! Your Magenta RT setup is working!")
        if frontend_ok:
            print("🎨 Frontend is also accessible!")
            print("\n🎵 Ready to generate music!")
            print("   1. Open http://localhost:3000 in your browser")
            print("   2. Enter a style prompt")
            print("   3. Click 'Generate Music'")
            print("   4. Enjoy your AI-generated music!")
        else:
            print("⚠️  Frontend is not accessible, but backend is working")
    else:
        print("\n❌ Some tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()
