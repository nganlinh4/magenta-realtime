"use client";

import { useState, useEffect, useCallback } from "react";
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

// Add axios defaults for better error handling
axios.defaults.timeout = 10000;
axios.defaults.headers.common['Content-Type'] = 'application/json';

interface GenerateChunkRequest {
  style_embedding?: number[];
  seed?: number;
  temperature?: number;
  topk?: number;
  guidance_weight?: number;
}

interface GenerateChunkResponse {
  audio_data: string;
  sample_rate: number;
  chunk_index: number;
  success: boolean;
  error?: string;
}

interface HealthResponse {
  status: string;
  magenta_rt_loaded: boolean;
  gpu_available: boolean;
  jax_version: string;
}

export function useAudioGeneration() {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentChunk, setCurrentChunk] = useState<GenerateChunkResponse | null>(null);
  const [chunkIndex, setChunkIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  // Check backend health
  const checkHealth = useCallback(async () => {
    try {
      console.log("Checking health at:", `${API_BASE_URL}/api/health`);
      const response = await axios.get<HealthResponse>(`${API_BASE_URL}/api/health`, {
        timeout: 5000,
        headers: {
          'Content-Type': 'application/json',
        }
      });
      console.log("Health check response:", response.data);
      setIsConnected(response.data.status === "healthy");
      setError(null);
      return response.data;
    } catch (err) {
      setIsConnected(false);
      const errorMessage = axios.isAxiosError(err)
        ? `${err.message} (${err.code || 'Unknown'})`
        : "Unknown error";
      setError(`Failed to connect: ${errorMessage}`);
      console.error("Health check failed:", err);
      return null;
    }
  }, []);

  // Embed style text into embedding vector
  const embedStyle = useCallback(async (textOrAudio: string, weight: number = 1.0): Promise<number[]> => {
    try {
      setIsLoading(true);
      const response = await axios.post(`${API_BASE_URL}/api/embed-style`, {
        text_or_audio: textOrAudio,
        weight: weight
      });
      
      if (response.data.success) {
        return response.data.style_embedding;
      } else {
        throw new Error("Style embedding failed");
      }
    } catch (err) {
      const errorMessage = axios.isAxiosError(err) 
        ? err.response?.data?.detail || err.message
        : "Unknown error occurred";
      setError(`Style embedding failed: ${errorMessage}`);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Generate a single audio chunk
  const generateChunk = useCallback(async (request: GenerateChunkRequest): Promise<GenerateChunkResponse> => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await axios.post<GenerateChunkResponse>(
        `${API_BASE_URL}/api/generate-chunk`,
        request
      );
      
      if (response.data.success) {
        setCurrentChunk(response.data);
        setChunkIndex(response.data.chunk_index);
        return response.data;
      } else {
        throw new Error(response.data.error || "Generation failed");
      }
    } catch (err) {
      const errorMessage = axios.isAxiosError(err) 
        ? err.response?.data?.detail || err.message
        : "Unknown error occurred";
      setError(`Generation failed: ${errorMessage}`);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // WebSocket connection for real-time generation (future enhancement)
  const connectWebSocket = useCallback(() => {
    // TODO: Implement WebSocket connection for real-time streaming
    // This would allow for continuous generation and real-time audio streaming
    console.log("WebSocket connection not yet implemented");
  }, []);

  // Reset generation state
  const reset = useCallback(() => {
    setCurrentChunk(null);
    setChunkIndex(0);
    setError(null);
  }, []);

  // Check health on mount and periodically
  useEffect(() => {
    // Wait a bit before first health check to let everything load
    const initialTimeout = setTimeout(() => {
      checkHealth();
    }, 1000);

    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);

    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [checkHealth]);

  return {
    // State
    isConnected,
    error,
    currentChunk,
    chunkIndex,
    isLoading,
    
    // Actions
    embedStyle,
    generateChunk,
    checkHealth,
    connectWebSocket,
    reset,
  };
}
