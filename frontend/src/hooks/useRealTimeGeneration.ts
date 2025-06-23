"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

interface StylePrompt {
  id: number;
  text: string;
  weight: number;
}

interface StreamingParams {
  stylePrompts: StylePrompt[];
  temperature: number;
  topk: number;
  guidance_weight: number;
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

export function useRealTimeGeneration() {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentChunk, setCurrentChunk] = useState<GenerateChunkResponse | null>(null);
  const [chunkIndex, setChunkIndex] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);

  // WebSocket connection for real-time streaming
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const currentParamsRef = useRef<StreamingParams | null>(null);

  // Check backend health
  const checkHealth = useCallback(async () => {
    try {
      const response = await axios.get<HealthResponse>(`${API_BASE_URL}/api/health`);
      setIsConnected(response.data.status === "healthy");
      setError(null);
      return response.data;
    } catch (err) {
      setIsConnected(false);
      setError("Failed to connect to backend");
      console.error("Health check failed:", err);
      return null;
    }
  }, []);

  // Embed multiple style prompts and create weighted embedding
  const createWeightedEmbedding = useCallback(async (stylePrompts: StylePrompt[]): Promise<number[]> => {
    try {
      const activePrompts = stylePrompts.filter(p => p.text.trim() && p.weight > 0);
      if (activePrompts.length === 0) {
        throw new Error("No active style prompts");
      }

      // Embed each style prompt
      const embeddings = await Promise.all(
        activePrompts.map(async (prompt) => {
          const response = await axios.post(`${API_BASE_URL}/api/embed-style`, {
            text_or_audio: prompt.text,
            weight: 1.0
          });
          
          if (response.data.success) {
            return {
              embedding: response.data.style_embedding,
              weight: prompt.weight
            };
          } else {
            throw new Error(`Failed to embed style: ${prompt.text}`);
          }
        })
      );

      // Create weighted average
      const embeddingSize = embeddings[0].embedding.length;
      const weightedEmbedding = new Array(embeddingSize).fill(0);
      let totalWeight = 0;

      embeddings.forEach(({ embedding, weight }) => {
        for (let i = 0; i < embeddingSize; i++) {
          weightedEmbedding[i] += embedding[i] * weight;
        }
        totalWeight += weight;
      });

      // Normalize by total weight
      if (totalWeight > 0) {
        for (let i = 0; i < embeddingSize; i++) {
          weightedEmbedding[i] /= totalWeight;
        }
      }

      return weightedEmbedding;
    } catch (err) {
      const errorMessage = axios.isAxiosError(err) 
        ? err.response?.data?.detail || err.message
        : "Unknown error occurred";
      setError(`Style embedding failed: ${errorMessage}`);
      throw err;
    }
  }, []);

  // Generate a single chunk with current parameters
  const generateSingleChunk = useCallback(async (styleEmbedding: number[], params: StreamingParams): Promise<GenerateChunkResponse> => {
    try {
      const response = await axios.post<GenerateChunkResponse>(
        `${API_BASE_URL}/api/generate-chunk`,
        {
          style_embedding: styleEmbedding,
          temperature: params.temperature,
          topk: params.topk,
          guidance_weight: params.guidance_weight
        }
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
    }
  }, []);

  // Convert base64 audio to AudioBuffer and play it
  const playAudioChunk = useCallback(async (audioData: string, sampleRate: number) => {
    if (!audioContextRef.current) return;

    try {
      // Decode base64 to ArrayBuffer
      const binaryString = atob(audioData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Decode audio data
      const audioBuffer = await audioContextRef.current.decodeAudioData(bytes.buffer);
      
      // Create and play audio source
      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);
      source.start();

      setAudioBuffer(audioBuffer);
    } catch (err) {
      console.error("Failed to play audio chunk:", err);
    }
  }, []);

  // Start real-time streaming
  const startStreaming = useCallback(async (params: StreamingParams) => {
    if (!isConnected) {
      setError("Not connected to backend");
      return;
    }

    try {
      setIsStreaming(true);
      setError(null);
      currentParamsRef.current = params;

      // Initialize audio context if needed
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }

      // Resume audio context if suspended
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      // Create initial weighted embedding
      const styleEmbedding = await createWeightedEmbedding(params.stylePrompts);

      // Start streaming loop
      const streamingLoop = async () => {
        if (!isStreaming || !currentParamsRef.current) return;

        try {
          // Recreate weighted embedding with current parameters (for real-time updates)
          const currentEmbedding = await createWeightedEmbedding(currentParamsRef.current.stylePrompts);
          
          // Generate chunk
          const chunk = await generateSingleChunk(currentEmbedding, currentParamsRef.current);
          
          // Play the chunk
          await playAudioChunk(chunk.audio_data, chunk.sample_rate);
          
        } catch (err) {
          console.error("Streaming loop error:", err);
          setError("Streaming interrupted");
        }
      };

      // Start the streaming loop (generate chunks every 2 seconds to match the chunk length)
      streamingIntervalRef.current = setInterval(streamingLoop, 2000);
      
      // Generate first chunk immediately
      streamingLoop();

    } catch (err) {
      setIsStreaming(false);
      setError("Failed to start streaming");
      console.error("Start streaming error:", err);
    }
  }, [isConnected, createWeightedEmbedding, generateSingleChunk, playAudioChunk]);

  // Stop streaming
  const stopStreaming = useCallback(async () => {
    setIsStreaming(false);
    
    if (streamingIntervalRef.current) {
      clearInterval(streamingIntervalRef.current);
      streamingIntervalRef.current = null;
    }

    currentParamsRef.current = null;
  }, []);

  // Reset generation state
  const resetGeneration = useCallback(async () => {
    await stopStreaming();
    setCurrentChunk(null);
    setChunkIndex(0);
    setError(null);
    setAudioBuffer(null);
  }, [stopStreaming]);

  // Update streaming parameters in real-time
  const updateStreamingParams = useCallback((params: Partial<StreamingParams>) => {
    if (currentParamsRef.current) {
      currentParamsRef.current = { ...currentParamsRef.current, ...params };
    }
  }, []);

  // Check health on mount and periodically
  useEffect(() => {
    checkHealth();
    
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [stopStreaming]);

  return {
    // State
    isConnected,
    error,
    currentChunk,
    chunkIndex,
    isStreaming,
    audioBuffer,
    
    // Actions
    startStreaming,
    stopStreaming,
    resetGeneration,
    updateStreamingParams,
    checkHealth,
  };
}
