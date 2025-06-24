"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Play,
  Pause,
  Square,
  Music,
  Settings,
  RotateCcw,
  Volume2,
  Zap
} from "lucide-react";
import { useAudioGeneration } from "@/hooks/useAudioGeneration";

interface StylePrompt {
  id: number;
  text: string;
  weight: number;
}

export function MagentaRTStudio() {
  // Real-time streaming state
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingProgress, setStreamingProgress] = useState(0);

  // Style prompts with individual weights
  const [stylePrompts, setStylePrompts] = useState<StylePrompt[]>([
    { id: 0, text: "synthwave", weight: 1.0 },
    { id: 1, text: "flamenco guitar", weight: 0.0 },
    { id: 2, text: "", weight: 0.0 },
    { id: 3, text: "", weight: 0.0 },
  ]);

  // Sampling parameters
  const [temperature, setTemperature] = useState([1.3]);
  const [topK, setTopK] = useState([40]);
  const [guidanceWeight, setGuidanceWeight] = useState([5.0]);
  const [volume, setVolume] = useState([0.7]);

  // Audio context and playback
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentAudioUrl, setCurrentAudioUrl] = useState<string | null>(null);

  const {
    generateChunk,
    embedStyle,
    isConnected,
    error,
    currentChunk,
    chunkIndex,
    checkHealth
  } = useAudioGeneration();

  // Update style prompts
  const updateStylePrompt = useCallback((id: number, text: string) => {
    setStylePrompts(prev => prev.map(prompt =>
      prompt.id === id ? { ...prompt, text } : prompt
    ));
  }, []);

  const updateStyleWeight = useCallback((id: number, weight: number) => {
    setStylePrompts(prev => prev.map(prompt =>
      prompt.id === id ? { ...prompt, weight } : prompt
    ));
  }, []);

  // Convert base64 audio to blob URL for playback
  const base64ToAudioUrl = useCallback((base64Data: string) => {
    try {
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'audio/wav' });
      return URL.createObjectURL(blob);
    } catch (error) {
      console.error('Failed to convert base64 to audio URL:', error);
      return null;
    }
  }, []);

  // Create weighted style embedding from multiple prompts
  const createWeightedEmbedding = useCallback(async (prompts: StylePrompt[]) => {
    const activePrompts = prompts.filter(p => p.text.trim() && p.weight > 0);
    if (activePrompts.length === 0) return null;

    // For now, just use the first active prompt
    // TODO: Implement proper weighted mixing
    const firstPrompt = activePrompts[0];
    return await embedStyle(firstPrompt.text);
  }, [embedStyle]);

  // Real-time streaming controls
  const handleStartStreaming = useCallback(async () => {
    if (!isConnected) return;

    const activePrompts = stylePrompts.filter(p => p.text.trim() && p.weight > 0);
    if (activePrompts.length === 0) return;

    setIsStreaming(true);
    setIsPlaying(true);

    try {
      // Generate and play audio chunk
      const styleEmbedding = await createWeightedEmbedding(stylePrompts);
      if (styleEmbedding) {
        const chunk = await generateChunk({
          style_embedding: styleEmbedding,
          temperature: temperature[0],
          topk: topK[0],
          guidance_weight: guidanceWeight[0]
        });

        if (chunk && chunk.audio_data) {
          // Convert to audio URL and play
          const audioUrl = base64ToAudioUrl(chunk.audio_data);
          if (audioUrl) {
            // Clean up previous audio URL
            if (currentAudioUrl) {
              URL.revokeObjectURL(currentAudioUrl);
            }

            setCurrentAudioUrl(audioUrl);

            // Play the audio
            if (audioElementRef.current) {
              audioElementRef.current.src = audioUrl;
              audioElementRef.current.volume = volume[0];
              try {
                await audioElementRef.current.play();
                console.log('Audio playing successfully');
              } catch (playError) {
                console.error('Failed to play audio:', playError);
              }
            }
          }
        }
      }
    } catch (err) {
      console.error('Streaming error:', err);
      setIsStreaming(false);
      setIsPlaying(false);
    }
  }, [isConnected, stylePrompts, temperature, topK, guidanceWeight, createWeightedEmbedding, generateChunk, base64ToAudioUrl, currentAudioUrl, volume]);

  const handleStopStreaming = useCallback(() => {
    setIsStreaming(false);
    setIsPlaying(false);

    // Stop audio playback
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
    }
  }, []);

  const handleReset = useCallback(() => {
    setIsStreaming(false);
    setIsPlaying(false);
    setStreamingProgress(0);

    // Stop and reset audio
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
    }

    // Clean up audio URL
    if (currentAudioUrl) {
      URL.revokeObjectURL(currentAudioUrl);
      setCurrentAudioUrl(null);
    }
  }, [currentAudioUrl]);

  // Initialize audio context
  useEffect(() => {
    if (typeof window !== 'undefined') {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Update streaming progress
  useEffect(() => {
    if (isStreaming && chunkIndex > 0) {
      setStreamingProgress((chunkIndex % 10) * 10); // Simple progress indicator
    }
  }, [isStreaming, chunkIndex]);

  // Update audio volume when volume slider changes
  useEffect(() => {
    if (audioElementRef.current) {
      audioElementRef.current.volume = volume[0];
    }
  }, [volume]);

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent mb-2">
          Magenta RT Studio
        </h1>
        <p className="text-muted-foreground text-lg">
          Streaming music generation! 🎵
        </p>
        <div className="flex items-center justify-center gap-2 mt-4">
          <Badge variant={isConnected ? "default" : "destructive"}>
            {isConnected ? "Connected" : "Disconnected"}
          </Badge>
          {isStreaming && (
            <Badge variant="secondary" className="animate-pulse">
              <Zap className="h-3 w-3 mr-1" />
              Streaming
            </Badge>
          )}
          {error && (
            <Badge variant="destructive" className="max-w-md truncate">
              {error}
            </Badge>
          )}
          {!isConnected && (
            <Button
              variant="outline"
              size="sm"
              onClick={checkHealth}
              className="ml-2"
            >
              Retry Connection
            </Button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sampling Options Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Sampling Options
            </CardTitle>
            <CardDescription>
              Control the generation behavior in real-time
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <label className="text-sm font-medium mb-2 block">
                Temperature: {temperature[0].toFixed(2)}
              </label>
              <Slider
                value={temperature}
                onValueChange={setTemperature}
                min={0.0}
                max={4.0}
                step={0.01}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Controls chaos (low = predictable, high = experimental)
              </p>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">
                Top-K: {topK[0]}
              </label>
              <Slider
                value={topK}
                onValueChange={setTopK}
                min={0}
                max={1024}
                step={1}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Vocabulary filtering (low = safer, high = more variety)
              </p>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">
                Guidance: {guidanceWeight[0].toFixed(2)}
              </label>
              <Slider
                value={guidanceWeight}
                onValueChange={setGuidanceWeight}
                min={0.0}
                max={10.0}
                step={0.01}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Style adherence (high = strict, low = creative freedom)
              </p>
            </div>

            <div className="pt-4 border-t space-y-2">
              <div className="flex gap-2">
                <Button
                  onClick={isStreaming ? handleStopStreaming : handleStartStreaming}
                  disabled={!isConnected}
                  className="flex-1"
                  variant={isStreaming ? "destructive" : "default"}
                >
                  {isStreaming ? (
                    <>
                      <Square className="mr-2 h-4 w-4" />
                      Stop
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Start
                    </>
                  )}
                </Button>

                <Button
                  onClick={handleReset}
                  variant="outline"
                  size="icon"
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </div>

              {/* Test button for single generation */}
              <Button
                onClick={async () => {
                  try {
                    const styleEmbedding = await createWeightedEmbedding(stylePrompts);
                    if (styleEmbedding) {
                      const chunk = await generateChunk({
                        style_embedding: styleEmbedding,
                        temperature: temperature[0],
                        topk: topK[0],
                        guidance_weight: guidanceWeight[0]
                      });

                      if (chunk && chunk.audio_data) {
                        const audioUrl = base64ToAudioUrl(chunk.audio_data);
                        if (audioUrl) {
                          if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
                          setCurrentAudioUrl(audioUrl);
                          console.log('Generated audio chunk, URL created');
                        }
                      }
                    }
                  } catch (err) {
                    console.error('Test generation failed:', err);
                  }
                }}
                variant="outline"
                className="w-full"
                disabled={!isConnected}
              >
                🎵 Test Generate Single Chunk
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Style Prompts Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Music className="h-5 w-5" />
              Style Prompts
            </CardTitle>
            <CardDescription>
              Mix multiple styles with individual weights
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {stylePrompts.map((prompt) => (
              <div key={prompt.id} className="space-y-2">
                <div className="flex items-center gap-2">
                  <Input
                    placeholder="Enter style prompt..."
                    value={prompt.text}
                    onChange={(e) => updateStylePrompt(prompt.id, e.target.value)}
                    className="flex-1"
                  />
                  <div className="text-xs text-muted-foreground min-w-[3rem] text-right">
                    {prompt.weight.toFixed(1)}
                  </div>
                </div>
                <Slider
                  value={[prompt.weight]}
                  onValueChange={(value) => updateStyleWeight(prompt.id, value[0])}
                  min={0.0}
                  max={2.0}
                  step={0.1}
                  className="w-full"
                  disabled={!prompt.text.trim()}
                />
              </div>
            ))}

            <div className="pt-4 border-t">
              <div className="text-xs text-muted-foreground">
                💡 Tip: Adjust sliders in real-time while streaming to blend styles!
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Real-time Audio Status Panel */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Volume2 className="h-5 w-5" />
              Real-time Audio Stream
            </CardTitle>
            <CardDescription>
              {isStreaming ? `Streaming chunk ${chunkIndex}...` : "Ready to stream"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isStreaming ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Stream Status</span>
                  <Badge variant="secondary" className="animate-pulse">
                    <Zap className="h-3 w-3 mr-1" />
                    Live
                  </Badge>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Chunks Generated</span>
                    <span>{chunkIndex}</span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all duration-300"
                      style={{ width: `${streamingProgress}%` }}
                    />
                  </div>
                </div>

                <div className="text-xs text-muted-foreground">
                  🎵 Audio is streaming in real-time. Adjust the sliders above to change the music style!
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Music className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="mb-2">Ready to start streaming</p>
                <p className="text-sm">
                  1. Set your style prompts and weights<br/>
                  2. Adjust sampling parameters<br/>
                  3. Click "Start" to begin real-time generation
                </p>
              </div>
            )}

            <div className="mt-4 pt-4 border-t">
              <div className="flex items-center gap-2 mb-4">
                <Volume2 className="h-4 w-4" />
                <span className="text-sm font-medium">Volume: {Math.round(volume[0] * 100)}%</span>
                <Slider
                  value={volume}
                  onValueChange={setVolume}
                  min={0}
                  max={1}
                  step={0.1}
                  className="flex-1 ml-2"
                />
              </div>

              {/* Hidden audio element for playback */}
              <audio
                ref={audioElementRef}
                onEnded={() => {
                  setIsPlaying(false);
                  setIsStreaming(false);
                }}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onError={(e) => {
                  console.error('Audio playback error:', e);
                  setIsPlaying(false);
                  setIsStreaming(false);
                }}
                style={{ display: 'none' }}
              />

              {/* Audio controls for debugging */}
              {currentAudioUrl && (
                <div className="mt-2 p-2 bg-secondary rounded">
                  <div className="text-xs text-muted-foreground mb-1">Audio Controls (for testing)</div>
                  <audio
                    src={currentAudioUrl}
                    controls
                    className="w-full h-8"
                    style={{ height: '32px' }}
                  />
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
