"use client";

import { useState, useCallback } from "react";
import { HeroSection } from "@/components/HeroSection";
import { MoodSelector } from "@/components/MoodSelector";
import { GenerateButton } from "@/components/GenerateButton";
import { MidiPlayer } from "@/components/MidiPlayer";
import { StatusFeedback, StatusType } from "@/components/StatusFeedback";
import { GenerationProgress } from "@/components/GenerationProgress";
import { CreativeControls } from "@/components/CreativeControls";
import { EmptyState } from "@/components/EmptyState";
import { Footer } from "@/components/Footer";
import { toast } from "sonner";

// UI State type
type UIState = "idle" | "selecting" | "generating" | "playing" | "error";

export default function Home() {
  const [selectedMood, setSelectedMood] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [midiData, setMidiData] = useState<ArrayBuffer | null>(null);
  const [uiState, setUiState] = useState<UIState>("idle");
  const [status, setStatus] = useState<{ type: StatusType; message: string }>({
    type: "idle",
    message: "Select a mood to get started",
  });

  // Creative controls state
  const [temperature, setTemperature] = useState(0.5);
  const [bpm, setBpm] = useState(120);

  const handleMoodSelect = useCallback((mood: string) => {
    setSelectedMood(mood);
    setUiState("selecting");
    setStatus({
      type: "info",
      message: `Mood selected: ${mood.charAt(0).toUpperCase() + mood.slice(1)}`,
    });
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!selectedMood) {
      toast.error("Please select a mood first");
      return;
    }

    setIsGenerating(true);
    setUiState("generating");
    setStatus({ type: "loading", message: "Creating your unique composition..." });

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";
      const queryParams = new URLSearchParams({
        mood: selectedMood,
        temperature: temperature.toString(),
        bpm: bpm.toString()
      });
      const response = await fetch(`${baseUrl}/generate?${queryParams}`);

      if (!response.ok) {
        if (response.status === 400) {
          throw new Error("Invalid parameters");
        }
        throw new Error("Failed to generate music");
      }

      const data = await response.arrayBuffer();
      console.log("Received MIDI data. Size:", data.byteLength, "bytes");
      setMidiData(data);
      setUiState("playing");
      setStatus({ type: "success", message: "Your track is ready to play!" });
      toast.success("Music generated successfully! Hit play to listen.", {
        duration: 4000,
      });
    } catch (error) {
      console.error("API Error:", error);
      const message = error instanceof Error ? error.message : "Generation failed";
      setUiState("error");
      setStatus({ type: "error", message });
      toast.error(message, {
        description: "Try selecting a different mood or try again later.",
        duration: 5000,
      });
    } finally {
      setIsGenerating(false);
    }
  }, [selectedMood, temperature, bpm]);

  const handleDownload = useCallback(() => {
    if (!midiData || !selectedMood) return;

    const blob = new Blob([midiData], { type: "audio/midi" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `generated_${selectedMood}.mid`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast.success("MIDI file downloaded!", {
      description: "Open it in any DAW or MIDI-compatible software.",
    });
  }, [midiData, selectedMood]);

  return (
    <div className="min-h-screen bg-background">
      {/* Generation overlay */}
      <GenerationProgress isGenerating={isGenerating} />

      <main className={isGenerating ? "pointer-events-none" : ""}>
        <HeroSection />

        <StatusFeedback status={status.type} message={status.message} />

        <MoodSelector
          selectedMood={selectedMood}
          onMoodSelect={handleMoodSelect}
          disabled={isGenerating}
        />

        {/* Creative controls */}
        <CreativeControls
          temperature={temperature}
          onTemperatureChange={setTemperature}
          bpm={bpm}
          onBpmChange={setBpm}
          disabled={isGenerating}
        />

        <GenerateButton
          onClick={handleGenerate}
          isGenerating={isGenerating}
          disabled={!selectedMood}
        />

        {/* Show empty state or player */}
        {midiData ? (
          <MidiPlayer
            midiData={midiData}
            selectedMood={selectedMood}
            onDownload={handleDownload}
          />
        ) : (
          <EmptyState hasSelectedMood={!!selectedMood} />
        )}
      </main>

      <Footer />
    </div>
  );
}
