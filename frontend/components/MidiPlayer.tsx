import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { AnimatedEqualizer } from "./AnimatedEqualizer";
import { cn } from "@/lib/utils";
import {
  Play,
  Pause,
  Square,
  Repeat,
  Volume2,
  VolumeX,
  Download,
  Music
} from "lucide-react";
import * as Tone from "tone";
import { Midi } from "@tonejs/midi";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface MidiPlayerProps {
  midiData: ArrayBuffer | null;
  selectedMood: string | null;
  onDownload: () => void;
}

export const MidiPlayer = ({ midiData, selectedMood, onDownload }: MidiPlayerProps) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [volume, setVolume] = useState(80);
  const [isMuted, setIsMuted] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const synthRef = useRef<Tone.PolySynth | null>(null);
  const partRef = useRef<Tone.Part | null>(null);
  const intervalRef = useRef<number | null>(null);

  // Initialize synth
  useEffect(() => {
    synthRef.current = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "triangle" },
      envelope: { attack: 0.02, decay: 0.1, sustain: 0.3, release: 0.8 }
    }).toDestination();

    return () => {
      if (synthRef.current) {
        synthRef.current.dispose();
      }
      if (partRef.current) {
        partRef.current.dispose();
      }
    };
  }, []);

  // Update volume
  useEffect(() => {
    if (synthRef.current) {
      const db = isMuted ? -Infinity : Tone.gainToDb(volume / 100);
      synthRef.current.volume.value = db;
    }
  }, [volume, isMuted]);

  // Load MIDI data
  useEffect(() => {
    if (!midiData) return;

    const loadMidi = async () => {
      try {
        const midi = new Midi(midiData);
        console.log("MIDI Loaded:", midi.name, "Duration:", midi.duration, "Tracks:", midi.tracks.length);
        setDuration(midi.duration);

        // Dispose previous part
        if (partRef.current) {
          partRef.current.dispose();
        }

        // Create notes array from all tracks
        const notes: { time: number; note: string; duration: number }[] = [];
        midi.tracks.forEach((track, i) => {
          console.log(`  Track ${i}: ${track.notes.length} notes`);
          track.notes.forEach((note) => {
            notes.push({
              time: note.time,
              note: note.name,
              duration: note.duration,
            });
          });
        });

        if (notes.length === 0) {
          console.warn("No notes found in MIDI data!");
        }

        // Create Tone.Part
        partRef.current = new Tone.Part((time, value) => {
          if (synthRef.current) {
            synthRef.current.triggerAttackRelease(
              value.note,
              value.duration,
              time
            );
          }
        }, notes);

        partRef.current.loop = isLooping;
        partRef.current.loopEnd = midi.duration;
        console.log("Tone.Part created with", notes.length, "notes");
      } catch (error) {
        console.error("Error loading MIDI:", error);
      }
    };

    loadMidi();
  }, [midiData]);

  // Update loop setting and Tone.Transport looping
  useEffect(() => {
    if (partRef.current) {
      partRef.current.loop = isLooping;
    }

    // Sync Tone.Transport looping for accurate currentTime tracking
    Tone.Transport.loop = isLooping;
    if (duration > 0) {
      Tone.Transport.loopEnd = duration;
    }
  }, [isLooping, duration]);

  // Track playback time
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = window.setInterval(() => {
        // Use Transport.seconds directly as it will wrap if looping
        const time = Tone.Transport.seconds;

        if (!isLooping && time >= duration) {
          handleStop();
        } else {
          setCurrentTime(time);
        }
      }, 100);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, duration]);

  const handlePlay = useCallback(async () => {
    if (!partRef.current) return;

    console.log("Handling Play. Current state:", isPlaying);
    await Tone.start();
    console.log("Audio Context started");

    if (isPlaying) {
      Tone.Transport.pause();
      setIsPlaying(false);
      console.log("⏸ Paused");
    } else {
      if (partRef.current) {
        // Ensure part is scheduled
        if (partRef.current.state !== "started") {
          partRef.current.start(0);
        }
        Tone.Transport.start();
        setIsPlaying(true);
        console.log("Playing from", Tone.Transport.seconds);
      } else {
        console.error("Cannot play: partRef.current is null");
      }
    }
  }, [isPlaying]);

  const handleStop = useCallback(() => {
    Tone.Transport.stop();
    Tone.Transport.position = 0;
    setIsPlaying(false);
    setCurrentTime(0);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

  if (!midiData) {
    return null;
  }

  return (
    <section className="py-12 px-4 animate-fade-in">
      <div className={cn(
        "max-w-2xl mx-auto glass-panel p-6 space-y-6",
        "transition-all duration-500",
        isPlaying && "shadow-[0_0_40px_hsla(270,91%,55%,0.3)]"
      )}>
        {/* Track info with playing indicator */}
        <div className="text-center relative">
          <div className="flex items-center justify-center gap-3">
            {isPlaying && (
              <div className="flex gap-0.5">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div
                    key={i}
                    className="w-1 bg-primary rounded-full animate-wave"
                    style={{
                      height: 12,
                      animationDelay: `${i * 0.15}s`
                    }}
                  />
                ))}
              </div>
            )}
            <h3 className="font-display text-xl font-semibold text-primary">
              Generated Track
            </h3>
            {isPlaying && (
              <span className="px-2 py-0.5 text-xs font-medium bg-primary/20 text-primary rounded-full animate-pulse">
                NOW PLAYING
              </span>
            )}
          </div>
          <p className="text-sm text-muted-foreground capitalize mt-1">
            Mood: {selectedMood}
          </p>
        </div>

        {/* Equalizer visualization */}
        <AnimatedEqualizer isPlaying={isPlaying} barCount={40} className="h-24" />

        {/* Progress bar with visual timeline */}
        <div className="space-y-2">
          {/* Visual progress track */}
          <div className="relative h-1 bg-muted rounded-full overflow-hidden mb-4">
            <div
              className={cn(
                "absolute inset-y-0 left-0 rounded-full transition-all duration-100",
                isPlaying
                  ? "bg-gradient-to-r from-primary via-secondary to-primary bg-[length:200%_100%] animate-shimmer"
                  : "bg-primary"
              )}
              style={{ width: `${progressPercent}%` }}
            />
            {/* Playhead */}
            <div
              className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-primary rounded-full shadow-[0_0_10px_hsla(270,91%,55%,0.8)] transition-all duration-100"
              style={{ left: `calc(${progressPercent}% - 6px)` }}
            />
          </div>

          <Slider
            value={[currentTime]}
            max={duration || 100}
            step={0.1}
            className="cursor-pointer"
            onValueChange={(value) => {
              Tone.Transport.seconds = value[0];
              setCurrentTime(value[0]);
            }}
          />
          <div className="flex justify-between text-xs text-muted-foreground font-mono">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Controls with tooltips and hover effects */}
        <TooltipProvider delayDuration={200}>
          <div className="flex items-center justify-center gap-4">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="glass"
                  size="icon"
                  onClick={() => setIsLooping(!isLooping)}
                  className={cn(
                    "transition-all duration-300 hover:scale-110 active:scale-95",
                    isLooping && "text-primary border-primary bg-primary/10"
                  )}
                >
                  <Repeat className="w-5 h-5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{isLooping ? "Disable loop" : "Enable loop"}</p>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="glass"
                  size="icon"
                  onClick={handleStop}
                  className="transition-all duration-300 hover:scale-110 active:scale-95"
                >
                  <Square className="w-5 h-5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Stop</p>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="neon"
                  size="xl"
                  onClick={handlePlay}
                  className={cn(
                    "rounded-full w-16 h-16 transition-all duration-300",
                    "hover:scale-110 active:scale-95",
                    isPlaying && "animate-pulse-glow"
                  )}
                >
                  {isPlaying ? (
                    <Pause className="w-6 h-6" />
                  ) : (
                    <Play className="w-6 h-6 ml-1" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{isPlaying ? "Pause" : "Play"}</p>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="glass"
                  size="icon"
                  onClick={() => setIsMuted(!isMuted)}
                  className={cn(
                    "transition-all duration-300 hover:scale-110 active:scale-95",
                    isMuted && "text-destructive border-destructive/50"
                  )}
                >
                  {isMuted ? (
                    <VolumeX className="w-5 h-5" />
                  ) : (
                    <Volume2 className="w-5 h-5" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{isMuted ? "Unmute" : "Mute"}</p>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="glass"
                  size="icon"
                  onClick={onDownload}
                  className="transition-all duration-300 hover:scale-110 hover:text-green-400 hover:border-green-400/50 active:scale-95"
                >
                  <Download className="w-5 h-5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Download MIDI</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </TooltipProvider>

        {/* Volume slider */}
        <div className="flex items-center gap-4 max-w-xs mx-auto">
          <VolumeX className={cn(
            "w-4 h-4 transition-colors",
            isMuted ? "text-destructive" : "text-muted-foreground"
          )} />
          <Slider
            value={[volume]}
            max={100}
            step={1}
            onValueChange={(value) => setVolume(value[0])}
            className={cn(isMuted && "opacity-50")}
          />
          <Volume2 className="w-4 h-4 text-muted-foreground" />
        </div>
      </div>
    </section>
  );
};
