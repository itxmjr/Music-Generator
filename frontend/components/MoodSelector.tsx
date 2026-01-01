import { Button } from "./ui/button";
import {
  Zap,
  CloudRain,
  Wind,
  Music,
  Radio,
  Mic2,
  Guitar,
  Headphones,
  Disc,
  Sparkles,
  Activity,
  Waves
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";

const moods = [
  { id: "energetic", label: "Energetic", icon: Zap, description: "High-energy beats with driving rhythms" },
  { id: "melancholic", label: "Melancholic", icon: CloudRain, description: "Emotional, reflective melodies" },
  { id: "ambient", label: "Ambient", icon: Wind, description: "Atmospheric, ethereal soundscapes" },
  { id: "classical", label: "Classical", icon: Music, description: "Elegant, orchestral compositions" },
  { id: "synthwave", label: "Synthwave", icon: Radio, description: "Retro 80s electronic vibes" },
  { id: "jazz", label: "Jazz", icon: Mic2, description: "Smooth, improvisational harmonies" },
  { id: "rock", label: "Rock", icon: Guitar, description: "Bold, guitar-driven power" },
  { id: "lofi", label: "Lo-Fi", icon: Headphones, description: "Chill, nostalgic hip-hop beats" },
  { id: "pop", label: "Pop", icon: Sparkles, description: "Catchy, upbeat melodies" },
  { id: "hiphop", label: "Hip-Hop", icon: Disc, description: "Rhythmic beats and grooves" },
  { id: "electro", label: "Electro", icon: Activity, description: "Electronic dance energy" },
  { id: "blues", label: "Blues", icon: Waves, description: "Soulful, expressive tunes" },
];

interface MoodSelectorProps {
  selectedMood: string | null;
  onMoodSelect: (mood: string) => void;
  disabled?: boolean;
}

export const MoodSelector = ({ selectedMood, onMoodSelect, disabled }: MoodSelectorProps) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <section className="py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <h2 className="font-display text-2xl md:text-3xl font-semibold text-center mb-8">
          Select Your <span className="text-primary">Mood</span>
        </h2>

        <TooltipProvider delayDuration={200}>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {moods.map((mood) => {
              const Icon = mood.icon;
              const isSelected = selectedMood === mood.id;

              return (
                <Tooltip key={mood.id}>
                  <TooltipTrigger asChild>
                    <Button
                      variant="mood"
                      size="lg"
                      data-selected={isSelected}
                      disabled={disabled}
                      onClick={() => onMoodSelect(mood.id)}
                      className={cn(
                        "flex flex-col h-24 gap-2 group relative overflow-hidden",
                        "transition-all duration-300 ease-out",
                        "active:scale-95",
                        isSelected && "animate-mood-selected"
                      )}
                    >
                      {/* Pulse ring animation for selected mood */}
                      {isSelected && (
                        <div className="absolute inset-0 rounded-lg animate-pulse-ring" />
                      )}

                      {/* Sound wave hover effect */}
                      <div className={cn(
                        "absolute inset-0 flex items-end justify-center gap-0.5 opacity-0 group-hover:opacity-20 transition-opacity duration-300",
                        isSelected && "opacity-30"
                      )}>
                        {Array.from({ length: 8 }).map((_, i) => (
                          <div
                            key={i}
                            className="w-0.5 bg-primary rounded-full animate-wave"
                            style={{
                              height: mounted ? `${20 + Math.random() * 60}%` : "20%",
                              animationDelay: `${i * 0.1}s`,
                            }}
                          />
                        ))}
                      </div>

                      <Icon
                        className={cn(
                          "w-6 h-6 transition-all duration-300 relative z-10",
                          isSelected
                            ? "text-primary scale-110 drop-shadow-[0_0_8px_hsla(270,91%,55%,0.8)]"
                            : "text-muted-foreground group-hover:text-primary group-hover:scale-110"
                        )}
                      />
                      <span className={cn(
                        "font-medium transition-all duration-300 relative z-10",
                        isSelected ? "text-primary" : "group-hover:text-primary"
                      )}>
                        {mood.label}
                      </span>

                      {/* Lock-in indicator */}
                      {isSelected && (
                        <div className="absolute -top-1 -right-1 w-3 h-3 bg-primary rounded-full animate-pulse shadow-[0_0_10px_hsla(270,91%,55%,0.8)]" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent
                    side="bottom"
                    className="glass-panel border-primary/30 text-foreground max-w-[200px]"
                  >
                    <p className="text-sm">{mood.description}</p>
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        </TooltipProvider>
      </div>
    </section>
  );
};
