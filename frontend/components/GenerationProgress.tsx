import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Sparkles, Music, Wand2, CheckCircle2 } from "lucide-react";

interface GenerationProgressProps {
  isGenerating: boolean;
}

const generationSteps = [
  { message: "Analyzing mood...", icon: Sparkles, duration: 1500 },
  { message: "Composing melody...", icon: Music, duration: 2000 },
  { message: "Arranging harmony...", icon: Wand2, duration: 2000 },
  { message: "Finalizing track...", icon: CheckCircle2, duration: 1500 },
];

export const GenerationProgress = ({ isGenerating }: GenerationProgressProps) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!isGenerating) {
      setCurrentStep(0);
      setProgress(0);
      return;
    }

    // Progress bar animation
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 95) return prev;
        return prev + Math.random() * 3;
      });
    }, 100);

    // Step cycling
    let stepTimeout: number;
    const cycleSteps = (step: number) => {
      if (step < generationSteps.length) {
        setCurrentStep(step);
        stepTimeout = window.setTimeout(() => cycleSteps(step + 1), generationSteps[step].duration);
      }
    };
    cycleSteps(0);

    return () => {
      clearInterval(progressInterval);
      clearTimeout(stepTimeout);
    };
  }, [isGenerating]);

  if (!isGenerating) return null;

  const CurrentIcon = generationSteps[currentStep]?.icon || Sparkles;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-md animate-fade-in">
      <div className="max-w-md w-full mx-4 space-y-8">
        {/* Animated equalizer during generation */}
        <div className="flex items-end justify-center gap-1 h-32">
          {Array.from({ length: 32 }).map((_, index) => (
            <div
              key={index}
              className="w-1.5 rounded-full bg-gradient-to-t from-primary via-secondary to-neon-cyan"
              style={{
                height: mounted ? `${20 + Math.random() * 80}%` : "50%",
                animationName: "equalizer-aggressive",
                animationDuration: mounted ? `${0.3 + Math.random() * 0.4}s` : "0.5s",
                animationTimingFunction: "ease-in-out",
                animationIterationCount: "infinite",
                animationDelay: `${index * 0.03}s`,
              }}
            />
          ))}
        </div>

        {/* Progress bar */}
        <div className="relative h-2 bg-muted rounded-full overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary via-secondary to-primary bg-[length:200%_100%] animate-shimmer rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
          {/* Glow effect */}
          <div
            className="absolute inset-y-0 left-0 bg-primary/50 blur-sm rounded-full"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Current step */}
        <div className="flex items-center justify-center gap-3 text-center">
          <CurrentIcon className="w-6 h-6 text-primary animate-pulse" />
          <span className="text-lg font-display text-foreground animate-pulse">
            {generationSteps[currentStep]?.message}
          </span>
        </div>

        {/* Step indicators */}
        <div className="flex justify-center gap-2">
          {generationSteps.map((step, index) => (
            <div
              key={index}
              className={cn(
                "w-2 h-2 rounded-full transition-all duration-300",
                index <= currentStep
                  ? "bg-primary shadow-[0_0_8px_hsla(270,91%,55%,0.8)]"
                  : "bg-muted"
              )}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
