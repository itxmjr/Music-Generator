import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

interface AnimatedEqualizerProps {
  isPlaying?: boolean;
  barCount?: number;
  className?: string;
}

export const AnimatedEqualizer = ({
  isPlaying = false,
  barCount = 20,
  className
}: AnimatedEqualizerProps) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className={cn("flex items-end justify-center gap-1 h-16", className)}>
      {Array.from({ length: barCount }).map((_, index) => (
        <div
          key={index}
          className={cn(
            "w-1 rounded-full transition-all duration-150",
            isPlaying
              ? "bg-gradient-to-t from-primary via-secondary to-neon-cyan"
              : "bg-muted"
          )}
          style={{
            height: isPlaying && mounted ? `${Math.random() * 80 + 20}%` : "20%",
            animationName: isPlaying && mounted ? "equalizer" : "none",
            animationDuration: isPlaying && mounted ? `${1.5 + Math.random() * 1}s` : "0s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite",
            animationDelay: `${index * 0.1}s`,
          }}
        />
      ))}
    </div>
  );
};
