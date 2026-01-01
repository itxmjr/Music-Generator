import { Music, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  hasSelectedMood: boolean;
}

export const EmptyState = ({ hasSelectedMood }: EmptyStateProps) => {
  return (
    <section className="py-12 px-4 animate-fade-in">
      <div className="max-w-2xl mx-auto glass-panel p-8 text-center space-y-6">
        {/* Animated icon */}
        <div className="relative w-24 h-24 mx-auto">
          <div className="absolute inset-0 bg-primary/20 rounded-full animate-ping" />
          <div className="relative flex items-center justify-center w-24 h-24 bg-muted rounded-full">
            <Music className="w-12 h-12 text-muted-foreground" />
          </div>
          <Sparkles className="absolute -top-2 -right-2 w-6 h-6 text-primary animate-pulse" />
        </div>

        {/* Message */}
        <div className="space-y-2">
          <h3 className="text-xl font-display font-semibold text-foreground">
            {hasSelectedMood ? "Ready to Create" : "No Music Yet"}
          </h3>
          <p className="text-muted-foreground max-w-sm mx-auto">
            {hasSelectedMood 
              ? "Click 'Generate Music' to create your unique AI-composed track"
              : "Select a mood above to get started with AI music generation"
            }
          </p>
        </div>

        {/* Visual hint */}
        <div className={cn(
          "flex items-center justify-center gap-2 text-sm",
          hasSelectedMood ? "text-primary" : "text-muted-foreground"
        )}>
          <div className={cn(
            "w-2 h-2 rounded-full",
            hasSelectedMood ? "bg-primary animate-pulse" : "bg-muted-foreground"
          )} />
          <span>{hasSelectedMood ? "Mood selected — you're all set!" : "Waiting for mood selection..."}</span>
        </div>
      </div>
    </section>
  );
};
