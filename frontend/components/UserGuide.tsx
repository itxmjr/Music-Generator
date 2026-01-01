import { cn } from "@/lib/utils";
import { MousePointer, Wand2, Play, Download, X } from "lucide-react";
import { Button } from "./ui/button";

interface UserGuideProps {
  isVisible: boolean;
  onDismiss: () => void;
}

const steps = [
  { icon: MousePointer, label: "Select a mood", color: "text-neon-cyan" },
  { icon: Wand2, label: "Generate", color: "text-primary" },
  { icon: Play, label: "Play", color: "text-secondary" },
  { icon: Download, label: "Download", color: "text-green-400" },
];

export const UserGuide = ({ isVisible, onDismiss }: UserGuideProps) => {
  if (!isVisible) return null;

  return (
    <div className="py-6 px-4 animate-fade-in">
      <div className="max-w-lg mx-auto glass-panel p-4 relative">
        <Button
          variant="ghost"
          size="icon"
          onClick={onDismiss}
          className="absolute top-2 right-2 w-6 h-6 text-muted-foreground hover:text-foreground"
        >
          <X className="w-4 h-4" />
        </Button>
        
        <p className="text-sm text-muted-foreground text-center mb-4">
          🎵 First time here? Here's how it works:
        </p>
        
        <div className="flex items-center justify-center gap-2 flex-wrap">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <div key={index} className="flex items-center gap-2">
                <div className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-muted/50",
                  "transition-all duration-300 hover:scale-105"
                )}>
                  <Icon className={cn("w-4 h-4", step.color)} />
                  <span className="text-sm font-medium text-foreground">{step.label}</span>
                </div>
                {index < steps.length - 1 && (
                  <span className="text-muted-foreground">→</span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
