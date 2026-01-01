import { Button } from "./ui/button";
import { Loader2, Wand2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

interface GenerateButtonProps {
  onClick: () => void;
  isGenerating: boolean;
  disabled: boolean;
}

export const GenerateButton = ({ onClick, isGenerating, disabled }: GenerateButtonProps) => {
  return (
    <section className="py-8 px-4">
      <div className="flex justify-center">
        <Button
          variant="neon"
          size="xl"
          onClick={onClick}
          disabled={disabled || isGenerating}
          className={cn(
            "min-w-[280px] text-lg font-display tracking-wider relative overflow-hidden group",
            "transition-all duration-300",
            "hover:scale-105 active:scale-95",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          {/* Sparkle effects on hover */}
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <Sparkles className="absolute top-2 left-4 w-3 h-3 text-primary-foreground/50 animate-pulse" />
            <Sparkles className="absolute bottom-2 right-6 w-3 h-3 text-primary-foreground/50 animate-pulse animation-delay-200" />
          </div>
          
          {isGenerating ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Wand2 className={cn(
                "w-5 h-5 transition-transform duration-300",
                "group-hover:rotate-12"
              )} />
              Generate Music
            </>
          )}
        </Button>
      </div>
      
      {/* Hint text */}
      {disabled && !isGenerating && (
        <p className="text-center text-sm text-muted-foreground mt-3 animate-pulse">
          Select a mood above to enable generation
        </p>
      )}
    </section>
  );
};
