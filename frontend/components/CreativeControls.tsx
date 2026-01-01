import { Slider } from "./ui/slider";
import { Thermometer, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

interface CreativeControlsProps {
  temperature: number;
  onTemperatureChange: (value: number) => void;
  bpm: number;
  onBpmChange: (value: number) => void;
  disabled?: boolean;
}

export const CreativeControls = ({
  temperature,
  onTemperatureChange,
  bpm,
  onBpmChange,
  disabled
}: CreativeControlsProps) => {
  return (
    <section className="py-8 px-4">
      <div className="max-w-md mx-auto glass-panel p-6 space-y-6">
        <h3 className="font-display text-lg font-semibold text-center text-foreground">
          Creative <span className="text-primary">Controls</span>
        </h3>

        {/* Temperature slider */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Thermometer className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium text-foreground">Creativity</span>
            </div>
            <span className={cn(
              "text-sm font-mono px-2 py-0.5 rounded",
              temperature < 0.4 ? "text-neon-cyan bg-neon-cyan/10" :
                temperature < 0.7 ? "text-primary bg-primary/10" :
                  "text-secondary bg-secondary/10"
            )}>
              {temperature < 0.4 ? "Conservative" : temperature < 0.7 ? "Balanced" : "Experimental"}
            </span>
          </div>
          <Slider
            value={[temperature * 100]}
            max={100}
            step={1}
            disabled={disabled}
            onValueChange={(value) => onTemperatureChange(value[0] / 100)}
            className="cursor-pointer"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Safe</span>
            <span>Creative</span>
          </div>
        </div>

        {/* BPM slider */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-secondary animate-pulse" />
              <span className="text-sm font-medium text-foreground">Tempo (BPM)</span>
            </div>
            <span className="text-xl font-mono font-bold text-secondary">{bpm}</span>
          </div>
          <Slider
            value={[bpm]}
            min={40}
            max={240}
            step={1}
            disabled={disabled}
            onValueChange={(value) => onBpmChange(value[0])}
            className="cursor-pointer"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Slow</span>
            <span>Fast</span>
          </div>
        </div>

        <p className="text-xs text-muted-foreground text-center">
          Adjust creativity and tempo to influence AI composition style
        </p>
      </div>
    </section>
  );
};
