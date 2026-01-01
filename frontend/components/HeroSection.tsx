import { FloatingNotes } from "./FloatingNotes";
import { AnimatedEqualizer } from "./AnimatedEqualizer";
import { Sparkles } from "lucide-react";

export const HeroSection = () => {
  return (
    <section className="relative min-h-[50vh] flex flex-col items-center justify-center py-16 overflow-hidden">
      {/* Ambient background layers */}
      <div className="absolute inset-0">
        {/* Deep gradient base */}
        <div className="absolute inset-0 bg-gradient-to-b from-background via-background to-background" />
        
        {/* Soft purple glow - top left */}
        <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-primary/8 rounded-full blur-[120px] animate-pulse-slow" />
        
        {/* Soft pink glow - bottom right */}
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-secondary/6 rounded-full blur-[100px] animate-pulse-slow" style={{ animationDelay: "2s" }} />
        
        {/* Center ambient glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-primary/5 rounded-full blur-[150px]" />
      </div>
      
      {/* Floating music notes */}
      <FloatingNotes />
      
      {/* Decorative orbs */}
      <div className="absolute top-20 right-20 w-2 h-2 bg-primary rounded-full animate-pulse-slow opacity-60" />
      <div className="absolute top-40 right-40 w-1 h-1 bg-secondary rounded-full animate-pulse-slow opacity-40" style={{ animationDelay: "1s" }} />
      <div className="absolute bottom-32 left-20 w-1.5 h-1.5 bg-primary rounded-full animate-pulse-slow opacity-50" style={{ animationDelay: "2s" }} />
      <div className="absolute top-32 left-32 w-1 h-1 bg-secondary rounded-full animate-pulse-slow opacity-30" style={{ animationDelay: "3s" }} />
      
      {/* Sparkle decorations */}
      <Sparkles className="absolute top-24 left-1/4 w-4 h-4 text-primary/20 animate-pulse-slow" style={{ animationDelay: "0.5s" }} />
      <Sparkles className="absolute bottom-40 right-1/4 w-3 h-3 text-secondary/15 animate-pulse-slow" style={{ animationDelay: "1.5s" }} />

      {/* Content */}
      <div className="relative z-10 text-center space-y-6 px-4">
        <h1 className="font-display text-5xl md:text-7xl lg:text-8xl font-bold tracking-wider">
          <span className="animate-breathe bg-gradient-to-r from-primary via-secondary to-primary bg-clip-text text-transparent">
            AI Music
          </span>
          <br />
          <span className="text-foreground/80 font-semibold tracking-tight">Generator</span>
        </h1>
        
        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
          Turn your imagination into music with the power of AI.
          <br className="hidden md:block" />
          Select a mood and let the neural network compose for you.
        </p>

        {/* Animated equalizer preview */}
        <div className="pt-8">
          <AnimatedEqualizer isPlaying barCount={30} className="h-12 opacity-60" />
        </div>
      </div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
    </section>
  );
};
