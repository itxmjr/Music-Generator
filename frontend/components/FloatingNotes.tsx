import { Music, Music2, Music3, Music4 } from "lucide-react";

const notes = [
  { Icon: Music, delay: "0s", x: "8%", y: "25%" },
  { Icon: Music2, delay: "2s", x: "85%", y: "20%" },
  { Icon: Music3, delay: "4s", x: "20%", y: "75%" },
  { Icon: Music4, delay: "6s", x: "75%", y: "70%" },
];

export const FloatingNotes = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {notes.map((note, index) => {
        const NoteIcon = note.Icon;
        return (
          <div
            key={index}
            className="absolute animate-float-slow opacity-15"
            style={{
              left: note.x,
              top: note.y,
              animationDelay: note.delay,
              animationDuration: `${12 + index * 2}s`,
            }}
          >
            <NoteIcon 
              className="text-primary/60" 
              size={20 + (index % 2) * 6}
            />
          </div>
        );
      })}
    </div>
  );
};
