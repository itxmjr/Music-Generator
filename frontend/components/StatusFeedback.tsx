import { cn } from "@/lib/utils";
import { CheckCircle2, AlertCircle, Info, Loader2 } from "lucide-react";

export type StatusType = "idle" | "info" | "loading" | "success" | "error";

interface StatusFeedbackProps {
  status: StatusType;
  message: string;
}

const statusConfig = {
  idle: {
    icon: Info,
    className: "text-muted-foreground border-border",
  },
  info: {
    icon: Info,
    className: "text-neon-cyan border-neon-cyan/30",
  },
  loading: {
    icon: Loader2,
    className: "text-primary border-primary/30",
  },
  success: {
    icon: CheckCircle2,
    className: "text-green-400 border-green-400/30",
  },
  error: {
    icon: AlertCircle,
    className: "text-destructive border-destructive/30",
  },
};

export const StatusFeedback = ({ status, message }: StatusFeedbackProps) => {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <div className="py-4 px-4">
      <div
        className={cn(
          "max-w-md mx-auto glass-panel border px-4 py-3 flex items-center gap-3 transition-all duration-300",
          config.className
        )}
      >
        <Icon
          className={cn(
            "w-5 h-5 flex-shrink-0",
            status === "loading" && "animate-spin"
          )}
        />
        <span className="text-sm font-medium">{message}</span>
      </div>
    </div>
  );
};
