import { AlertTriangle } from 'lucide-react';

interface InjuryHistoryProps {
  injuries: string[];
}

export function InjuryHistory({ injuries }: InjuryHistoryProps) {
  if (!injuries || injuries.length === 0) return null;

  return (
    <div className="p-5 bg-gradient-to-br from-card to-background border border-border/50 rounded-xl card-glow card-hover">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2.5 rounded-xl bg-amber-500/20">
          <AlertTriangle className="w-5 h-5 text-amber-500" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Injury History</h3>
          <p className="text-xs text-muted-foreground">Past injuries and notes</p>
        </div>
      </div>

      <div className="space-y-2">
        {injuries.map((injury, index) => (
          <div 
            key={index} 
            className="p-3 bg-background/50 rounded-lg text-sm text-muted-foreground"
          >
            {injury}
          </div>
        ))}
      </div>
    </div>
  );
}
