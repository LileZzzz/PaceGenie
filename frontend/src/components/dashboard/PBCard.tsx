import { Award } from 'lucide-react';
import type { PersonalBests } from '@/types';

interface PBCardProps {
  pbs: PersonalBests;
}

export function PBCard({ pbs }: PBCardProps) {
  const formatTime = (minutes: number) => {
    const hrs = Math.floor(minutes / 60);
    const mins = Math.floor(minutes % 60);
    const secs = Math.floor((minutes % 1) * 60);
    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="p-5 bg-gradient-to-br from-card to-background border border-border/50 rounded-xl card-glow card-hover">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2.5 rounded-xl bg-yellow-500/20">
          <Award className="w-5 h-5 text-yellow-500" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Personal Bests</h3>
          <p className="text-xs text-muted-foreground">Your best records</p>
        </div>
      </div>
      
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-background/50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-white">{formatTime(pbs['5k_minutes'])}</div>
          <div className="text-xs text-muted-foreground">5K</div>
        </div>
        <div className="bg-background/50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-white">{formatTime(pbs['10k_minutes'])}</div>
          <div className="text-xs text-muted-foreground">10K</div>
        </div>
        <div className="bg-background/50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-white">{formatTime(pbs.half_marathon_minutes)}</div>
          <div className="text-xs text-muted-foreground">Half</div>
        </div>
        <div className="bg-background/50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-white">{pbs.longest_run_km}k</div>
          <div className="text-xs text-muted-foreground">Longest</div>
        </div>
      </div>
    </div>
  );
}
