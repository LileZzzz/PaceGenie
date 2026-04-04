import { TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { WeeklySummary, Profile } from '@/types';

interface WeeklyCardProps {
  weekly: WeeklySummary;
  profile: Profile;
  className?: string;
}

export function WeeklyCard({ weekly, profile, className }: WeeklyCardProps) {
  const thisWeek = weekly.this_week_km;
  const lastWeek = weekly.last_week_km;
  const target = profile.weekly_target_km || 50;
  const progress = Math.min(Math.round((thisWeek / target) * 100), 100);
  const delta = thisWeek - lastWeek;
  const isPositive = delta >= 0;

  return (
    <div className={cn("mx-4 p-5 rounded-2xl bg-gradient-to-br from-emerald-500/15 to-emerald-900/20 border border-emerald-500/20", className)}>
      <div className="text-xs text-emerald-400/80 uppercase tracking-wider font-medium mb-2">
        This Week
      </div>
      
      <div className="flex items-baseline gap-2 mb-3">
        <span className="text-4xl font-bold text-white">{thisWeek.toFixed(1)}</span>
        <span className="text-slate-400">km</span>
        <span className={cn(
          "ml-auto px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1",
          isPositive ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"
        )}>
          {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          {isPositive ? '+' : ''}{delta.toFixed(1)} km vs last week
        </span>
      </div>
      
      <div className="h-2 bg-black/30 rounded-full overflow-hidden mb-3">
        <div 
          className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
      
      <div className="flex justify-between text-sm text-slate-400">
        <span>{weekly.this_week_runs} runs</span>
        <span>Target {target} km</span>
      </div>
    </div>
  );
}
