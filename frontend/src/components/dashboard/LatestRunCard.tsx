import { MapPin, Activity, TrendingUp, Calendar } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Run } from '@/types';

interface LatestRunCardProps {
  run: Run;
  className?: string;
}

export function LatestRunCard({ run, className }: LatestRunCardProps) {
  const typeColors: Record<string, string> = {
    easy: 'bg-blue-500/20 text-blue-400',
    tempo: 'bg-orange-500/20 text-orange-400',
    long: 'bg-purple-500/20 text-purple-400',
    interval: 'bg-red-500/20 text-red-400',
    recovery: 'bg-emerald-500/20 text-emerald-400',
  };

  return (
    <div className={cn("mx-4 p-4 rounded-xl bg-white/5 border border-white/5", className)}>
      <div className="flex justify-between items-center mb-3">
        <span className="text-xs text-slate-400 uppercase tracking-wider font-medium">Latest Run</span>
        <span className={cn("px-2 py-1 rounded-full text-xs font-medium capitalize", typeColors[run.type] || 'bg-slate-500/20 text-slate-400')}>
          {run.type}
        </span>
      </div>
      
      <div className="flex items-baseline gap-2 mb-2">
        <span className="text-3xl font-bold text-white">{run.distance_km.toFixed(1)}</span>
        <span className="text-slate-400">km</span>
      </div>
      
      <div className="flex gap-4 text-sm text-slate-400 mb-2">
        <span className="flex items-center gap-1">
          <TrendingUp className="w-3 h-3" />
          {run.avg_pace_per_km} /km
        </span>
        <span className="flex items-center gap-1">
          <Activity className="w-3 h-3" />
          {run.avg_hr} bpm
        </span>
        <span className="flex items-center gap-1">
          <MapPin className="w-3 h-3" />
          {run.elevation_gain_m} m
        </span>
      </div>
      
      <div className="flex items-center gap-1 text-xs text-slate-500">
        <Calendar className="w-3 h-3" />
        {run.date}
      </div>
      
      {run.notes && (
        <div className="mt-3 text-xs text-slate-400 italic border-t border-white/5 pt-2">
          "{run.notes}"
        </div>
      )}
    </div>
  );
}
