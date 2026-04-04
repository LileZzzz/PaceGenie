import { Route, Timer, Calendar, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { WeeklySummary as WeeklySummaryType } from '@/types';

interface WeeklySummaryProps {
  summary: WeeklySummaryType;
}

export function WeeklySummary({ summary }: WeeklySummaryProps) {
  const kmDelta = summary.last_week_km > 0 
    ? ((summary.this_week_km - summary.last_week_km) / summary.last_week_km) * 100 
    : 0;
  
  const runsDelta = summary.last_week_runs > 0
    ? ((summary.this_week_runs - summary.last_week_runs) / summary.last_week_runs) * 100
    : 0;

  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="w-3 h-3" />;
    if (change < 0) return <TrendingDown className="w-3 h-3" />;
    return <Minus className="w-3 h-3" />;
  };

  const getTrendColor = (change: number) => {
    if (change > 0) return 'text-emerald-400';
    if (change < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  return (
    <div className="p-5 bg-gradient-to-br from-card to-background border border-border/50 rounded-xl card-glow card-hover">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2.5 rounded-xl bg-orange-500/20">
          <TrendingUp className="w-5 h-5 text-orange-500" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Weekly Summary</h3>
          <p className="text-xs text-muted-foreground">This week vs last week</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {/* This Week Distance */}
        <div className="bg-background/50 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Route className="w-3.5 h-3.5" />
              This Week
            </div>
            <div className={`flex items-center gap-1 text-xs ${getTrendColor(kmDelta)}`}>
              {getTrendIcon(kmDelta)}
              {Math.abs(Math.round(kmDelta))}%
            </div>
          </div>
          <div className="flex items-baseline gap-0.5">
            <span className="text-2xl font-bold text-white">{summary.this_week_km.toFixed(1)}</span>
            <span className="text-xs text-muted-foreground">km</span>
          </div>
        </div>

        {/* Last Week Distance */}
        <div className="bg-background/50 rounded-xl p-4">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-2">
            <Route className="w-3.5 h-3.5" />
            Last Week
          </div>
          <div className="flex items-baseline gap-0.5">
            <span className="text-2xl font-bold text-white">{summary.last_week_km.toFixed(1)}</span>
            <span className="text-xs text-muted-foreground">km</span>
          </div>
        </div>

        {/* This Week Runs */}
        <div className="bg-background/50 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Calendar className="w-3.5 h-3.5" />
              Runs
            </div>
            <div className={`flex items-center gap-1 text-xs ${getTrendColor(runsDelta)}`}>
              {getTrendIcon(runsDelta)}
              {Math.abs(Math.round(runsDelta))}%
            </div>
          </div>
          <div className="flex items-baseline gap-0.5">
            <span className="text-2xl font-bold text-white">{summary.this_week_runs}</span>
            <span className="text-xs text-muted-foreground">runs</span>
          </div>
        </div>

        {/* 4 Week Average */}
        <div className="bg-background/50 rounded-xl p-4">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-2">
            <Timer className="w-3.5 h-3.5" />
            4-Week Avg
          </div>
          <div className="flex items-baseline gap-0.5">
            <span className="text-2xl font-bold text-white">{summary['4_week_avg_km'].toFixed(1)}</span>
            <span className="text-xs text-muted-foreground">km</span>
          </div>
        </div>
      </div>
    </div>
  );
}
