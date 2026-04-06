import { useState } from 'react';
import { 
  History, 
  MapPin, 
  Timer, 
  Flame, 
  Heart, 
  ChevronRight,
  Calendar
} from 'lucide-react';
import type { Run } from '@/types';

interface RunHistoryProps {
  runs: Run[];
}

export function RunHistory({ runs }: RunHistoryProps) {
  const [selectedRun, setSelectedRun] = useState<number | null>(null);

  const formatDuration = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
  };

  const sortedRuns = [...runs].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  return (
    <div className="p-5 bg-gradient-to-br from-card to-background border border-border/50 rounded-xl card-glow card-hover">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-blue-500/20">
            <History className="w-5 h-5 text-blue-500" />
          </div>
          <div>
            <h3 className="font-semibold text-white">Run History</h3>
            <p className="text-xs text-muted-foreground">Recent {runs.length} runs</p>
          </div>
        </div>
      </div>

      <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
        {sortedRuns.map((run, index) => (
          <div
            key={`${run.date}-${run.type}`}
            onClick={() => setSelectedRun(selectedRun === index ? null : index)}
            className={`group relative p-4 rounded-xl cursor-pointer transition-all duration-300 ${
              selectedRun === index 
                ? 'bg-red-500/10 border border-red-500/30' 
                : 'bg-background/50 hover:bg-background border border-transparent'
            }`}
          >
            {/* Date and distance header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Calendar className="w-3 h-3" />
                  {run.date}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-lg font-bold text-white">{run.distance_km.toFixed(1)}</span>
                <span className="text-xs text-muted-foreground">km</span>
              </div>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-4 gap-2">
              <div className="flex items-center gap-1.5">
                <Timer className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-xs text-white">{formatDuration(run.duration_minutes)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <MapPin className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-xs text-white">{run.avg_pace_per_km}/km</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Flame className="w-3.5 h-3.5 text-orange-500" />
                <span className="text-xs text-white">{run.type}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Heart className="w-3.5 h-3.5 text-red-500" />
                <span className="text-xs text-white">{run.avg_hr}</span>
              </div>
            </div>

            {/* Expand indicator */}
            <div className={`absolute right-3 top-1/2 -translate-y-1/2 transition-transform ${
              selectedRun === index ? 'rotate-90' : ''
            }`}>
              <ChevronRight className="w-4 h-4 text-muted-foreground" />
            </div>

            {/* Expanded details */}
            {selectedRun === index && (
              <div className="mt-3 pt-3 border-t border-border/50 animate-fade-in">
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-background/80 rounded-lg p-2.5">
                    <p className="text-xs text-muted-foreground mb-1">Avg Heart Rate</p>
                    <p className="text-sm font-semibold text-white flex items-center gap-1">
                      <Heart className="w-3 h-3 text-red-500" />
                      {run.avg_hr} BPM
                    </p>
                  </div>
                  <div className="bg-background/80 rounded-lg p-2.5">
                    <p className="text-xs text-muted-foreground mb-1">Max Heart Rate</p>
                    <p className="text-sm font-semibold text-white flex items-center gap-1">
                      <Heart className="w-3 h-3 text-red-500 animate-heartbeat" />
                      {run.max_hr} BPM
                    </p>
                  </div>
                  <div className="bg-background/80 rounded-lg p-2.5">
                    <p className="text-xs text-muted-foreground mb-1">Elevation Gain</p>
                    <p className="text-sm font-semibold text-white">
                      {run.elevation_gain_m} m
                    </p>
                  </div>
                  <div className="bg-background/80 rounded-lg p-2.5">
                    <p className="text-xs text-muted-foreground mb-1">Run Type</p>
                    <p className="text-sm font-semibold text-white capitalize">
                      {run.type}
                    </p>
                  </div>
                </div>
                {run.notes && (
                  <div className="mt-2 text-xs text-muted-foreground italic">
                    "{run.notes}"
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
