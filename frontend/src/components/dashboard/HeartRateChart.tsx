import { Heart, Activity, TrendingUp, TrendingDown } from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { Run, TrainingZones } from '@/types';

interface HeartRateChartProps {
  runs: Run[];
  profile: { resting_hr: number; max_hr: number };
  zones: TrainingZones;
}

export function HeartRateChart({ runs, profile, zones }: HeartRateChartProps) {
  const sortedRuns = [...runs].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  const recentRuns = sortedRuns.slice(-10);
  
  const data = recentRuns.map(run => ({
    time: run.date.slice(5),
    bpm: run.avg_hr,
    max: run.max_hr,
  }));

  const avgBpm = data.length > 0 ? Math.round(data.reduce((sum, d) => sum + d.bpm, 0) / data.length) : profile.resting_hr;
  const maxBpm = data.length > 0 ? Math.max(...data.map(d => d.max), profile.max_hr) : profile.max_hr;
  const minBpm = data.length > 0 ? Math.min(...data.map(d => d.bpm), profile.resting_hr) : profile.resting_hr;

  return (
    <div className="p-5 bg-gradient-to-br from-card to-background border border-border/50 rounded-xl card-glow card-hover">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="p-2.5 rounded-xl bg-red-500/20">
              <Heart className="w-5 h-5 text-red-500 animate-heartbeat" />
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse" />
          </div>
          <div>
            <h3 className="font-semibold text-white">Heart Rate Monitor</h3>
            <p className="text-xs text-muted-foreground">Avg HR per run · last 10 runs</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="flex items-center gap-1 justify-end">
              <span className="text-2xl font-bold text-red-500">{avgBpm}</span>
              <span className="text-xs text-muted-foreground">BPM</span>
            </div>
            <span className="text-xs text-emerald-400">
              Avg HR
            </span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
            <defs>
              <linearGradient id="heartRateGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
            <XAxis 
              dataKey="time" 
              stroke="hsl(var(--muted-foreground))" 
              fontSize={10}
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              stroke="hsl(var(--muted-foreground))" 
              fontSize={10}
              tickLine={false}
              axisLine={false}
              domain={[40, 200]}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-card border border-border rounded-lg p-2 shadow-lg">
                      <p className="text-xs text-muted-foreground">{data.time}</p>
                      <p className="text-sm font-semibold text-red-400">
                        {data.bpm} BPM
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            {/* Zone boundaries derived from training_zones data */}
            <ReferenceLine y={zones.zone2_max_hr} stroke="#10b981" strokeDasharray="3 3" opacity={0.5} label={{ value: 'Z2', position: 'right', fontSize: 9, fill: '#10b981' }} />
            <ReferenceLine y={zones.zone3_max_hr} stroke="#3b82f6" strokeDasharray="3 3" opacity={0.5} label={{ value: 'Z3', position: 'right', fontSize: 9, fill: '#3b82f6' }} />
            <ReferenceLine y={zones.zone4_max_hr} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} label={{ value: 'Z4', position: 'right', fontSize: 9, fill: '#ef4444' }} />
            <Area
              type="monotone"
              dataKey="bpm"
              stroke="#ef4444"
              strokeWidth={2}
              fill="url(#heartRateGradient)"
              animationDuration={1000}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-border/50">
        <div className="text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <Activity className="w-3 h-3" />
            Avg
          </div>
          <span className="text-lg font-semibold text-white">{avgBpm}</span>
        </div>
        <div className="text-center border-x border-border/50">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <TrendingUp className="w-3 h-3" />
            Max
          </div>
          <span className="text-lg font-semibold text-red-400">{maxBpm}</span>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground mb-1">
            <TrendingDown className="w-3 h-3" />
            Min
          </div>
          <span className="text-lg font-semibold text-emerald-400">{minBpm}</span>
        </div>
      </div>
    </div>
  );
}
