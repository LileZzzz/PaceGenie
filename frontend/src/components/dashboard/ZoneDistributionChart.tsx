import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity } from 'lucide-react';
import type { Run, TrainingZones } from '@/types';

const ZONES = [
  { label: 'Z1 Recovery',   color: '#6b7280' },
  { label: 'Z2 Aerobic',    color: '#10b981' },
  { label: 'Z3 Tempo',      color: '#f59e0b' },
  { label: 'Z4 Threshold',  color: '#f97316' },
  { label: 'Z5 Anaerobic',  color: '#ef4444' },
];

function getZoneIndex(hr: number, zones: TrainingZones): number {
  if (hr <= zones.zone1_max_hr) return 0;
  if (hr <= zones.zone2_max_hr) return 1;
  if (hr <= zones.zone3_max_hr) return 2;
  if (hr <= zones.zone4_max_hr) return 3;
  return 4;
}

interface ZoneDistributionChartProps {
  runs: Run[];
  zones: TrainingZones;
}

export function ZoneDistributionChart({ runs, zones }: ZoneDistributionChartProps) {
  const counts = [0, 0, 0, 0, 0];
  runs.forEach(r => { counts[getZoneIndex(r.avg_hr, zones)]++; });

  const total = runs.length;
  const data = ZONES.map((z, i) => ({
    name: z.label,
    color: z.color,
    value: counts[i],
    pct: total > 0 ? Math.round((counts[i] / total) * 100) : 0,
  })).filter(d => d.value > 0);

  return (
    <div className="p-5 bg-gradient-to-br from-card to-background border border-border/50 rounded-xl card-glow card-hover">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2.5 rounded-xl bg-blue-500/20">
          <Activity className="w-5 h-5 text-blue-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Zone Distribution</h3>
          <p className="text-xs text-muted-foreground">% of runs by avg HR zone · {total} runs</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="h-36 w-36 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={38}
                outerRadius={62}
                paddingAngle={2}
                dataKey="value"
                animationDuration={800}
              >
                {data.map((d, i) => (
                  <Cell key={i} fill={d.color} />
                ))}
              </Pie>
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="bg-card border border-border rounded-lg p-2 shadow-lg text-xs">
                      <p className="font-semibold text-white">{d.name}</p>
                      <p className="text-muted-foreground">{d.value} runs · {d.pct}%</p>
                    </div>
                  );
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="flex-1 space-y-2">
          {data.map((d, i) => (
            <div key={i} className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
              <span className="text-xs text-muted-foreground flex-1 truncate">{d.name}</span>
              <div className="flex items-center gap-1.5">
                <div className="w-12 h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full rounded-full" style={{ width: `${d.pct}%`, backgroundColor: d.color }} />
                </div>
                <span className="text-xs font-semibold text-white w-7 text-right">{d.pct}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
