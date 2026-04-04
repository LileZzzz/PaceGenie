import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from 'recharts';
import { calculateWeeklyData } from '@/lib/utils';
import type { Run } from '@/types';

interface VolumeChartProps {
  runs: Run[];
  weeks?: number;
}

export function VolumeChart({ runs, weeks = 4 }: VolumeChartProps) {
  const data = calculateWeeklyData(runs, weeks);
  const maxValue = Math.max(...data.map((d: { week: string; distance: number }) => d.distance), 1);

  return (
    <div className="mx-4">
      <div className="text-xs text-slate-400 uppercase tracking-wider font-medium mb-3">
        4-Week Volume
      </div>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
            <XAxis 
              dataKey="week" 
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
            />
            <YAxis 
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
            />
            <Bar dataKey="distance" radius={[4, 4, 0, 0]}>
              {data.map((entry: { week: string; distance: number }, index: number) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.distance > 0 
                    ? `hsl(${140 + (entry.distance / maxValue) * 40}, 70%, 45%)` 
                    : '#374151'
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
