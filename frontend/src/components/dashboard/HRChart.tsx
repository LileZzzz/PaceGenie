import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import type { Run } from '@/types';

interface HRChartProps {
  runs: Run[];
  limit?: number;
}

export function HRChart({ runs, limit = 6 }: HRChartProps) {
  const sortedRuns = [...runs].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  const recentRuns = sortedRuns.slice(-limit);
  
  const data = recentRuns.map(run => ({
    date: new Date(run.date).toLocaleDateString('en-US', { month: '2-digit', day: '2-digit' }),
    hr: run.avg_hr,
  }));

  return (
    <div className="mx-4">
      <div className="text-xs text-slate-400 uppercase tracking-wider font-medium mb-3">
        Heart Rate Trend
      </div>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
            <defs>
              <linearGradient id="hrGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis 
              dataKey="date" 
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
            />
            <YAxis 
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
            />
            <Area 
              type="monotone" 
              dataKey="hr" 
              stroke="#10b981" 
              strokeWidth={2}
              fill="url(#hrGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
