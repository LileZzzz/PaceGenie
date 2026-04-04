import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Run } from '@/types';

interface RunsTableProps {
  runs: Run[];
  className?: string;
}

export function RunsTable({ runs, className }: RunsTableProps) {
  const [expanded, setExpanded] = useState(false);
  const sortedRuns = [...runs].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  const displayRuns = expanded ? sortedRuns : sortedRuns.slice(0, 5);

  const typeColors: Record<string, string> = {
    easy: 'text-blue-400',
    tempo: 'text-orange-400',
    long: 'text-purple-400',
    interval: 'text-red-400',
    recovery: 'text-emerald-400',
  };

  return (
    <div className={cn("mx-4", className)}>
      <div className="text-xs text-slate-400 uppercase tracking-wider font-medium mb-3">
        Recent Runs ({runs.length})
      </div>
      
      <div className="rounded-xl overflow-hidden bg-white/5 border border-white/5">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/5 text-slate-400">
              <th className="text-left py-2 px-3 font-medium">Date</th>
              <th className="text-left py-2 px-3 font-medium">Type</th>
              <th className="text-right py-2 px-3 font-medium">Dist</th>
              <th className="text-right py-2 px-3 font-medium">Pace</th>
              <th className="text-right py-2 px-3 font-medium">HR</th>
            </tr>
          </thead>
          <tbody>
            {displayRuns.map((run, index) => (
              <tr key={index} className="border-b border-white/5 last:border-0 hover:bg-white/5 transition-colors">
                <td className="py-2 px-3 text-slate-300">{run.date.slice(5)}</td>
                <td className={cn("py-2 px-3 capitalize", typeColors[run.type] || 'text-slate-400')}>
                  {run.type}
                </td>
                <td className="py-2 px-3 text-right text-white">{run.distance_km.toFixed(1)}k</td>
                <td className="py-2 px-3 text-right text-slate-400">{run.avg_pace_per_km}</td>
                <td className="py-2 px-3 text-right text-slate-400">{run.avg_hr}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {runs.length > 5 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full mt-2 py-2 text-xs text-slate-400 hover:text-white flex items-center justify-center gap-1 transition-colors"
        >
          {expanded ? (
            <>
              <ChevronUp className="w-3 h-3" />
              Show less
            </>
          ) : (
            <>
              <ChevronDown className="w-3 h-3" />
              Show all {runs.length} runs
            </>
          )}
        </button>
      )}
    </div>
  );
}
