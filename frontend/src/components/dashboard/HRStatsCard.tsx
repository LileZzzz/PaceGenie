import { Heart } from 'lucide-react';
import type { Profile, Run } from '@/types';

interface HRStatsCardProps {
  profile: Profile;
  runs: Run[];
}

export function HRStatsCard({ profile, runs }: HRStatsCardProps) {
  const latestHR = runs.length > 0 ? runs[runs.length - 1].avg_hr : '--';

  return (
    <div className="mx-4 p-4 rounded-xl bg-white/5 border border-white/5">
      <div className="text-xs text-slate-400 uppercase tracking-wider font-medium mb-3 flex items-center gap-2">
        <Heart className="w-3 h-3" />
        Heart Rate Snapshot
      </div>
      <div className="flex justify-between">
        <div className="text-center">
          <div className="text-xl font-bold text-white">{latestHR}</div>
          <div className="text-xs text-slate-500">Latest Avg</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-white">{profile.resting_hr}</div>
          <div className="text-xs text-slate-500">Resting</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-white">{profile.max_hr}</div>
          <div className="text-xs text-slate-500">Max</div>
        </div>
      </div>
    </div>
  );
}
