import { cn } from '@/lib/utils';
import type { Profile } from '@/types';

interface ProfileCardProps {
  profile: Profile;
  className?: string;
}

export function ProfileCard({ profile, className }: ProfileCardProps) {
  const initial = profile.name?.[0]?.toUpperCase() || 'R';
  
  return (
    <div className={cn("p-5 border-b border-white/5", className)}>
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-emerald-500 to-emerald-700 flex items-center justify-center text-white text-xl font-bold shadow-lg shadow-emerald-500/20">
          {initial}
        </div>
        <div>
          <div className="text-xs text-emerald-400/80 uppercase tracking-wider font-medium">Runner Profile</div>
          <div className="text-white font-semibold text-lg">{profile.name}</div>
          <div className="text-slate-400 text-sm">
            Age {profile.age} · Mock Garmin data
          </div>
        </div>
      </div>
    </div>
  );
}
