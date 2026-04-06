import { TrendingUp, Heart, Target } from 'lucide-react';
import type { Profile } from '@/types';

interface UserProfileProps {
  profile: Profile;
}

export function UserProfile({ profile }: UserProfileProps) {
  return (
    <div className="p-5 bg-gradient-to-br from-card to-background border-b border-border/50">
      <div className="flex items-center mb-4">
        <div className="flex items-center gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center text-3xl border-2 border-red-500/30 select-none">
              🏃
            </div>
            <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center border-2 border-background">
              <TrendingUp className="w-3 h-3 text-white" />
            </div>
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">{profile.name}</h2>
            <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-red-500/20 text-red-400 text-xs font-medium mt-1">
              <Target className="w-3 h-3" />
              Weekly Target: {profile.weekly_target_km}km
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="bg-background/50 rounded-xl p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
            <Heart className="w-3 h-3 text-red-500" />
            Max HR
          </div>
          <div className="flex items-baseline gap-0.5">
            <span className="text-xl font-bold text-white">{profile.max_hr}</span>
            <span className="text-xs text-muted-foreground">bpm</span>
          </div>
        </div>
        <div className="bg-background/50 rounded-xl p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
            <Heart className="w-3 h-3 text-emerald-500" />
            Resting HR
          </div>
          <div className="flex items-baseline gap-0.5">
            <span className="text-xl font-bold text-white">{profile.resting_hr}</span>
            <span className="text-xs text-muted-foreground">bpm</span>
          </div>
        </div>
        <div className="bg-background/50 rounded-xl p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
            Age
          </div>
          <div className="flex items-baseline gap-0.5">
            <span className="text-xl font-bold text-white">{profile.age}</span>
            <span className="text-xs text-muted-foreground">years</span>
          </div>
        </div>
      </div>
    </div>
  );
}
