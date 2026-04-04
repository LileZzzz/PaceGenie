import { Calendar, Target, Flag } from 'lucide-react';
import type { UpcomingRace } from '@/types';

interface RaceCardProps {
  races: UpcomingRace[];
}

export function RaceCard({ races }: RaceCardProps) {
  if (!races || races.length === 0) return null;
  
  const race = races[0];
  const goalTime = `${Math.floor(race.goal_time_minutes / 60)}:${(race.goal_time_minutes % 60).toString().padStart(2, '0')}`;

  return (
    <div className="p-5 bg-gradient-to-br from-amber-500/10 to-amber-900/20 border border-amber-500/20 rounded-xl card-glow card-hover">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2.5 rounded-xl bg-amber-500/20">
          <Flag className="w-5 h-5 text-amber-500" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Upcoming Race</h3>
          <p className="text-xs text-muted-foreground">Next challenge</p>
        </div>
      </div>
      
      <div className="text-white font-semibold text-lg mb-3">{race.name}</div>
      
      <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
        <span className="flex items-center gap-1">
          <Calendar className="w-4 h-4" />
          {race.date}
        </span>
        <span>{race.distance_km} km</span>
        <span className="flex items-center gap-1">
          <Target className="w-4 h-4" />
          Goal {goalTime}
        </span>
      </div>
    </div>
  );
}
