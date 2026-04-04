import { UserProfile } from './UserProfile';
import { WeeklySummary } from './WeeklySummary';
import { HeartRateChart } from './HeartRateChart';
import { RunHistory } from './RunHistory';
import { RaceCard } from './RaceCard';
import { ZoneCard } from './ZoneCard';
import { PBCard } from './PBCard';
import { InjuryHistory } from './InjuryHistory';
import type { MockGarminData } from '@/types';

interface DashboardProps {
  data: MockGarminData;
}

export function Dashboard({ data }: DashboardProps) {
  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* User Profile */}
      <UserProfile profile={data.profile} />

      {/* Weekly Summary - from weekly_summary */}
      <WeeklySummary summary={data.weekly_summary} />

      {/* Heart Rate Chart - from recent_runs */}
      <HeartRateChart runs={data.recent_runs} profile={data.profile} zones={data.training_zones} />

      {/* Run History - from recent_runs */}
      <RunHistory runs={data.recent_runs} />

      {/* Personal Bests - from personal_bests */}
      <PBCard pbs={data.personal_bests} />

      {/* Upcoming Race - from upcoming_races */}
      <RaceCard races={data.upcoming_races} />

      {/* Training Zones - from training_zones */}
      <ZoneCard zones={data.training_zones} />

      {/* Injury History - from injury_history */}
      <InjuryHistory injuries={data.injury_history} />

      {/* Bottom spacing */}
      <div className="h-4" />
    </div>
  );
}
