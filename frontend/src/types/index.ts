export interface Profile {
  name: string;
  age: number;
  max_hr: number;
  resting_hr: number;
  weekly_target_km: number;
}

export interface Run {
  date: string;
  distance_km: number;
  duration_minutes: number;
  avg_pace_per_km: string;
  avg_hr: number;
  max_hr: number;
  elevation_gain_m: number;
  type: string;
  notes: string;
}

export interface WeeklySummary {
  this_week_km: number;
  last_week_km: number;
  '4_week_avg_km': number;
  this_week_runs: number;
  last_week_runs: number;
}

export interface PersonalBests {
  '5k_minutes': number;
  '10k_minutes': number;
  half_marathon_minutes: number;
  longest_run_km: number;
}

export interface TrainingZones {
  zone1_max_hr: number;
  zone2_max_hr: number;
  zone3_max_hr: number;
  zone4_max_hr: number;
  zone5_max_hr: number;
}

export interface UpcomingRace {
  name: string;
  date: string;
  distance_km: number;
  goal_time_minutes: number;
}

export interface MockGarminData {
  user_id: string;
  profile: Profile;
  recent_runs: Run[];
  weekly_summary: WeeklySummary;
  personal_bests: PersonalBests;
  injury_history: string[];
  training_zones: TrainingZones;
  upcoming_races: UpcomingRace[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
