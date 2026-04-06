import { useState } from 'react';
import { Zap } from 'lucide-react';
import type { PersonalBests } from '@/types';

// Riegel formula: T2 = T1 × (D2 / D1)^1.06
// 1.06 = empirically derived fatigue constant — humans slow down
// non-linearly with distance due to glycogen depletion and fatigue.
function riegel(t1Min: number, d1Km: number, d2Km: number): number {
  return t1Min * Math.pow(d2Km / d1Km, 1.06);
}

function formatTime(minutes: number): string {
  const h = Math.floor(minutes / 60);
  const m = Math.floor(minutes % 60);
  const s = Math.round((minutes * 60) % 60);
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s`;
  return `${m}m ${String(s).padStart(2, '0')}s`;
}

function formatPace(minutes: number, distKm: number): string {
  const pace = minutes / distKm;
  const pm = Math.floor(pace);
  const ps = Math.round((pace - pm) * 60);
  return `${pm}:${String(ps).padStart(2, '0')} /km`;
}

const PRESETS = [
  { label: '5K',  km: 5 },
  { label: '10K', km: 10 },
  { label: 'HM',  km: 21.1 },
  { label: 'FM',  km: 42.2 },
];

interface PacePredictorProps {
  pbs: PersonalBests;
}

export function PacePredictor({ pbs }: PacePredictorProps) {
  const [targetKm, setTargetKm] = useState(21.1);

  const refs = [
    { label: '5K PB',  d: 5,    t: pbs['5k_minutes'] },
    { label: '10K PB', d: 10,   t: pbs['10k_minutes'] },
    { label: 'HM PB',  d: 21.1, t: pbs.half_marathon_minutes },
  ];

  // Use the reference closest to target (excluding exact match to avoid 1:1 trivial case)
  const candidates = refs.filter(r => r.d !== targetKm);
  const ref = candidates.reduce((best, r) =>
    Math.abs(r.d - targetKm) < Math.abs(best.d - targetKm) ? r : best
  , candidates[0] ?? refs[0]);

  const predicted = riegel(ref.t, ref.d, targetKm);

  return (
    <div className="p-5 bg-gradient-to-br from-card to-background border border-border/50 rounded-xl card-glow card-hover">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2.5 rounded-xl bg-yellow-500/20">
          <Zap className="w-5 h-5 text-yellow-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Race Time Predictor</h3>
          <p className="text-xs text-muted-foreground">Riegel formula · from your personal bests</p>
        </div>
      </div>

      {/* Preset buttons */}
      <div className="flex gap-2 mb-4">
        {PRESETS.map(p => (
          <button
            key={p.label}
            onClick={() => setTargetKm(p.km)}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              targetKm === p.km
                ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                : 'bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-white'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Custom distance slider */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-muted-foreground mb-1.5">
          <span>Target distance</span>
          <span className="text-white font-semibold">{targetKm.toFixed(1)} km</span>
        </div>
        <input
          type="range"
          min={1}
          max={42.2}
          step={0.1}
          value={targetKm}
          onChange={e => setTargetKm(parseFloat(e.target.value))}
          className="w-full accent-yellow-400 cursor-pointer"
        />
        <div className="flex justify-between text-xs text-muted-foreground mt-1">
          <span>1 km</span>
          <span>42.2 km</span>
        </div>
      </div>

      {/* Prediction result */}
      <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4 text-center">
        <div className="text-3xl font-bold text-yellow-400 tracking-tight">{formatTime(predicted)}</div>
        <div className="text-sm text-white/70 mt-1">{formatPace(predicted, targetKm)}</div>
        <div className="text-xs text-muted-foreground mt-2">
          Based on your {ref.label} of {formatTime(ref.t)}
        </div>
      </div>
    </div>
  );
}
