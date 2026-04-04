import { Activity } from 'lucide-react';
import type { TrainingZones } from '@/types';

interface ZoneCardProps {
  zones: TrainingZones;
}

export function ZoneCard({ zones }: ZoneCardProps) {
  const zoneData = [
    { name: 'Z1', max: zones.zone1_max_hr, label: 'Recovery', color: '#6b7280' },
    { name: 'Z2', max: zones.zone2_max_hr, label: 'Aerobic', color: '#10b981' },
    { name: 'Z3', max: zones.zone3_max_hr, label: 'Tempo', color: '#f59e0b' },
    { name: 'Z4', max: zones.zone4_max_hr, label: 'Threshold', color: '#f97316' },
    { name: 'Z5', max: zones.zone5_max_hr, label: 'Anaerobic', color: '#ef4444' },
  ];

  const maxHR = zones.zone5_max_hr;

  return (
    <div className="p-5 bg-gradient-to-br from-card to-background border border-border/50 rounded-xl card-glow card-hover">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2.5 rounded-xl bg-purple-500/20">
          <Activity className="w-5 h-5 text-purple-500" />
        </div>
        <div>
          <h3 className="font-semibold text-white">Training Zones</h3>
          <p className="text-xs text-muted-foreground">Heart rate zones</p>
        </div>
      </div>
      
      <div className="space-y-2">
        {zoneData.map((zone, index) => {
          const prevMax = index === 0 ? 0 : zoneData[index - 1].max;
          const progress = (zone.max / maxHR) * 100;
          
          return (
            <div key={zone.name} className="flex items-center gap-3">
              <span className="w-6 text-xs font-medium text-slate-400">{zone.name}</span>
              <div className="flex-1 h-2 bg-black/30 rounded-full overflow-hidden">
                <div 
                  className="h-full rounded-full transition-all duration-500"
                  style={{ 
                    width: `${progress}%`,
                    backgroundColor: zone.color,
                  }}
                />
              </div>
              <span className="text-xs text-muted-foreground w-24 text-right">
                {prevMax + 1}-{zone.max} bpm
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
