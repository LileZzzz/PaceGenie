import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import type { Run } from "@/types"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export function calculateWeeklyData(runs: Run[], weeks: number = 4): { week: string; distance: number }[] {
  const now = new Date()
  return Array.from({ length: weeks }, (_, i) => {
    const weekStart = new Date(now)
    weekStart.setDate(now.getDate() - (weeks - i) * 7)
    const weekEnd = new Date(weekStart)
    weekEnd.setDate(weekStart.getDate() + 7)
    const distance = runs
      .filter(r => {
        const d = new Date(r.date)
        return d >= weekStart && d < weekEnd
      })
      .reduce((sum, r) => sum + r.distance_km, 0)
    return { week: `W${i + 1}`, distance: Math.round(distance * 10) / 10 }
  })
}
