const API_BASE = import.meta.env.VITE_API_URL ?? '';

export const API_ENDPOINTS = {
  health: `${API_BASE}/health`,
  chat: `${API_BASE}/api/chat`,
} as const;
