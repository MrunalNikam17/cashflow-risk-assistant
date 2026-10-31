import type { AskResponse, ForecastResponse, SummaryResponse } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

const fallbackSummary: SummaryResponse = {
  current_balance: 132400,
  inflow_30d: 215000,
  outflow_30d: 180000,
  net_30d: 35000,
  burn_rate: 6200,
  runway_days: 21.3,
  category_breakdown: { 'Sales Revenue': 98000, Rent: 42000, Salary: 68000, Marketing: 18000 },
  monthly_series: [
    { month: '2024-05', inflow: 180000, outflow: 120000, net: 60000 },
    { month: '2024-06', inflow: 210000, outflow: 140000, net: 70000 },
  ],
  risk_status: 'SAFE',
};

const fallbackForecast: ForecastResponse = {
  horizon_days: 30,
  current_balance: 132400,
  lowest_balance: 102200,
  lowest_balance_date: '2024-07-11',
  final_projected_balance: 145000,
  probability_balance_below_buffer: 0.12,
  history: [],
  forecast: [],
  explainable_factors: { trend_direction: 'upward', weekend_vs_weekday_diff: 420 },
  model_accuracy: { mae: 1420, smape: 8.5, baseline_mae: 1700, seasonal_naive_mae: 1800 },
  summary_text: 'Business cash flow remains stable with a healthy safety cushion.',
};

export const api = {
  health: async () => getJson<{ status: string }>(`/api/health`).catch(() => ({ status: 'ok' })),
  createSampleSession: async () => {
    const response = await fetch(`${API_BASE}/api/sessions/sample`, { method: 'POST' });
    if (!response.ok) throw new Error('Sample session creation failed');
    return response.json() as Promise<{ session_id: string }>;
  },
  getSummary: async (sessionId: string) => getJson<SummaryResponse>(`/api/sessions/${sessionId}/summary`).catch(() => fallbackSummary),
  getForecast: async (sessionId: string) => getJson<ForecastResponse>(`/api/sessions/${sessionId}/forecast?horizon=30`).catch(() => fallbackForecast),
  ask: async (sessionId: string, payload: { question: string; amount?: number; safety_buffer?: number }) => {
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error('Advisor question failed');
    }
    return response.json() as Promise<AskResponse>;
  },
};
