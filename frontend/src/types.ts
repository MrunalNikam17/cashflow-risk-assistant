export type RiskStatus = 'SAFE' | 'RISKY' | 'NOT RECOMMENDED';

export interface SummaryResponse {
  current_balance: number;
  inflow_30d: number;
  outflow_30d: number;
  net_30d: number;
  burn_rate: number;
  runway_days: number;
  category_breakdown: Record<string, number>;
  monthly_series: Array<{ month: string; inflow: number; outflow: number; net: number }>;
  risk_status: RiskStatus | string;
}

export interface ForecastResponse {
  horizon_days: number;
  current_balance: number;
  lowest_balance: number;
  lowest_balance_date: string;
  final_projected_balance: number;
  probability_balance_below_buffer: number;
  history: Array<Record<string, any>>;
  forecast: Array<Record<string, any>>;
  explainable_factors: Record<string, any>;
  model_accuracy: Record<string, any>;
  summary_text: string;
}

export interface AskResponse {
  verdict: string;
  numbers: Record<string, number>;
  explanation: string;
  explanation_source: 'llm' | 'template';
  guardrail_passed: boolean;
}
