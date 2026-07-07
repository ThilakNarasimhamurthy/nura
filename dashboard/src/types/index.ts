export interface ModelMetrics {
  resolution_rate: number;
  outcomes_count: number;
  next_retrain_at: number;
  improvement_pct: number;
  before_score: number;
  after_score: number;
  brain_recommendation: string;
}
