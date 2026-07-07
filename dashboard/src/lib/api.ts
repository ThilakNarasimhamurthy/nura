import type { ModelMetrics } from "@/types";

const DEMO_DATA: ModelMetrics = {
  resolution_rate: 0.33,
  outcomes_count: 847,
  next_retrain_at: 1347,
  improvement_pct: 6.5,
  before_score: 31.0,
  after_score: 33.0,
  brain_recommendation:
    "Model improved 6.5% in 20 steps. Run 100+ steps on GPU for larger gains.",
};

export async function fetchMetrics(): Promise<ModelMetrics> {
  try {
    const res = await fetch("http://localhost:8000/v1/metrics", {
      next: { revalidate: 30 },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as ModelMetrics;
  } catch {
    return DEMO_DATA;
  }
}
