import type { ModelMetrics } from "@/types";

interface Props {
  metrics: ModelMetrics;
}

interface CardProps {
  label: string;
  value: string;
  subtitle: string;
  highlight?: boolean;
}

function MetricCard({ label, value, subtitle, highlight }: CardProps) {
  return (
    <div className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
      <p className="text-sm font-medium text-gray-500">{label}</p>
      <p
        className={`mt-1 text-3xl font-bold ${
          highlight ? "text-indigo-500" : "text-gray-900"
        }`}
      >
        {value}
      </p>
      <p className="mt-1 text-xs text-gray-400">{subtitle}</p>
    </div>
  );
}

export function MetricsRow({ metrics }: Props) {
  const ratePercent = `${(metrics.resolution_rate * 100).toFixed(1)}% ↑`;
  const toRetrain = Math.max(0, metrics.next_retrain_at - metrics.outcomes_count);

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <MetricCard
        label="Resolution rate"
        value={ratePercent}
        subtitle={`+${metrics.improvement_pct}% since first run`}
        highlight
      />
      <MetricCard
        label="Interactions learned"
        value={metrics.outcomes_count.toLocaleString()}
        subtitle="since last retrain"
      />
      <MetricCard
        label="Next retrain in"
        value={toRetrain.toLocaleString()}
        subtitle="outcomes needed"
      />
    </div>
  );
}
