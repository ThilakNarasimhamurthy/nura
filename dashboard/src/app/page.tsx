import { fetchMetrics } from "@/lib/api";
import { ApiKeyCard } from "@/components/features/ApiKeyCard";
import { ImprovementChart } from "@/components/features/ImprovementChart";
import { MetricsRow } from "@/components/features/MetricsRow";
import { RecommendationBox } from "@/components/features/RecommendationBox";

export default async function DashboardPage() {
  const metrics = await fetchMetrics();

  return (
    <main className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b border-gray-100 px-6 py-4">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Nura</h1>
            <p className="text-sm text-gray-400">
              Your data. Your model. Gets smarter every day.
            </p>
          </div>
          <span className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-600">
            Live
          </span>
        </div>
      </header>

      {/* Main content */}
      <div className="mx-auto max-w-5xl space-y-6 px-6 py-8">
        {/* Metrics row */}
        <MetricsRow metrics={metrics} />

        {/* Chart + Recommendation side by side on wide screens */}
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <ImprovementChart
            beforeScore={metrics.before_score}
            afterScore={metrics.after_score}
          />
          <div className="space-y-6">
            <RecommendationBox recommendation={metrics.brain_recommendation} />
            <ApiKeyCard />
          </div>
        </div>
      </div>
    </main>
  );
}
