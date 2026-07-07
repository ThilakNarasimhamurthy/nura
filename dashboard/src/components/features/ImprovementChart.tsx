"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Props {
  beforeScore: number;
  afterScore: number;
}

export function ImprovementChart({ beforeScore, afterScore }: Props) {
  const data = [
    { label: "Baseline", score: beforeScore },
    { label: "Run 1", score: afterScore },
  ];

  return (
    <div className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
      <p className="text-sm font-medium text-gray-500">Model improvement</p>
      <p className="mt-0.5 text-xs text-gray-400">
        Resolution rate across training runs
      </p>

      <div className="mt-4 h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 4, right: 8, left: -16, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 12, fill: "#9ca3af" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={(v: number) => `${v}%`}
              tick={{ fontSize: 12, fill: "#9ca3af" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              formatter={(value: number) => [`${value}%`, "Resolution rate"]}
              contentStyle={{
                borderRadius: 8,
                border: "1px solid #e5e7eb",
                fontSize: 12,
              }}
            />
            <Line
              type="monotone"
              dataKey="score"
              stroke="#6366f1"
              strokeWidth={2.5}
              dot={{ fill: "#6366f1", r: 5, strokeWidth: 0 }}
              activeDot={{ r: 7, strokeWidth: 0 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex items-center gap-2 text-xs text-gray-400">
        <span className="inline-block h-0.5 w-6 rounded bg-indigo-500" />
        Before → after GRPO fine-tuning
      </div>
    </div>
  );
}
