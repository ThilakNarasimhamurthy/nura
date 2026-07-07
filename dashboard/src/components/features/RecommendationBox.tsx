interface Props {
  recommendation: string;
}

export function RecommendationBox({ recommendation }: Props) {
  return (
    <div
      className="rounded-2xl p-5"
      style={{
        backgroundColor: "#eef2ff",
        borderLeft: "3px solid #6366f1",
      }}
    >
      <p className="text-xs font-semibold uppercase tracking-wide text-indigo-500">
        Nura recommends
      </p>
      <p className="mt-2 text-sm leading-relaxed text-gray-700">
        {recommendation}
      </p>
    </div>
  );
}
