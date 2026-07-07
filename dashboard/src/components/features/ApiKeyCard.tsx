"use client";

import { useState } from "react";

const DISPLAY_KEY = "nura_sk_••••••••demo";
const REAL_KEY = "nura_sk_live_demo_12345";

export function ApiKeyCard() {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    await navigator.clipboard.writeText(REAL_KEY);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
      <p className="text-sm font-medium text-gray-900">Your API key</p>
      <p className="mt-0.5 text-xs text-gray-400">
        Use this in your app to call your model
      </p>

      <div className="mt-3 flex items-center gap-2">
        <code className="flex-1 rounded-lg bg-gray-50 px-3 py-2 font-mono text-sm text-gray-600">
          {DISPLAY_KEY}
        </code>
        <button
          onClick={handleCopy}
          className="shrink-0 rounded-lg bg-indigo-500 px-3 py-2 text-xs font-medium text-white transition hover:bg-indigo-600 active:scale-95"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
    </div>
  );
}
