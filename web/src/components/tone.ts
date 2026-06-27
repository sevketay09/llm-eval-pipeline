// Shared score → tone mapping so every page colors scores identically.
export type Tone = "good" | "mid" | "low";

export function scoreTone(score: number): Tone {
  if (score >= 0.7) return "good";
  if (score >= 0.4) return "mid";
  return "low";
}
