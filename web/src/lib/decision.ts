import type { ModelConfig } from "@/api/client";

const DECISION_PROVIDERS = new Set(["typesafe", "typesafe-mock"]);

export function isDecisionProvider(provider: string | undefined | null): boolean {
  return !!provider && DECISION_PROVIDERS.has(provider);
}

export function isDecisionModel(cfg: Pick<ModelConfig, "provider">): boolean {
  return isDecisionProvider(cfg.provider);
}

export function splitModels(
  models: Record<string, ModelConfig>,
): { llm: string[]; decision: string[] } {
  const llm: string[] = [];
  const decision: string[] = [];
  for (const [id, cfg] of Object.entries(models)) {
    (isDecisionModel(cfg) ? decision : llm).push(id);
  }
  return { llm, decision };
}
