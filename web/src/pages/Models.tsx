import { useEffect, useState } from "react";
import { Plus, Trash2, Upload, Download, Zap } from "lucide-react";
import { modelsApi, type ModelConfig, type ModelListResponse, type ModelTestResult } from "@/api/client";
import { Badge, useToast } from "@/components";
import { isDecisionModel } from "@/lib/decision";

const PROVIDERS = ["openai", "anthropic", "ollama", "lmstudio", "vllm", "typesafe"];

export default function Models() {
  const [data, setData] = useState<ModelListResponse | null>(null);
  const [tab, setTab] = useState<"list" | "add" | "import">("list");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);
  const toast = useToast();

  const reload = () => modelsApi.list().then(setData);

  const handleTest = async (id: string) => {
    setTestingId(id);
    try {
      const result: ModelTestResult = await modelsApi.test(id);
      if (result.ok) {
        toast.success(`'${id}' responded in ${result.latency_ms.toFixed(0)} ms`);
      } else {
        toast.error(result.error || `'${id}' test failed`);
      }
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Test failed");
    } finally {
      setTestingId(null);
    }
  };

  useEffect(() => {
    reload();
  }, []);

  const [newId, setNewId] = useState("");
  const [newConfig, setNewConfig] = useState<Partial<ModelConfig>>({
    provider: "openai",
    model_name: "",
    api_key: "",
    base_url: "",
    max_tokens: 4096,
    temperature: 0,
    supports_function_calling: true,
    supports_streaming: true,
  });

  const [importText, setImportText] = useState("");
  const [importFormat, setImportFormat] = useState<"yaml" | "json">("yaml");

  const handleCreate = async () => {
    if (!newId.trim() || !newConfig.model_name) {
      setError("Model ID and name are required");
      return;
    }
    setError(null);
    try {
      await modelsApi.create(newId.trim(), newConfig);
      setSuccess(`Model '${newId}' created`);
      setNewId("");
      setNewConfig({ ...newConfig, model_name: "", api_key: "", base_url: "" });
      await reload();
      setTab("list");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed");
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm(`Delete model '${id}'?`)) return;
    try {
      await modelsApi.delete(id);
      await reload();
      setSuccess(`Deleted '${id}'`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed");
    }
  };

  const handleImport = async () => {
    setError(null);
    try {
      const parsed = importFormat === "json" ? JSON.parse(importText) : null;
      if (importFormat === "yaml") {
        setError("YAML import: paste as JSON for now (YAML support coming soon)");
        return;
      }
      if (!parsed || typeof parsed !== "object") {
        setError("Invalid format — expected { model_id: { config } }");
        return;
      }
      await modelsApi.import(parsed, false);
      setSuccess("Import successful");
      setImportText("");
      await reload();
      setTab("list");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Parse error");
    }
  };

  const handleExport = async () => {
    const yaml = await modelsApi.exportYaml();
    const blob = new Blob([yaml], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `models_export_${Date.now()}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleProviderChange = (provider: string) => {
    const wasDecision = isDecisionModel({ provider: newConfig.provider ?? "" });
    const nowDecision = isDecisionModel({ provider });
    setNewConfig({
      ...newConfig,
      provider,
      model_name: nowDecision && !wasDecision ? "jev-latest" : newConfig.model_name,
    });
  };

  const isNewDecision = isDecisionModel({ provider: newConfig.provider ?? "" });

  return (
    <div className="page-shell motion-shell max-w-5xl">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <header className="page-header motion-hero">
          <p className="page-kicker">Registry</p>
          <h1 className="page-title">Model Management</h1>
          <p className="page-subtitle">
            Maintain the model ledger with a cleaner surface for add, import and export workflows.
          </p>
        </header>
        <button onClick={handleExport} className="button-secondary motion-rise motion-delay-1 w-fit">
          <Download size={14} />
          Export YAML
        </button>
      </div>

      {error && <div className="alert-box alert-danger motion-rise motion-delay-2">{error}</div>}
      {success && <div className="alert-box alert-success motion-rise motion-delay-2">{success}</div>}

      <div className="tab-strip motion-rise motion-delay-2 w-fit">
        {(["list", "add", "import"] as const).map((t) => (
          <button
            key={t}
            onClick={() => {
              setTab(t);
              setError(null);
              setSuccess(null);
            }}
            className={`tab-button ${tab === t ? "tab-button-active" : ""}`.trim()}
          >
            {t === "list" ? "Models" : t === "add" ? "Add" : "Import"}
          </button>
        ))}
      </div>

      {tab === "list" && data && (
        <div className="motion-stagger-stack space-y-4">
          {Object.entries(data.models).map(([id, cfg]) => {
            const decision = isDecisionModel(cfg);
            return (
              <div key={id} className="panel-surface panel-quiet flex items-center justify-between gap-4">
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className="font-medium">{id}</span>
                    <span className="provider-chip">{cfg.provider}</span>
                    {decision && <Badge tone="violet">Decision</Badge>}
                  </div>
                  <p className="body-copy font-mono text-sm">{cfg.model_name}</p>
                  {decision ? (
                    <div className="flex flex-wrap gap-4 text-xs muted-copy">
                      <span>timeout: {cfg.timeout_s ?? 10}s</span>
                      {cfg.cost_per_mtok_input != null && <span>${cfg.cost_per_mtok_input}/MTok in</span>}
                    </div>
                  ) : (
                    <div className="flex flex-wrap gap-4 text-xs muted-copy">
                      <span>temp: {cfg.temperature}</span>
                      <span>max_tokens: {cfg.max_tokens}</span>
                      <span>fn_call: {cfg.supports_function_calling ? "✓" : "✗"}</span>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleTest(id)}
                    disabled={testingId === id}
                    className="button-secondary"
                  >
                    <Zap size={14} />
                    {testingId === id ? "Testing…" : "Test connection"}
                  </button>
                  <button onClick={() => handleDelete(id)} className="button-danger">
                    <Trash2 size={16} />
                    Remove
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {tab === "add" && (
        <div className="panel-surface panel-roomy motion-rise motion-delay-3 space-y-5">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="control-group">
              <label className="label">Model ID</label>
              <input
                value={newId}
                onChange={(e) => setNewId(e.target.value)}
                placeholder="my-gpt4o"
                className="control-surface"
              />
            </div>
            <div className="control-group">
              <label className="label">Provider</label>
              <select
                value={newConfig.provider}
                onChange={(e) => handleProviderChange(e.target.value)}
                className="control-surface"
              >
                {PROVIDERS.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label className="label">Model Name</label>
              <input
                value={newConfig.model_name}
                onChange={(e) => setNewConfig({ ...newConfig, model_name: e.target.value })}
                placeholder={isNewDecision ? "jev-latest" : "gpt-4o"}
                className="control-surface"
              />
            </div>
            <div className="control-group">
              <label className="label">API Key</label>
              <input
                value={newConfig.api_key}
                onChange={(e) => setNewConfig({ ...newConfig, api_key: e.target.value })}
                placeholder={isNewDecision ? "${TYPESAFE_API_KEY}" : "${OPENAI_API_KEY}"}
                className="control-surface"
              />
            </div>
            {!isNewDecision && (
              <div className="control-group">
                <label className="label">Base URL (optional)</label>
                <input
                  value={newConfig.base_url}
                  onChange={(e) => setNewConfig({ ...newConfig, base_url: e.target.value })}
                  placeholder="http://localhost:11434/v1"
                  className="control-surface"
                />
              </div>
            )}
            {isNewDecision ? (
              <div className="control-group">
                <label className="label">Timeout (seconds)</label>
                <input
                  type="number"
                  value={newConfig.timeout_s ?? 10}
                  onChange={(e) => setNewConfig({ ...newConfig, timeout_s: parseFloat(e.target.value) })}
                  min={1}
                  max={120}
                  step={1}
                  className="control-surface"
                />
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                <div className="control-group">
                  <label className="label">Temperature</label>
                  <input
                    type="number"
                    value={newConfig.temperature}
                    onChange={(e) =>
                      setNewConfig({ ...newConfig, temperature: parseFloat(e.target.value) })
                    }
                    min={0}
                    max={2}
                    step={0.1}
                    className="control-surface"
                  />
                </div>
                <div className="control-group">
                  <label className="label">Max Tokens</label>
                  <input
                    type="number"
                    value={newConfig.max_tokens}
                    onChange={(e) =>
                      setNewConfig({ ...newConfig, max_tokens: parseInt(e.target.value) })
                    }
                    min={256}
                    max={32768}
                    step={256}
                    className="control-surface"
                  />
                </div>
              </div>
            )}
          </div>

          {!isNewDecision && (
            <div className="button-row">
              <label className="toggle-card cursor-pointer">
                <input
                  type="checkbox"
                  checked={newConfig.supports_function_calling}
                  onChange={(e) =>
                    setNewConfig({ ...newConfig, supports_function_calling: e.target.checked })
                  }
                  className="control-check"
                />
                <span>Function Calling</span>
              </label>
              <label className="toggle-card cursor-pointer">
                <input
                  type="checkbox"
                  checked={newConfig.supports_streaming}
                  onChange={(e) => setNewConfig({ ...newConfig, supports_streaming: e.target.checked })}
                  className="control-check"
                />
                <span>Streaming</span>
              </label>
            </div>
          )}

          <button onClick={handleCreate} className="button-primary w-fit">
            <Plus size={16} />
            Add Model
          </button>
        </div>
      )}

      {tab === "import" && (
        <div className="panel-surface panel-roomy motion-rise motion-delay-3 space-y-5">
          <div className="tab-strip w-fit">
            {(["json", "yaml"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setImportFormat(f)}
                className={`tab-button ${importFormat === f ? "tab-button-active" : ""}`.trim()}
              >
                {f.toUpperCase()}
              </button>
            ))}
          </div>
          <textarea
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
            placeholder={`{\n  "my-model": {\n    "provider": "openai",\n    "model_name": "gpt-4o",\n    "api_key": "\${OPENAI_API_KEY}"\n  }\n}`}
            rows={10}
            className="control-surface resize-none font-mono text-sm"
          />
          <button onClick={handleImport} className="button-primary w-fit">
            <Upload size={16} />
            Import Models
          </button>
        </div>
      )}
    </div>
  );
}
