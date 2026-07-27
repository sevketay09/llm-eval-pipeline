import { useEffect, useState } from "react";
import { Play, Square, Loader2 } from "lucide-react";
import { NavLink } from "react-router-dom";
import {
  ApiError,
  customDatasetsApi,
  modelsApi,
  evaluationsApi,
  type CustomDatasetSummary,
  type ModelListResponse,
  type EvalRunRequest,
  type EvalRunStatus,
} from "@/api/client";
import { useEvalProgress } from "@/hooks/useWebSocket";

function formatRunLaunchError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.errorCode || error.errorStage) {
      const context = [error.errorStage, error.errorCode].filter(Boolean).join(" / ");
      return `${error.message} (${context})`;
    }
    return error.message;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

export default function RunEvaluation() {
  const [models, setModels] = useState<ModelListResponse | null>(null);
  const [suites, setSuites] = useState<string[]>([]);
  const [suiteDetail, setSuiteDetail] = useState<Record<string, string[]>>({});
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedSuite, setSuite] = useState("smoke");
  const [selectedTests, setSelectedTests] = useState<string[]>([]);
  const [judgeModel, setJudgeModel] = useState("");
  const [parallel, setParallel] = useState(false);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState("");
  const [maxWorkers, setMaxWorkers] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [maxTokens, setMaxTokens] = useState(4096);
  const [activeRun, setActiveRun] = useState<EvalRunStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useCustomDataset, setUseCustomDataset] = useState(false);
  const [customDatasetId, setCustomDatasetId] = useState("");
  const [recentDatasets, setRecentDatasets] = useState<CustomDatasetSummary[]>([]);

  const { progress } = useEvalProgress(activeRun?.run_id ?? null);

  // Mock-provider models (e.g. demo-model, used by the offline `make demo` CLI flow)
  // are excluded here — they return canned, non-JSON responses and have no place
  // in a real evaluation run or as a judge. They remain selectable via the CLI/config
  // directly for the demo flow; this only filters the web UI's real-run picker.
  const evaluableModelIds = models
    ? Object.keys(models.models).filter((id) => models.models[id]!.provider !== "mock")
    : [];

  useEffect(() => {
    modelsApi.list().then((r) => {
      setModels(r);
      // Deliberately no default selection — auto-checking the first configured
      // model (often the offline "demo-model" mock, first in models.yaml) let
      // it silently ride along into real runs whenever a user forgot to
      // uncheck it before adding their actual models.
    });
    evaluationsApi.listSuites().then((r) => {
      setSuites(r.suites);
      setSuiteDetail(r.detail ?? {});
      if (r.suites.length > 0) {
        const first = r.suites[0]!;
        setSuite(first);
        setSelectedTests(r.detail?.[first] ?? []);
      }
    });
    customDatasetsApi.list(24).then(setRecentDatasets);
    // Reattach to a run already in progress on the backend — without this,
    // reloading the page (or a dropped connection, e.g. the machine went to
    // sleep mid-run) loses all visible progress even though the run itself
    // keeps going server-side; the websocket below only connects once
    // activeRun is set.
    evaluationsApi.listRuns(5).then((runs) => {
      const inProgress = runs.find((r) => r.status === "running");
      if (inProgress) {
        setActiveRun(inProgress);
      }
    });
  }, []);

  const handleStart = async () => {
    if (selectedModels.length === 0) return;
    setError(null);
    try {
      const req: EvalRunRequest = {
        models: selectedModels,
        suite: selectedSuite,
        parallel,
        temperature,
        top_p: topP.trim() ? Number(topP) : undefined,
        max_workers: parallel && maxWorkers.trim() ? Number(maxWorkers) : undefined,
        max_tokens: maxTokens,
        judge_model: judgeModel || undefined,
        tests:
          !useCustomDataset &&
          selectedTests.length > 0 &&
          selectedTests.length < (suiteDetail[selectedSuite]?.length ?? 0)
            ? selectedTests
            : undefined,
        custom_dataset_id: useCustomDataset && customDatasetId ? customDatasetId : undefined,
        output_path: outputPath.trim() ? outputPath.trim() : undefined,
      };
      const run = await evaluationsApi.run(req);
      setActiveRun(run);
    } catch (e: unknown) {
      setError(formatRunLaunchError(e));
    }
  };

  const handleCancel = async () => {
    if (!activeRun) return;
    try {
      const cancelledRun = await evaluationsApi.cancel(activeRun.run_id);
      setActiveRun(cancelledRun);
    } catch (e: unknown) {
      setError(formatRunLaunchError(e));
    }
  };

  const isRunning = activeRun?.status === "running" || progress?.status === "running";

  return (
    <div className="page-shell motion-shell max-w-5xl">
      <div className="motion-stagger-grid grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Models + Suite */}
        <section className="panel-surface panel-roomy space-y-5">
          <div>
            <label className="label">Models</label>
            <div className="option-list">
              {evaluableModelIds.map((id) => (
                  <label key={id} className="option-row cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(id)}
                      onChange={(e) =>
                        setSelectedModels((prev) =>
                          e.target.checked ? [...prev, id] : prev.filter((m) => m !== id),
                        )
                      }
                      className="control-check"
                    />
                    <span>{id}</span>
                  </label>
                ))}
            </div>
          </div>

          <div className="control-group">
            <label className="label">Test Suite</label>
            <select
              value={selectedSuite}
              onChange={(e) => {
                const suite = e.target.value;
                setSuite(suite);
                setSelectedTests(suiteDetail[suite] ?? []);
              }}
              className="control-surface"
              disabled={useCustomDataset}
            >
              {suites.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            {useCustomDataset && (
              <p className="micro-copy mt-2">
                The next run will use the custom dataset instead of the selected suite.
              </p>
            )}
          </div>

          {!useCustomDataset &&
            suiteDetail[selectedSuite] &&
            suiteDetail[selectedSuite].length > 0 && (
              <div className="control-group">
                <label className="label">Tests in Suite</label>
                <div className="option-list max-h-48 overflow-y-auto">
                  <label className="option-row cursor-pointer text-xs font-medium opacity-70">
                    <input
                      type="checkbox"
                      checked={selectedTests.length === (suiteDetail[selectedSuite]?.length ?? 0)}
                      onChange={(e) =>
                        setSelectedTests(
                          e.target.checked ? (suiteDetail[selectedSuite] ?? []) : [],
                        )
                      }
                      className="control-check"
                    />
                    <span>Select All</span>
                  </label>
                  {suiteDetail[selectedSuite].map((test) => (
                    <label key={test} className="option-row cursor-pointer">
                      <input
                        type="checkbox"
                        checked={selectedTests.includes(test)}
                        onChange={(e) =>
                          setSelectedTests((prev) =>
                            e.target.checked ? [...prev, test] : prev.filter((t) => t !== test),
                          )
                        }
                        className="control-check"
                      />
                      <span>{test}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}
        </section>

        {/* Parameters */}
        <section className="panel-surface panel-roomy space-y-5">
          <div className="control-group">
            <label className="label">Temperature: {temperature.toFixed(1)}</label>
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="control-range"
            />
          </div>

          <div className="control-group">
            <label className="label">Max Tokens</label>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value) || 4096)}
              min={256}
              max={32768}
              step={256}
              className="control-surface"
            />
          </div>

          <div className="control-group">
            <label className="label">Top P</label>
            <input
              type="number"
              value={topP}
              onChange={(e) => setTopP(e.target.value)}
              min={0}
              max={1}
              step={0.05}
              placeholder="Use model default"
              className="control-surface"
            />
            <p className="micro-copy mt-2">Leave blank to preserve the model configuration default.</p>
          </div>

          <div className="control-group">
            <label className="label">Judge Model</label>
            <select
              value={judgeModel}
              onChange={(e) => setJudgeModel(e.target.value)}
              className="control-surface"
            >
              <option value="">— Auto</option>
              {evaluableModelIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>

          <label className="toggle-card cursor-pointer">
            <input
              type="checkbox"
              checked={parallel}
              onChange={(e) => setParallel(e.target.checked)}
              className="control-check"
            />
            <span>Parallel execution</span>
          </label>

          {parallel && (
            <div className="control-group">
              <label className="label">Parallel Workers</label>
              <input
                type="number"
                value={maxWorkers}
                onChange={(e) => setMaxWorkers(e.target.value)}
                min={1}
                max={16}
                step={1}
                placeholder="Auto"
                className="control-surface"
              />
              <p className="micro-copy mt-2">
                Matches the CLI <code>--parallel-workers</code> override. Leave blank for automatic
                sizing.
              </p>
            </div>
          )}

          <div className="control-group">
            <label className="label">Output Path</label>
            <input
              type="text"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
              placeholder="Use API default report location"
              className="control-surface"
            />
            <p className="micro-copy mt-2">
              Matches the CLI <code>--output</code> override and writes JSON, Markdown, and HTML
              artifacts together.
            </p>
          </div>
        </section>
      </div>

      {/* Custom Dataset picker */}
      <section className="panel-surface panel-roomy motion-rise motion-delay-3 space-y-4">
        <p className="section-caption">Custom Dataset</p>
        <label className="toggle-card cursor-pointer">
          <input
            type="checkbox"
            checked={useCustomDataset}
            onChange={(e) => setUseCustomDataset(e.target.checked)}
            className="control-check"
          />
          <span>Use a custom dataset for this run</span>
        </label>
        {useCustomDataset && (
          <div className="space-y-3">
            <div className="control-group">
              <label className="label">Dataset</label>
              <select
                value={customDatasetId}
                onChange={(e) => setCustomDatasetId(e.target.value)}
                className="control-surface"
              >
                <option value="">— Select a dataset</option>
                {recentDatasets.map((ds) => (
                  <option key={ds.dataset_id} value={ds.dataset_id}>
                    {ds.title} · {ds.dataset_kind ?? "single_turn"} · {ds.sample_count} cases
                  </option>
                ))}
              </select>
            </div>
            <NavLink to="/datasets" className="ghost-button inline-flex items-center gap-2">
              Open Dataset Studio to create or manage datasets →
            </NavLink>
          </div>
        )}
      </section>

      <div className="button-row motion-rise motion-delay-4">
        <button
          onClick={handleStart}
          disabled={isRunning || selectedModels.length === 0}
          className="button-primary"
        >
          <Play size={16} />
          Start Evaluation
        </button>
        {isRunning && (
          <button onClick={handleCancel} className="button-danger">
            <Square size={16} />
            Cancel
          </button>
        )}
      </div>

      {error && <div className="alert-box alert-danger motion-rise motion-delay-5">{error}</div>}

      {progress && (
        <section className="panel-surface panel-quiet live-panel motion-delay-6 space-y-4">
          <div className="flex items-center justify-between gap-4">
            <div className="live-status flex items-center gap-2 body-copy">
              {progress.status === "running" && (
                <Loader2 size={16} className="accent-icon animate-spin" />
              )}
              <span className="font-medium">
                {progress.status === "running" ? "Running" : progress.status}
              </span>
            </div>
            <span className="micro-copy">{Math.round(progress.elapsed_seconds)}s elapsed</span>
          </div>

          <div className="progress-track live-meter">
            <div className="progress-fill" style={{ width: `${progress.progress * 100}%` }} />
          </div>

          <div className="flex items-center justify-between gap-4 text-sm">
            <span className="body-copy">{progress.message}</span>
            <span className="micro-copy">{(progress.progress * 100).toFixed(0)}%</span>
          </div>

          {progress.status === "running" && progress.progress === 0 && progress.elapsed_seconds < 60 && (
            <div className="rounded-[1rem] border border-[rgba(180,140,40,0.25)] bg-[rgba(180,140,40,0.08)] px-4 py-3 text-sm">
              <p className="micro-copy">
                <span className="body-copy font-medium">First run:</span>{" "}
                NLP data packages (NLTK) may be downloading. This can take 30–60 seconds on the first run — the progress bar may start later than expected.
              </p>
            </div>
          )}

          {progress.status === "failed" && (progress.error_code || progress.error_stage) && (
            <div className="rounded-[1rem] border border-[rgba(155,61,46,0.22)] bg-[rgba(155,61,46,0.08)] px-4 py-3 text-sm">
              <p className="body-copy font-medium">Structured failure context</p>
              <p className="micro-copy mt-2">
                Stage: <span className="body-copy">{progress.error_stage || "execution"}</span>
              </p>
              <p className="micro-copy mt-1">
                Code: <span className="body-copy">{progress.error_code || "unexpected_failure"}</span>
              </p>
            </div>
          )}

          {progress.current_model && (
            <p className="micro-copy">
              Model: <span className="body-copy">{progress.current_model}</span>
              {progress.current_test && (
                <>
                  {" "}
                  · Test: <span className="body-copy">{progress.current_test}</span>
                </>
              )}
            </p>
          )}
        </section>
      )}
    </div>
  );
}
