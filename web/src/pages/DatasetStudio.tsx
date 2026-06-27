import { useEffect, useState } from "react";
import { Sparkles, Upload, Loader2, Play } from "lucide-react";
import { NavLink } from "react-router-dom";
import {
  customDatasetsApi,
  modelsApi,
  type CustomDatasetDetail,
  type CustomDatasetSummary,
  type ModelListResponse,
  type WorkspaceSourceFile,
} from "@/api/client";

type DatasetStageStatus = "completed" | "active" | "pending";

type DatasetStage = {
  key: string;
  label: string;
  detail: string;
  status: DatasetStageStatus;
};

function buildDatasetStages({
  generatorModel,
  projectDescription,
  datasetGenerationMode,
  datasetSourceMaterial,
  datasetSourcePaths,
  datasetBusy,
  generatedDataset,
}: {
  generatorModel: string;
  projectDescription: string;
  datasetGenerationMode: string;
  datasetSourceMaterial: string;
  datasetSourcePaths: string;
  datasetBusy: boolean;
  generatedDataset: CustomDatasetDetail | null;
}): DatasetStage[] {
  const briefReady = Boolean(generatorModel.trim()) && projectDescription.trim().length >= 40;
  const groundingReady =
    datasetGenerationMode === "generate_from_scratch" ||
    datasetSourceMaterial.trim().length >= 40 ||
    datasetSourcePaths
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean).length > 0;
  const reviewStarted = Boolean(generatedDataset && generatedDataset.review_status !== "draft");
  const finalized = Boolean(generatedDataset?.finalized_path);
  const promoted = Boolean(generatedDataset?.regression_dataset_path);

  return [
    {
      key: "brief",
      label: "Brief",
      detail: briefReady
        ? "Generator model and product brief are ready."
        : "Pick a generator model and write a detailed project brief.",
      status: briefReady ? "completed" : "active",
    },
    {
      key: "grounding",
      label: "Grounding",
      detail:
        datasetGenerationMode === "generate_from_scratch"
          ? "Scratch mode skips external grounding."
          : groundingReady
            ? "Source docs or contexts are attached for grounded generation."
            : "Add source docs, contexts, or workspace file paths.",
      status: groundingReady ? "completed" : briefReady ? "active" : "pending",
    },
    {
      key: "generate",
      label: "Generate",
      detail: generatedDataset
        ? `${generatedDataset.sample_count} cases prepared for preview.`
        : datasetBusy
          ? "Dataset generation is in progress."
          : "Create or import a dataset to unlock preview and review.",
      status: generatedDataset ? "completed" : datasetBusy ? "active" : groundingReady ? "active" : "pending",
    },
    {
      key: "review",
      label: "Review",
      detail: generatedDataset
        ? reviewStarted
          ? `Status: ${generatedDataset.review_status}${generatedDataset.review_role ? ` · ${generatedDataset.review_role.toUpperCase()}` : ""}`
          : "Approve or reject the generated dataset after review."
        : "Generation preview must exist before review starts.",
      status: reviewStarted ? "completed" : generatedDataset ? "active" : "pending",
    },
    {
      key: "finalize",
      label: "Finalize",
      detail: finalized
        ? `${generatedDataset?.finalized_case_count ?? 0} finalized cases captured in snapshot.`
        : reviewStarted
          ? "Approved datasets will materialize a finalized snapshot."
          : "Finalize unlocks after review approval.",
      status: finalized ? "completed" : reviewStarted ? "active" : "pending",
    },
    {
      key: "promote",
      label: "Promote",
      detail: promoted
        ? "Regression artifact exported and ready for reuse."
        : finalized
          ? "Promote the finalized artifact into regression storage."
          : "Regression promotion waits for a finalized snapshot.",
      status: promoted ? "completed" : finalized ? "active" : "pending",
    },
  ];
}

const reviewRoleOptions = ["qa", "sme", "pm"] as const;

export default function DatasetStudio() {
  const [models, setModels] = useState<ModelListResponse | null>(null);
  const [generatorModel, setGeneratorModel] = useState("");
  const [datasetTitle, setDatasetTitle] = useState("");
  const [datasetKind, setDatasetKind] = useState("single_turn");
  const [datasetGenerationMode, setDatasetGenerationMode] = useState("generate_from_scratch");
  const [datasetSourceLabel, setDatasetSourceLabel] = useState("");
  const [datasetSourceMaterial, setDatasetSourceMaterial] = useState("");
  const [datasetSourcePaths, setDatasetSourcePaths] = useState("");
  const [workspaceSourceFiles, setWorkspaceSourceFiles] = useState<WorkspaceSourceFile[]>([]);
  const [workspaceSourceLoading, setWorkspaceSourceLoading] = useState(false);
  const [projectDescription, setProjectDescription] = useState("");
  const [focusAreas, setFocusAreas] = useState("");
  const [sampleCount, setSampleCount] = useState(12);
  const [datasetBusy, setDatasetBusy] = useState(false);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [generatedDataset, setGeneratedDataset] = useState<CustomDatasetDetail | null>(null);
  const [recentDatasets, setRecentDatasets] = useState<CustomDatasetSummary[]>([]);
  const [recentDatasetsLoading, setRecentDatasetsLoading] = useState(false);
  const [datasetLibraryQuery, setDatasetLibraryQuery] = useState("");
  const [selectedReviewRole, setSelectedReviewRole] = useState<(typeof reviewRoleOptions)[number]>("qa");
  const [selectedReusableMetricCandidate, setSelectedReusableMetricCandidate] = useState(false);
  const [editingCaseId, setEditingCaseId] = useState<string | null>(null);
  const [casePromptDraft, setCasePromptDraft] = useState("");
  const [caseEditDraft, setCaseEditDraft] = useState("");

  const selectedSourcePathSet = new Set(
    datasetSourcePaths
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean),
  );
  const selectedDatasetId = generatedDataset?.dataset_id ?? null;

  const normalizedDatasetLibraryQuery = datasetLibraryQuery.trim().toLocaleLowerCase("tr-TR");
  const visibleRecentDatasets = recentDatasets.filter((dataset) => {
    if (!normalizedDatasetLibraryQuery) return true;
    const haystack = [
      dataset.title,
      dataset.source_label,
      dataset.dataset_kind,
      dataset.generator_model,
      dataset.generation_mode,
      dataset.source_type,
      ...(dataset.dataset_tags ?? []),
    ]
      .filter(Boolean)
      .join(" ")
      .toLocaleLowerCase("tr-TR");
    return haystack.includes(normalizedDatasetLibraryQuery);
  });

  const datasetStages = buildDatasetStages({
    generatorModel,
    projectDescription,
    datasetGenerationMode,
    datasetSourceMaterial,
    datasetSourcePaths,
    datasetBusy,
    generatedDataset,
  });
  const completedDatasetStages = datasetStages.filter((stage) => stage.status === "completed").length;

  useEffect(() => {
    modelsApi.list().then((r) => {
      setModels(r);
      if (r.total > 0) {
        setGeneratorModel(Object.keys(r.models)[0]!);
      }
    });
    setRecentDatasetsLoading(true);
    customDatasetsApi
      .list(16)
      .then(setRecentDatasets)
      .finally(() => setRecentDatasetsLoading(false));
  }, []);

  const handleGenerateDataset = async () => {
    if (!generatorModel) {
      setDatasetError("Select a generator model first.");
      return;
    }
    if (projectDescription.trim().length < 40) {
      setDatasetError("Project brief should be detailed enough to generate a useful eval set.");
      return;
    }
    if (datasetGenerationMode !== "generate_from_scratch") {
      const normalizedPaths = datasetSourcePaths
        .split("\n")
        .map((item) => item.trim())
        .filter(Boolean);
      if (datasetSourceMaterial.trim().length < 40 && normalizedPaths.length === 0) {
        setDatasetError("Context or docs mode needs source material or workspace file paths.");
        return;
      }
    }

    const normalizedSourcePaths = datasetSourcePaths
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean);

    setDatasetBusy(true);
    setDatasetError(null);
    try {
      const dataset = await customDatasetsApi.generate({
        title: datasetTitle.trim() || undefined,
        project_description: projectDescription.trim(),
        sample_count: sampleCount,
        generator_model: generatorModel,
        focus_areas: focusAreas.trim() || undefined,
        dataset_kind: datasetKind,
        generation_mode: datasetGenerationMode,
        source_label: datasetSourceLabel.trim() || undefined,
        source_material: datasetSourceMaterial.trim() || undefined,
        source_paths: normalizedSourcePaths,
      });
      setGeneratedDataset(dataset);
      setSelectedReusableMetricCandidate(dataset.reusable_metric_candidate ?? false);
      setEditingCaseId(null);
      setCaseEditDraft("");
      setRecentDatasets((prev) =>
        [dataset, ...prev.filter((item) => item.dataset_id !== dataset.dataset_id)].slice(0, 16),
      );
    } catch (e: unknown) {
      setDatasetError(e instanceof Error ? e.message : "Dataset generation failed");
    } finally {
      setDatasetBusy(false);
    }
  };

  const handleImportDataset = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setDatasetBusy(true);
    setDatasetError(null);
    try {
      const dataset = await customDatasetsApi.importJson({
        dataset_json: await file.text(),
        title: datasetTitle.trim() || undefined,
        project_description: projectDescription.trim() || "Imported dataset",
        focus_areas: focusAreas.trim() || undefined,
        source_label: file.name,
      });
      setGeneratedDataset(dataset);
      setSelectedReusableMetricCandidate(dataset.reusable_metric_candidate ?? false);
      setEditingCaseId(null);
      setCaseEditDraft("");
      setRecentDatasets((prev) =>
        [dataset, ...prev.filter((item) => item.dataset_id !== dataset.dataset_id)].slice(0, 16),
      );
    } catch (e: unknown) {
      setDatasetError(e instanceof Error ? e.message : "Dataset import failed");
    } finally {
      event.target.value = "";
      setDatasetBusy(false);
    }
  };

  const handleLoadWorkspaceSourceFiles = async () => {
    setWorkspaceSourceLoading(true);
    try {
      const files = await customDatasetsApi.listWorkspaceFiles(24);
      setWorkspaceSourceFiles(files);
    } catch (e: unknown) {
      setDatasetError(e instanceof Error ? e.message : "Workspace file scan failed");
    } finally {
      setWorkspaceSourceLoading(false);
    }
  };

  const handleAddWorkspaceSourcePath = (path: string) => {
    setDatasetSourcePaths((prev) => {
      const existing = prev
        .split("\n")
        .map((item) => item.trim())
        .filter(Boolean);
      if (existing.includes(path)) return prev;
      return [...existing, path].join("\n");
    });
  };

  const handleLoadRecentDataset = async (datasetId: string) => {
    setDatasetBusy(true);
    setDatasetError(null);
    try {
      const dataset = await customDatasetsApi.get(datasetId);
      setGeneratedDataset(dataset);
      setSelectedReusableMetricCandidate(dataset.reusable_metric_candidate ?? false);
      setEditingCaseId(null);
      setCaseEditDraft("");
    } catch (e: unknown) {
      setDatasetError(e instanceof Error ? e.message : "Dataset load failed");
    } finally {
      setDatasetBusy(false);
    }
  };

  const handleDatasetReviewStatus = async (datasetId: string, reviewStatus: string) => {
    setDatasetBusy(true);
    setDatasetError(null);
    try {
      const dataset = await customDatasetsApi.updateReviewStatus(datasetId, {
        review_status: reviewStatus,
        reviewer_role: selectedReviewRole,
        reusable_metric_candidate: selectedReusableMetricCandidate,
      });
      setGeneratedDataset((prev) => (prev?.dataset_id === dataset.dataset_id ? dataset : prev));
      setSelectedReusableMetricCandidate(dataset.reusable_metric_candidate ?? false);
      setRecentDatasets((prev) =>
        [dataset, ...prev.filter((item) => item.dataset_id !== dataset.dataset_id)].slice(0, 16),
      );
    } catch (e: unknown) {
      setDatasetError(e instanceof Error ? e.message : "Dataset review update failed");
    } finally {
      setDatasetBusy(false);
    }
  };

  const handlePromoteDatasetToRegression = async (datasetId: string) => {
    setDatasetBusy(true);
    setDatasetError(null);
    try {
      const dataset = await customDatasetsApi.promoteToRegression(datasetId);
      setGeneratedDataset((prev) => (prev?.dataset_id === dataset.dataset_id ? dataset : prev));
      setSelectedReusableMetricCandidate(dataset.reusable_metric_candidate ?? false);
      setRecentDatasets((prev) =>
        [dataset, ...prev.filter((item) => item.dataset_id !== dataset.dataset_id)].slice(0, 16),
      );
    } catch (e: unknown) {
      setDatasetError(e instanceof Error ? e.message : "Dataset regression promotion failed");
    } finally {
      setDatasetBusy(false);
    }
  };

  const handleStartCaseEdit = (item: CustomDatasetDetail["preview"][number]) => {
    setEditingCaseId(item.id);
    setCasePromptDraft(("turns" in item ? item.persona : item.question) ?? "");
    setCaseEditDraft(("turns" in item ? item.expected_outcome : item.expected_answer) ?? "");
  };

  const handleCancelCaseEdit = () => {
    setEditingCaseId(null);
    setCasePromptDraft("");
    setCaseEditDraft("");
  };

  const handleSaveCaseEdit = async (item: CustomDatasetDetail["preview"][number]) => {
    if (!generatedDataset) return;
    const normalizedPromptDraft = casePromptDraft.trim();
    const normalizedDraft = caseEditDraft.trim();
    if (!normalizedPromptDraft || !normalizedDraft) {
      setDatasetError("Edited case fields cannot be empty");
      return;
    }
    setDatasetBusy(true);
    setDatasetError(null);
    try {
      const dataset = await customDatasetsApi.updateCase(
        generatedDataset.dataset_id,
        item.id,
        "turns" in item
          ? { persona: normalizedPromptDraft, expected_outcome: normalizedDraft }
          : { question: normalizedPromptDraft, expected_answer: normalizedDraft },
      );
      setGeneratedDataset(dataset);
      setSelectedReusableMetricCandidate(dataset.reusable_metric_candidate ?? false);
      setRecentDatasets((prev) =>
        [dataset, ...prev.filter((entry) => entry.dataset_id !== dataset.dataset_id)].slice(0, 16),
      );
      setEditingCaseId(null);
      setCasePromptDraft("");
      setCaseEditDraft("");
    } catch (e: unknown) {
      setDatasetError(e instanceof Error ? e.message : "Dataset case update failed");
    } finally {
      setDatasetBusy(false);
    }
  };

  return (
    <div className="page-shell motion-shell max-w-5xl">
      <header className="page-header motion-hero">
        <p className="page-kicker">Benchmark Factory</p>
        <h1 className="page-title">Dataset Studio</h1>
        <p className="page-subtitle">
          Build regression-ready eval sets from product briefs, docs, or live contexts. Each dataset
          moves through a six-stage lifecycle from brief to promotion.
        </p>
      </header>

      {/* ── Dataset Pipeline ──────────────────────────────── */}
      <div className="panel-surface panel-roomy motion-rise space-y-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="page-kicker mb-1">Pipeline</p>
            <h2 className="text-xl font-semibold tracking-tight" style={{ color: "var(--text-strong)" }}>
              Dataset Lifecycle
            </h2>
            <p className="page-subtitle mt-1 text-sm">
              Follow the six stages from brief to regression-ready artifact.
            </p>
          </div>
          <div className="text-right">
            <p className="micro-copy">Progress</p>
            <p className="mt-1 text-2xl font-bold tabular-nums" style={{ color: "var(--accent)" }}>
              {completedDatasetStages}
              <span className="text-base font-normal" style={{ color: "var(--text-dim)" }}>
                &nbsp;/ {datasetStages.length}
              </span>
            </p>
          </div>
        </div>

        {/* Progress bar */}
        <div className="h-[3px] w-full rounded-full" style={{ background: "rgba(129,177,166,0.14)" }}>
          <div
            className="h-[3px] rounded-full transition-all duration-500"
            style={{
              width: `${(completedDatasetStages / datasetStages.length) * 100}%`,
              background: "var(--accent)",
            }}
          />
        </div>

        {/* Stage cards */}
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
          {datasetStages.map((stage, index) => {
            const isCompleted = stage.status === "completed";
            const isActive = stage.status === "active";

            const borderColor = isCompleted
              ? "rgba(138,229,197,0.35)"
              : isActive
                ? "rgba(241,185,107,0.35)"
                : "rgba(129,177,166,0.12)";

            const bgColor = isCompleted
              ? "rgba(138,229,197,0.07)"
              : isActive
                ? "rgba(241,185,107,0.07)"
                : "rgba(0,0,0,0)";

            const labelColor = isCompleted
              ? "var(--accent)"
              : isActive
                ? "var(--accent-warm)"
                : "var(--text-dim)";

            const chipBg = isCompleted
              ? "rgba(138,229,197,0.14)"
              : isActive
                ? "rgba(241,185,107,0.14)"
                : "rgba(129,177,166,0.07)";

            const chipBorder = isCompleted
              ? "rgba(138,229,197,0.3)"
              : isActive
                ? "rgba(241,185,107,0.3)"
                : "rgba(129,177,166,0.12)";

            return (
              <div
                key={stage.key}
                className="rounded-[1rem] border px-4 py-3 transition-all"
                style={{ borderColor, background: bgColor }}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <p className="text-[0.62rem] font-semibold uppercase tracking-[0.22em]" style={{ color: "var(--text-dim)" }}>
                      Stage {index + 1}
                    </p>
                    <p className="mt-1 text-sm font-semibold leading-tight" style={{ color: labelColor }}>
                      {stage.label}
                    </p>
                  </div>
                  <span
                    className="shrink-0 rounded-full px-2 py-0.5 text-[0.58rem] font-semibold uppercase tracking-[0.16em]"
                    style={{ background: chipBg, border: `1px solid ${chipBorder}`, color: labelColor }}
                  >
                    {stage.status}
                  </span>
                </div>
                <p className="mt-2 text-xs leading-relaxed" style={{ color: "var(--text-dim)" }}>
                  {stage.detail}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Generator form + Preview ───────────────────────── */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(20rem,0.95fr)]">
        {/* Left: form */}
        <div className="space-y-5">
          <section className="panel-surface panel-roomy space-y-5">
            <div>
              <p className="section-caption mb-2">Build</p>
              <h2 className="section-heading">Generate a custom eval set</h2>
              <p className="page-subtitle mt-2 text-sm">
                Describe your product, workflow, or domain and turn it into a reusable QA benchmark.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="control-group">
                <label className="label">Dataset Title</label>
                <input
                  type="text"
                  value={datasetTitle}
                  onChange={(e) => setDatasetTitle(e.target.value)}
                  className="control-surface"
                  placeholder="Support QA, policy assistant, fintech onboarding..."
                />
              </div>

              <div className="control-group">
                <label className="label">Dataset Kind</label>
                <select
                  value={datasetKind}
                  onChange={(e) => setDatasetKind(e.target.value)}
                  className="control-surface"
                >
                  <option value="single_turn">Single-turn QA</option>
                  <option value="conversation">Conversation</option>
                </select>
              </div>

              <div className="control-group">
                <label className="label">Generator Model</label>
                <select
                  value={generatorModel}
                  onChange={(e) => setGeneratorModel(e.target.value)}
                  className="control-surface"
                >
                  {models &&
                    Object.keys(models.models).map((id) => (
                      <option key={id} value={id}>
                        {id}
                      </option>
                    ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-[minmax(0,14rem)_minmax(0,1fr)]">
              <div className="control-group">
                <label className="label">Generation Mode</label>
                <select
                  value={datasetGenerationMode}
                  onChange={(e) => setDatasetGenerationMode(e.target.value)}
                  className="control-surface"
                >
                  <option value="generate_from_scratch">Generate from scratch</option>
                  <option value="generate_from_contexts">Generate from contexts</option>
                  <option value="generate_from_docs">Generate from docs</option>
                </select>
              </div>

              <div className="control-group">
                <label className="label">Source Label</label>
                <input
                  type="text"
                  value={datasetSourceLabel}
                  onChange={(e) => setDatasetSourceLabel(e.target.value)}
                  className="control-surface"
                  placeholder="Knowledge base, onboarding guide, support FAQ..."
                />
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-[minmax(0,1fr)_12rem]">
              <div className="control-group">
                <label className="label">Focus Areas</label>
                <input
                  type="text"
                  value={focusAreas}
                  onChange={(e) => setFocusAreas(e.target.value)}
                  className="control-surface"
                  placeholder="Policy nuance, extraction failures, escalation tone, edge cases..."
                />
              </div>

              <div className="control-group">
                <label className="label">Requested Cases</label>
                <input
                  type="number"
                  value={sampleCount}
                  onChange={(e) => setSampleCount(parseInt(e.target.value, 10) || 12)}
                  min={3}
                  max={100}
                  step={1}
                  className="control-surface"
                />
              </div>
            </div>

            <div className="control-group">
              <label className="label">Project Brief</label>
              <textarea
                value={projectDescription}
                onChange={(e) => setProjectDescription(e.target.value)}
                rows={9}
                className="control-surface"
                placeholder="Explain what your product does, who it serves, the tasks models should handle, critical failure modes, risky edge cases, and what a correct answer should look like."
              />
            </div>

            {datasetGenerationMode !== "generate_from_scratch" && (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="control-group">
                  <label className="label">
                    {datasetGenerationMode === "generate_from_docs" ? "Source Docs" : "Source Contexts"}
                  </label>
                  <textarea
                    value={datasetSourceMaterial}
                    onChange={(e) => setDatasetSourceMaterial(e.target.value)}
                    rows={8}
                    className="control-surface"
                    placeholder={
                      datasetGenerationMode === "generate_from_docs"
                        ? "Paste docs, SOPs, API references, playbooks, or policy text."
                        : "Paste contextual snippets, transcripts, tickets, notes, or domain references."
                    }
                  />
                </div>

                <div className="control-group">
                  <label className="label">Workspace File Paths</label>
                  <textarea
                    value={datasetSourcePaths}
                    onChange={(e) => setDatasetSourcePaths(e.target.value)}
                    rows={8}
                    className="control-surface"
                    placeholder={"docs/guide.md\nREADME.md\napi/services/custom_dataset_service.py"}
                  />
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={handleLoadWorkspaceSourceFiles}
                      disabled={workspaceSourceLoading}
                      className="button-secondary"
                    >
                      {workspaceSourceLoading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                      Scan Workspace
                    </button>
                    <p className="micro-copy">Text-like docs/config/source files only.</p>
                  </div>
                  {workspaceSourceFiles.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {workspaceSourceFiles.map((file) => (
                        <button
                          key={file.path}
                          type="button"
                          onClick={() => handleAddWorkspaceSourcePath(file.path)}
                          className={`provider-chip ${selectedSourcePathSet.has(file.path) ? "opacity-100" : "opacity-60"}`}
                        >
                          {file.path} · {file.size_kb.toFixed(1)} KB
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="button-row">
              <button onClick={handleGenerateDataset} disabled={datasetBusy} className="button-secondary">
                {datasetBusy ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                Generate Dataset
              </button>
              <label className="button-secondary cursor-pointer">
                {datasetBusy ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
                Import JSON
                <input
                  type="file"
                  accept="application/json,.json"
                  onChange={handleImportDataset}
                  disabled={datasetBusy}
                  className="hidden"
                />
              </label>
              {generatedDataset && (
                <p className="micro-copy">
                  Active: <span className="body-copy">{generatedDataset.title}</span>
                </p>
              )}
            </div>

            <p className="micro-copy">
              Import accepts a JSON array or an object with{" "}
              <span className="body-copy">test_cases</span>,{" "}
              <span className="body-copy">cases</span>,{" "}
              <span className="body-copy">items</span>, or{" "}
              <span className="body-copy">data</span>.
            </p>
            <p className="micro-copy">
              Single-turn generates judgeable QA cases. Conversation generates multi-turn scenarios with
              persona, expected outcome, escalation need, and turn structure.
            </p>

            {datasetError && <div className="alert-box alert-danger">{datasetError}</div>}
          </section>

          {/* Dataset Library */}
          <section className="panel-surface panel-roomy space-y-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="section-caption mb-2">Dataset Library</p>
                <h2 className="section-heading">Saved Datasets</h2>
                <p className="page-subtitle mt-1 text-sm">
                  Reuse recently generated or imported datasets without rebuilding them.
                </p>
              </div>
              {recentDatasetsLoading && <Loader2 size={16} className="animate-spin" />}
            </div>

            <div className="control-group">
              <label className="label">Search Library</label>
              <input
                type="text"
                value={datasetLibraryQuery}
                onChange={(e) => setDatasetLibraryQuery(e.target.value)}
                className="control-surface"
                placeholder="Search by title, source, mode, or generator..."
              />
            </div>

            {visibleRecentDatasets.length > 0 ? (
              <div className="space-y-3">
                {visibleRecentDatasets.map((dataset) =>
                  (() => {
                    const sourcePaths = Array.isArray(dataset.source_attribution?.source_paths)
                      ? dataset.source_attribution.source_paths
                      : [];
                    const sourceChunks = Array.isArray(dataset.source_attribution?.source_chunks)
                      ? dataset.source_attribution.source_chunks
                      : [];
                    const filteringSummary =
                      dataset.source_attribution &&
                      typeof dataset.source_attribution.filtering_summary === "object"
                        ? (dataset.source_attribution.filtering_summary as Record<string, number>)
                        : null;
                    const conversationSummary = dataset.conversation_summary;
                    const datasetTags = dataset.dataset_tags ?? [];
                    const datasetTagSummary = dataset.dataset_tag_summary ?? {};
                    const finalizedDiffSummary = dataset.finalized_diff_summary;
                    const finalizedLabel = dataset.finalized_at
                      ? new Date(dataset.finalized_at).toLocaleDateString("tr-TR", {
                          day: "2-digit",
                          month: "short",
                        })
                      : null;
                    const reviewedLabel = dataset.reviewed_at
                      ? new Date(dataset.reviewed_at).toLocaleDateString("tr-TR", {
                          day: "2-digit",
                          month: "short",
                        })
                      : null;
                    const promotedLabel = dataset.promoted_to_regression_at
                      ? new Date(dataset.promoted_to_regression_at).toLocaleDateString("tr-TR", {
                          day: "2-digit",
                          month: "short",
                        })
                      : null;
                    const templateCoverageCount = conversationSummary
                      ? Object.keys(conversationSummary.template_counts ?? {}).length
                      : 0;
                    const variationCoverageCount = conversationSummary
                      ? Object.keys(conversationSummary.variation_counts ?? {}).length
                      : 0;
                    const isSelected = dataset.dataset_id === selectedDatasetId;

                    return (
                      <button
                        key={dataset.dataset_id}
                        type="button"
                        onClick={() => handleLoadRecentDataset(dataset.dataset_id)}
                        className={`flex w-full items-start justify-between gap-4 rounded-[1rem] border p-3 text-left transition-all ${
                          isSelected
                            ? "border-[rgba(138,229,197,0.4)] bg-[rgba(138,229,197,0.06)]"
                            : "border-[rgba(129,177,166,0.14)] hover:border-[rgba(138,229,197,0.25)]"
                        }`}
                      >
                        <div>
                          <p className="body-copy font-medium">{dataset.title}</p>
                          <p className="micro-copy mt-1">
                            {dataset.dataset_kind ?? "single_turn"} ·{" "}
                            {dataset.generation_mode ?? dataset.source_type} ·{" "}
                            {dataset.source_label ?? dataset.generator_model}
                          </p>
                          <div className="mt-2 flex flex-wrap gap-2">
                            <span className="provider-chip">{dataset.review_status}</span>
                            {dataset.review_role && (
                              <span className="provider-chip">{dataset.review_role.toUpperCase()}</span>
                            )}
                            {dataset.reusable_metric_candidate && (
                              <span className="provider-chip">metric candidate</span>
                            )}
                            {finalizedDiffSummary &&
                              (finalizedDiffSummary.changed_count > 0 ||
                                finalizedDiffSummary.added_count > 0 ||
                                finalizedDiffSummary.removed_count > 0) && (
                                <span className="provider-chip">
                                  drift{" "}
                                  {finalizedDiffSummary.changed_count +
                                    finalizedDiffSummary.added_count +
                                    finalizedDiffSummary.removed_count}
                                </span>
                              )}
                            {reviewedLabel && (
                              <span className="provider-chip">reviewed {reviewedLabel}</span>
                            )}
                            {dataset.finalized_case_count > 0 && (
                              <span className="provider-chip">{dataset.finalized_case_count} finalized</span>
                            )}
                            {finalizedLabel && (
                              <span className="provider-chip">finalized {finalizedLabel}</span>
                            )}
                            {dataset.regression_dataset_path && (
                              <span className="provider-chip">regression ready</span>
                            )}
                            {promotedLabel && (
                              <span className="provider-chip">promoted {promotedLabel}</span>
                            )}
                          </div>
                          <div className="mt-2 flex flex-wrap gap-2">
                            <span className="provider-chip">{dataset.sample_count} cases</span>
                            <span className="provider-chip">{dataset.base_case_count} base</span>
                            <span className="provider-chip">
                              {Math.max(0, dataset.sample_count - dataset.base_case_count)} variants
                            </span>
                            {conversationSummary && (
                              <span className="provider-chip">{templateCoverageCount} templates</span>
                            )}
                            {conversationSummary && (
                              <span className="provider-chip">{variationCoverageCount} variations</span>
                            )}
                            {conversationSummary && conversationSummary.escalation_count > 0 && (
                              <span className="provider-chip">
                                {conversationSummary.escalation_count} escalations
                              </span>
                            )}
                            {conversationSummary && (
                              <span className="provider-chip">
                                {conversationSummary.average_user_turns} avg turns
                              </span>
                            )}
                            {sourcePaths.length > 0 && (
                              <span className="provider-chip">{sourcePaths.length} files</span>
                            )}
                            {sourceChunks.length > 0 && (
                              <span className="provider-chip">{sourceChunks.length} chunks</span>
                            )}
                            {filteringSummary && (filteringSummary.duplicate_removed ?? 0) > 0 && (
                              <span className="provider-chip">
                                {filteringSummary.duplicate_removed} duplicates removed
                              </span>
                            )}
                            {filteringSummary &&
                              (filteringSummary.nondeterministic_removed ?? 0) > 0 && (
                                <span className="provider-chip">
                                  {filteringSummary.nondeterministic_removed} vague answers removed
                                </span>
                              )}
                          </div>
                          {datasetTags.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-2">
                              {datasetTags.map((tag) => (
                                <span key={`${dataset.dataset_id}-${tag}`} className="provider-chip">
                                  {tag}
                                </span>
                              ))}
                            </div>
                          )}
                          {Object.keys(datasetTagSummary).length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-2">
                              {Object.entries(datasetTagSummary).map(([tag, value]) => (
                                <span
                                  key={`${dataset.dataset_id}-${tag}-coverage`}
                                  className="provider-chip"
                                >
                                  {tag}: {value}
                                </span>
                              ))}
                            </div>
                          )}
                          {sourcePaths.length > 0 && (
                            <p className="micro-copy mt-2 break-words">
                              {sourcePaths.slice(0, 3).join(" · ")}
                            </p>
                          )}
                        </div>
                        <div className="shrink-0 text-right">
                          <p className="micro-copy">Created</p>
                          <p className="body-copy mt-1">
                            {new Date(dataset.created_at).toLocaleDateString("tr-TR", {
                              day: "2-digit",
                              month: "short",
                            })}
                          </p>
                          {isSelected && <p className="micro-copy mt-2">loaded</p>}
                          <div className="mt-3 flex flex-col gap-2">
                            <button
                              type="button"
                              className="ghost-button"
                              onClick={(event) => {
                                event.stopPropagation();
                                void handleDatasetReviewStatus(dataset.dataset_id, "approved");
                              }}
                              disabled={datasetBusy || dataset.review_status === "approved"}
                            >
                              Approve
                            </button>
                            <button
                              type="button"
                              className="ghost-button"
                              onClick={(event) => {
                                event.stopPropagation();
                                void handleDatasetReviewStatus(dataset.dataset_id, "rejected");
                              }}
                              disabled={datasetBusy || dataset.review_status === "rejected"}
                            >
                              Reject
                            </button>
                            <button
                              type="button"
                              className="ghost-button"
                              onClick={(event) => {
                                event.stopPropagation();
                                void handlePromoteDatasetToRegression(dataset.dataset_id);
                              }}
                              disabled={
                                datasetBusy ||
                                dataset.review_status !== "approved" ||
                                !dataset.finalized_path
                              }
                            >
                              Promote
                            </button>
                          </div>
                        </div>
                      </button>
                    );
                  })(),
                )}
              </div>
            ) : recentDatasets.length > 0 ? (
              <p className="micro-copy mt-4">No dataset matched the current search.</p>
            ) : (
              <p className="micro-copy mt-4">No saved datasets yet.</p>
            )}
          </section>
        </div>

        {/* Right: Preview */}
        <div className="panel-surface panel-quiet space-y-4">
          <div>
            <p className="section-caption mb-2">Preview</p>
            <h3 className="section-heading">Generated Cases</h3>
          </div>

          {generatedDataset ? (
            <div className="space-y-4">
              <div className="rounded-[1.2rem] border border-[rgba(129,177,166,0.18)] p-4">
                <p className="body-copy text-base font-semibold">{generatedDataset.title}</p>
                <div className="mt-3 control-group">
                  <label className="label">Reviewer Role</label>
                  <select
                    value={selectedReviewRole}
                    onChange={(e) =>
                      setSelectedReviewRole(e.target.value as (typeof reviewRoleOptions)[number])
                    }
                    className="control-surface"
                  >
                    {reviewRoleOptions.map((role) => (
                      <option key={role} value={role}>
                        {role.toUpperCase()}
                      </option>
                    ))}
                  </select>
                </div>
                <label className="option-row mt-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedReusableMetricCandidate}
                    onChange={(e) => setSelectedReusableMetricCandidate(e.target.checked)}
                    className="control-check"
                  />
                  <span>Mark as reusable metric candidate</span>
                </label>
                <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="micro-copy">Generator</p>
                    <p className="body-copy mt-1">{generatedDataset.generator_model}</p>
                  </div>
                  <div>
                    <p className="micro-copy">Dataset Kind</p>
                    <p className="body-copy mt-1">{generatedDataset.dataset_kind ?? "single_turn"}</p>
                  </div>
                  <div>
                    <p className="micro-copy">Generation Mode</p>
                    <p className="body-copy mt-1">
                      {generatedDataset.generation_mode ?? "generate_from_scratch"}
                    </p>
                  </div>
                  <div>
                    <p className="micro-copy">Source Label</p>
                    <p className="body-copy mt-1">{generatedDataset.source_label ?? "—"}</p>
                  </div>
                  <div>
                    <p className="micro-copy">Total Cases</p>
                    <p className="body-copy mt-1">{generatedDataset.sample_count}</p>
                  </div>
                  <div>
                    <p className="micro-copy">Base Cases</p>
                    <p className="body-copy mt-1">{generatedDataset.base_case_count}</p>
                  </div>
                  <div>
                    <p className="micro-copy">Stress Variants</p>
                    <p className="body-copy mt-1">
                      {generatedDataset.sample_count - generatedDataset.base_case_count}
                    </p>
                  </div>
                  <div>
                    <p className="micro-copy">Review Status</p>
                    <p className="body-copy mt-1">{generatedDataset.review_status}</p>
                  </div>
                  <div>
                    <p className="micro-copy">Reviewer Role</p>
                    <p className="body-copy mt-1">
                      {generatedDataset.review_role?.toUpperCase() ?? "—"}
                    </p>
                  </div>
                  <div>
                    <p className="micro-copy">Finalized Cases</p>
                    <p className="body-copy mt-1">{generatedDataset.finalized_case_count}</p>
                  </div>
                  <div>
                    <p className="micro-copy">Reviewed</p>
                    <p className="body-copy mt-1">
                      {generatedDataset.reviewed_at
                        ? new Date(generatedDataset.reviewed_at).toLocaleString("tr-TR", {
                            day: "2-digit",
                            month: "short",
                            hour: "2-digit",
                            minute: "2-digit",
                          })
                        : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="micro-copy">Metric Candidate</p>
                    <p className="body-copy mt-1">
                      {generatedDataset.reusable_metric_candidate ? "yes" : "no"}
                    </p>
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap gap-3">
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() =>
                      void handleDatasetReviewStatus(generatedDataset.dataset_id, "approved")
                    }
                    disabled={
                      datasetBusy || generatedDataset.review_status === "approved"
                    }
                  >
                    Approve Dataset
                  </button>
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() =>
                      void handleDatasetReviewStatus(generatedDataset.dataset_id, "rejected")
                    }
                    disabled={
                      datasetBusy || generatedDataset.review_status === "rejected"
                    }
                  >
                    Reject Dataset
                  </button>
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() =>
                      void handlePromoteDatasetToRegression(generatedDataset.dataset_id)
                    }
                    disabled={
                      datasetBusy ||
                      generatedDataset.review_status !== "approved" ||
                      !generatedDataset.finalized_path
                    }
                  >
                    Promote To Regression
                  </button>
                </div>

                {generatedDataset.finalized_path && generatedDataset.finalized_at && (
                  <div className="mt-4 rounded-[1rem] border border-[rgba(129,177,166,0.16)] bg-[rgba(138,229,197,0.05)] p-3">
                    <p className="micro-copy">Finalized Snapshot</p>
                    <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="micro-copy">Saved</p>
                        <p className="body-copy mt-1">
                          {new Date(generatedDataset.finalized_at).toLocaleString("tr-TR", {
                            day: "2-digit",
                            month: "short",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Case Count</p>
                        <p className="body-copy mt-1">{generatedDataset.finalized_case_count}</p>
                      </div>
                      <div>
                        <p className="micro-copy">Changed</p>
                        <p className="body-copy mt-1">
                          {generatedDataset.finalized_diff_summary?.changed_count ?? 0}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Added / Removed</p>
                        <p className="body-copy mt-1">
                          {generatedDataset.finalized_diff_summary?.added_count ?? 0} /{" "}
                          {generatedDataset.finalized_diff_summary?.removed_count ?? 0}
                        </p>
                      </div>
                    </div>
                    <p className="body-copy mt-3 break-words text-sm">
                      {generatedDataset.finalized_path}
                    </p>
                  </div>
                )}

                {generatedDataset.regression_dataset_path &&
                  generatedDataset.promoted_to_regression_at && (
                    <div className="mt-4 rounded-[1rem] border border-[rgba(129,177,166,0.16)] bg-[rgba(138,229,197,0.05)] p-3">
                      <p className="micro-copy">Regression Artifact</p>
                      <p className="body-copy mt-3 break-words text-sm">
                        {generatedDataset.regression_dataset_path}
                      </p>
                    </div>
                  )}

                {generatedDataset.dataset_tags.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {generatedDataset.dataset_tags.map((tag) => (
                      <span key={tag} className="provider-chip">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                {generatedDataset.conversation_summary && (
                  <div className="mt-4 rounded-[1rem] border border-[rgba(129,177,166,0.16)] bg-[rgba(138,229,197,0.04)] p-3">
                    <p className="micro-copy">Conversation Coverage</p>
                    <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="micro-copy">Templates</p>
                        <p className="body-copy mt-1">
                          {Object.keys(generatedDataset.conversation_summary.template_counts).length}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Variations</p>
                        <p className="body-copy mt-1">
                          {Object.keys(generatedDataset.conversation_summary.variation_counts).length}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Escalations</p>
                        <p className="body-copy mt-1">
                          {generatedDataset.conversation_summary.escalation_count}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Avg user turns</p>
                        <p className="body-copy mt-1">
                          {generatedDataset.conversation_summary.average_user_turns}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="rounded-[1.2rem] border border-[rgba(129,177,166,0.16)] p-4">
                <p className="section-caption mb-2">Stress Lab</p>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  {Object.entries(generatedDataset.mutation_summary)
                    .filter(([key]) => key !== "total")
                    .map(([key, value]) => (
                      <div key={key}>
                        <p className="micro-copy">{key.split("_").join(" ")}</p>
                        <p className="body-copy mt-1">{value}</p>
                      </div>
                    ))}
                </div>
              </div>

              <div className="space-y-3">
                {generatedDataset.preview.map((item, index) => (
                  <div
                    key={item.id}
                    className="rounded-[1.1rem] border border-[rgba(129,177,166,0.14)] p-4"
                  >
                    {(() => {
                      const isConversationCase = "turns" in item;
                      const sourceExcerpt = isConversationCase
                        ? (typeof item.metadata?.source_excerpt === "string"
                            ? item.metadata.source_excerpt
                            : "")
                        : (typeof item.mutation_metadata?.source_excerpt === "string"
                            ? item.mutation_metadata.source_excerpt
                            : "");
                      const hasSourceProvenance = Boolean(item.source_case_id || sourceExcerpt);

                      return (
                        <>
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <p className="section-caption mb-1">
                                #{index + 1} {item.category}
                              </p>
                              <p className="body-copy font-medium">
                                {isConversationCase ? (item.persona ?? "Conversation case") : item.question}
                              </p>
                            </div>
                            <div className="flex flex-wrap items-center justify-end gap-2">
                              {!isConversationCase && item.variant_label && (
                                <span className="provider-chip">{item.variant_label}</span>
                              )}
                              {isConversationCase && item.template_id && (
                                <span className="provider-chip">{item.template_id}</span>
                              )}
                              {isConversationCase && item.variation_type && (
                                <span className="provider-chip">{item.variation_type}</span>
                              )}
                              {isConversationCase && (
                                <span className="provider-chip">{item.turn_count} user turns</span>
                              )}
                              {isConversationCase && item.escalation_needed && (
                                <span className="provider-chip">escalation</span>
                              )}
                              <span className="provider-chip">{item.difficulty ?? "medium"}</span>
                            </div>
                          </div>
                          {item.risk_tags.length > 0 && (
                            <div className="mt-3 flex flex-wrap gap-2">
                              {item.risk_tags.map((tag) => (
                                <span key={`${item.id}-${tag}`} className="provider-chip">
                                  {tag}
                                </span>
                              ))}
                            </div>
                          )}
                          {hasSourceProvenance && (
                            <div className="mt-3 rounded-[0.9rem] border border-[rgba(129,177,166,0.14)] bg-[rgba(138,229,197,0.04)] px-3 py-2">
                              <p className="micro-copy">Source provenance</p>
                              <p className="body-copy mt-1 text-sm">
                                {item.source_case_id ? `${item.source_case_id}` : "grounded"}
                                {sourceExcerpt ? ` · ${sourceExcerpt}` : ""}
                              </p>
                            </div>
                          )}
                          {isConversationCase ? (
                            <>
                              <p className="micro-copy mt-3">Expected outcome</p>
                              {editingCaseId === item.id ? (
                                <div className="mt-2 space-y-2">
                                  <textarea
                                    value={casePromptDraft}
                                    onChange={(event) => setCasePromptDraft(event.target.value)}
                                    rows={2}
                                    className="control-surface"
                                  />
                                  <textarea
                                    value={caseEditDraft}
                                    onChange={(event) => setCaseEditDraft(event.target.value)}
                                    rows={4}
                                    className="control-surface"
                                  />
                                  <div className="flex flex-wrap gap-2">
                                    <button
                                      type="button"
                                      className="ghost-button"
                                      onClick={() => void handleSaveCaseEdit(item)}
                                      disabled={datasetBusy}
                                    >
                                      Save Outcome
                                    </button>
                                    <button
                                      type="button"
                                      className="ghost-button"
                                      onClick={handleCancelCaseEdit}
                                      disabled={datasetBusy}
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                <>
                                  <p className="body-copy mt-1 text-sm">{item.expected_outcome}</p>
                                  <button
                                    type="button"
                                    className="ghost-button mt-3"
                                    onClick={() => handleStartCaseEdit(item)}
                                    disabled={datasetBusy}
                                  >
                                    Edit Outcome
                                  </button>
                                </>
                              )}
                              <div className="mt-3 space-y-2">
                                {item.turns.map((turn, turnIndex) => (
                                  <div
                                    key={`${item.id}-turn-${turnIndex}`}
                                    className="rounded-[0.9rem] border border-[rgba(129,177,166,0.12)] px-3 py-2"
                                  >
                                    <p className="micro-copy">
                                      Turn {turnIndex + 1} · {turn.role ?? "evaluator"}
                                    </p>
                                    <p className="body-copy mt-1 text-sm">{turn.content}</p>
                                  </div>
                                ))}
                              </div>
                            </>
                          ) : (
                            <>
                              <p className="micro-copy mt-3">Expected answer</p>
                              {editingCaseId === item.id ? (
                                <div className="mt-2 space-y-2">
                                  <textarea
                                    value={casePromptDraft}
                                    onChange={(event) => setCasePromptDraft(event.target.value)}
                                    rows={2}
                                    className="control-surface"
                                    placeholder="Question"
                                  />
                                  <textarea
                                    value={caseEditDraft}
                                    onChange={(event) => setCaseEditDraft(event.target.value)}
                                    rows={4}
                                    className="control-surface"
                                    placeholder="Expected answer"
                                  />
                                  <div className="flex flex-wrap gap-2">
                                    <button
                                      type="button"
                                      className="ghost-button"
                                      onClick={() => void handleSaveCaseEdit(item)}
                                      disabled={datasetBusy}
                                    >
                                      Save Answer
                                    </button>
                                    <button
                                      type="button"
                                      className="ghost-button"
                                      onClick={handleCancelCaseEdit}
                                      disabled={datasetBusy}
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                <>
                                  <p className="body-copy mt-1 text-sm">{item.expected_answer}</p>
                                  <button
                                    type="button"
                                    className="ghost-button mt-3"
                                    onClick={() => handleStartCaseEdit(item)}
                                    disabled={datasetBusy}
                                  >
                                    Edit Answer
                                  </button>
                                </>
                              )}
                            </>
                          )}
                        </>
                      );
                    })()}
                  </div>
                ))}
              </div>

              {/* Link to Run page */}
              <NavLink
                to="/run"
                className="button-secondary flex w-full items-center justify-center gap-2"
              >
                <Play size={16} />
                Go to Run Evaluation
              </NavLink>
            </div>
          ) : (
            <div className="empty-state min-h-[18rem]">
              <Sparkles size={40} className="mb-3 opacity-40" />
              <p className="body-copy text-center">
                Generate a dataset to preview evaluation cases before launching a run.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
