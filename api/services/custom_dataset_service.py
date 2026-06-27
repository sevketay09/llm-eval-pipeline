"""Dataset Studio service for generating and storing custom evaluation datasets."""
from __future__ import annotations
from collections import Counter
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.config import get_settings
from api.schemas.evaluations import (
    ConversationCoverageSummary,
    CustomDatasetDetail,
    CustomDatasetGenerateRequest,
    CustomDatasetImportRequest,
    CustomDatasetSummary,
    FinalizedDiffSummary,
)
from utils.stress_lab import expand_stress_lab_cases, summarize_stress_lab_cases
from utils.structured_output import build_response_format, extract_json, validate_schema


DATASET_TAG_ORDER = [
    "standard",
    "variation",
    "edge_case",
    "adversarial",
    "policy",
    "tool_use",
    "rag",
    "structured_output",
]

SUPPORTED_REVIEW_STATUSES = {"draft", "approved", "rejected"}
SUPPORTED_REVIEW_ROLES = {"qa", "sme", "pm"}


DATASET_GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "test_cases"],
    "properties": {
        "title": {"type": "string", "minLength": 3},
        "test_cases": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "category", "difficulty", "question", "expected_answer"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "minLength": 1},
                    "difficulty": {"type": "string", "minLength": 1},
                    "question": {"type": "string", "minLength": 10},
                    "expected_answer": {"type": "string", "minLength": 10},
                    "system_prompt": {"type": "string"},
                    "source_case_id": {"type": "string"},
                },
            },
        },
    },
}

CONVERSATION_DATASET_GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "test_cases"],
    "properties": {
        "title": {"type": "string", "minLength": 3},
        "test_cases": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "category",
                    "difficulty",
                    "persona",
                    "template_id",
                    "variation_type",
                    "expected_outcome",
                    "escalation_needed",
                    "turns",
                ],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "minLength": 1},
                    "difficulty": {"type": "string", "minLength": 1},
                    "persona": {"type": "string", "minLength": 3},
                    "template_id": {"type": "string", "minLength": 3},
                    "variation_type": {"type": "string", "minLength": 3},
                    "expected_outcome": {"type": "string", "minLength": 10},
                    "escalation_needed": {"type": "boolean"},
                    "source_case_id": {"type": "string"},
                    "risk_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "turns": {
                        "type": "array",
                        "minItems": 4,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                                "expected_actions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "check": {"type": "string"},
                            },
                            "anyOf": [
                                {"required": ["role", "content"]},
                                {"required": ["check"]},
                                {"required": ["expected_actions"]},
                            ],
                        },
                    },
                },
            },
        },
    },
}

SOURCE_CHUNK_CHAR_LIMIT = 600
MAX_SOURCE_CHUNKS = 8
DISCOVERABLE_SOURCE_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}
IGNORED_SOURCE_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
}
MAX_DISCOVERABLE_FILE_BYTES = 64_000

SUPPORTED_GENERATION_MODES = {
    "generate_from_scratch",
    "generate_from_contexts",
    "generate_from_docs",
}

SUPPORTED_DATASET_KINDS = {
    "single_turn",
    "conversation",
}

CONVERSATION_TEMPLATE_LIBRARY = (
    {
        "template_id": "retention_followup",
        "category": "retention",
        "persona_pattern": "returning user with prior context",
        "scenario": "The second user turn depends on remembering and reusing facts from the first turn.",
        "variation": "context retention",
    },
    {
        "template_id": "policy_pushback",
        "category": "policy",
        "persona_pattern": "policy-challenging stakeholder",
        "scenario": "The user asks for an exception, then pushes back against the rule on a later turn.",
        "variation": "adversarial follow-up",
    },
    {
        "template_id": "escalation_handoff",
        "category": "escalation",
        "persona_pattern": "frustrated user needing handoff",
        "scenario": "The assistant must gather key details, refuse unsafe shortcuts, and escalate appropriately.",
        "variation": "handoff decision",
    },
    {
        "template_id": "goal_refinement",
        "category": "support",
        "persona_pattern": "user refining intent across turns",
        "scenario": "The initial request is broad and later turns narrow the task or introduce a constraint.",
        "variation": "intention chain",
    },
)

CONVERSATION_TEMPLATE_MAP = {
    template["template_id"]: template for template in CONVERSATION_TEMPLATE_LIBRARY
}

NONDETERMINISTIC_ANSWER_PATTERNS = (
    "it depends",
    "depends on",
    "duruma gore",
    "duruma göre",
    "degisir",
    "değişir",
    "baglidir",
    "bağlıdır",
    "belirtilmemis",
    "belirtilmemiş",
    "bilinmiyor",
    "net degil",
    "net değil",
    "yeterli bilgi yok",
    "cannot determine",
    "not enough information",
    "not specified",
    "consult the documentation",
    "see documentation",
    "refer to documentation",
)


class CustomDatasetService:
    """Generate and persist custom QA datasets from a project brief."""

    def __init__(self, datasets_dir: str | None = None):
        settings = get_settings()
        self._dir = Path(datasets_dir or settings.generated_datasets_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._workspace_root = Path(settings.workspace_root).resolve()

    def generate_dataset(self, request: CustomDatasetGenerateRequest) -> CustomDatasetDetail:
        from pipeline_runner import EvaluationPipeline

        pipeline = EvaluationPipeline()
        generator = pipeline.initialize_model(request.generator_model)
        dataset_kind = self._resolve_dataset_kind(request.dataset_kind)
        generation_mode = self._resolve_generation_mode(request.generation_mode)
        source_material, source_path_records = self._resolve_source_material(request, generation_mode)
        source_chunks = self._build_source_chunks(source_material, generation_mode)

        prompt = self._build_prompt(request, dataset_kind, generation_mode, source_chunks, source_material)
        response_schema = (
            CONVERSATION_DATASET_GENERATION_SCHEMA
            if dataset_kind == "conversation"
            else DATASET_GENERATION_SCHEMA
        )
        response = generator.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You design high-signal evaluation datasets for LLM benchmarking. "
                        "Return only valid JSON that matches the schema exactly. "
                        "Every expected answer or expected outcome must be deterministic, concise, and suitable for automated judging."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=min(max(1600, request.sample_count * 300), 6000),
            response_format=build_response_format(response_schema),
        )

        content = response.get("content") or ""
        parsed, parse_error = extract_json(content)
        if parsed is None:
            raise ValueError(f"Generated dataset could not be parsed: {parse_error}")

        schema_error = validate_schema(parsed, response_schema)
        if schema_error:
            raise ValueError(f"Generated dataset schema validation failed: {schema_error}")

        title = (request.title or parsed.get("title") or "Custom Dataset").strip()
        if dataset_kind == "conversation":
            normalized_cases, filtering_summary = self._normalize_conversation_cases(
                parsed.get("test_cases", []),
                request.sample_count,
            )
            self._annotate_generated_conversation_cases_with_sources(normalized_cases, source_chunks)
            return self._persist_dataset(
                title=title,
                generator_model=request.generator_model,
                source_type="generated",
                source_label=(request.source_label or request.generator_model).strip(),
                dataset_kind=dataset_kind,
                generation_mode=generation_mode,
                source_attribution=self._build_source_attribution(
                    request,
                    dataset_kind,
                    generation_mode,
                    source_chunks,
                    source_material,
                    source_path_records,
                    filtering_summary,
                ),
                project_description=request.project_description,
                focus_areas=request.focus_areas,
                stored_cases=normalized_cases,
                mutation_summary={"base": len(normalized_cases)},
                base_case_count=len(normalized_cases),
            )

        normalized_cases, filtering_summary = self._normalize_cases(parsed.get("test_cases", []), request.sample_count)
        self._annotate_generated_cases_with_sources(normalized_cases, source_chunks)
        return self._persist_dataset(
            title=title,
            generator_model=request.generator_model,
            source_type="generated",
            source_label=(request.source_label or request.generator_model).strip(),
            dataset_kind=dataset_kind,
            generation_mode=generation_mode,
            source_attribution=self._build_source_attribution(
                request,
                dataset_kind,
                generation_mode,
                source_chunks,
                source_material,
                source_path_records,
                filtering_summary,
            ),
            project_description=request.project_description,
            focus_areas=request.focus_areas,
            base_cases=normalized_cases,
        )

    def import_dataset(self, request: CustomDatasetImportRequest) -> CustomDatasetDetail:
        try:
            payload = json.loads(request.dataset_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Imported dataset is not valid JSON: {exc}") from exc

        raw_cases = self._extract_cases_from_payload(payload)
        dataset_kind = self._infer_import_dataset_kind(payload, raw_cases)
        if dataset_kind == "conversation":
            normalized_cases, filtering_summary = self._normalize_conversation_cases(raw_cases, len(raw_cases))
        else:
            normalized_cases, filtering_summary = self._normalize_cases(raw_cases, len(raw_cases))
        title = (request.title or self._extract_title_from_payload(payload) or "Imported Dataset").strip()
        source_label = (request.source_label or title or "Imported JSON").strip()

        if dataset_kind == "conversation":
            return self._persist_dataset(
                title=title,
                generator_model="imported-json",
                source_type="imported",
                source_label=source_label,
                dataset_kind="conversation",
                generation_mode="import_json",
                source_attribution={
                    "kind": "file_import",
                    "source_label": source_label,
                    "filtering_summary": filtering_summary,
                },
                project_description=request.project_description,
                focus_areas=request.focus_areas,
                stored_cases=normalized_cases,
                mutation_summary={"base": len(normalized_cases)},
                base_case_count=len(normalized_cases),
            )

        if self._looks_preexpanded(normalized_cases):
            return self._persist_dataset(
                title=title,
                generator_model="imported-json",
                source_type="imported",
                source_label=source_label,
                dataset_kind="single_turn",
                generation_mode="import_json",
                source_attribution={
                    "kind": "file_import",
                    "source_label": source_label,
                    "filtering_summary": filtering_summary,
                },
                project_description=request.project_description,
                focus_areas=request.focus_areas,
                stored_cases=normalized_cases,
                mutation_summary=summarize_stress_lab_cases(normalized_cases),
                base_case_count=self._count_base_cases(normalized_cases),
            )

        return self._persist_dataset(
            title=title,
            generator_model="imported-json",
            source_type="imported",
            source_label=source_label,
                dataset_kind="single_turn",
            generation_mode="import_json",
            source_attribution={
                "kind": "file_import",
                "source_label": source_label,
                "filtering_summary": filtering_summary,
            },
            project_description=request.project_description,
            focus_areas=request.focus_areas,
            base_cases=normalized_cases,
        )

    def list_datasets(self, limit: int = 50) -> list[CustomDatasetSummary]:
        items: list[CustomDatasetSummary] = []
        meta_files = sorted(self._dir.glob("*.meta.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for meta_path in meta_files[:limit]:
            detail = self._load_detail(meta_path)
            if detail is None:
                continue
            items.append(
                CustomDatasetSummary(
                    dataset_id=detail.dataset_id,
                    title=detail.title,
                    generator_model=detail.generator_model,
                    source_type=detail.source_type,
                    source_label=detail.source_label,
                    dataset_kind=detail.dataset_kind,
                    generation_mode=detail.generation_mode,
                    source_attribution=detail.source_attribution,
                    sample_count=detail.sample_count,
                    base_case_count=detail.base_case_count,
                    created_at=detail.created_at,
                    path=detail.path,
                    mutation_summary=detail.mutation_summary,
                    conversation_summary=detail.conversation_summary,
                    dataset_tags=detail.dataset_tags,
                    dataset_tag_summary=detail.dataset_tag_summary,
                    review_status=detail.review_status,
                    review_role=detail.review_role,
                    reviewed_at=detail.reviewed_at,
                    reusable_metric_candidate=detail.reusable_metric_candidate,
                    finalized_diff_summary=detail.finalized_diff_summary,
                    finalized_at=detail.finalized_at,
                    finalized_path=detail.finalized_path,
                    finalized_case_count=detail.finalized_case_count,
                    promoted_to_regression_at=detail.promoted_to_regression_at,
                    regression_dataset_path=detail.regression_dataset_path,
                )
            )
        return items

    def get_dataset(self, dataset_id: str) -> CustomDatasetDetail | None:
        meta_path = self._dir / f"{dataset_id}.meta.json"
        if not meta_path.exists():
            return None
        return self._load_detail(meta_path)

    def update_review_status(
        self,
        dataset_id: str,
        review_status: str,
        reviewer_role: str | None = None,
        reusable_metric_candidate: bool | None = None,
    ) -> CustomDatasetDetail | None:
        normalized_status = self._resolve_review_status(review_status)
        meta_path = self._dir / f"{dataset_id}.meta.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, TypeError, KeyError, json.JSONDecodeError):
            return None

        self._apply_review_metadata(
            metadata,
            normalized_status,
            reviewer_role,
            reusable_metric_candidate,
        )

        if not self._sync_finalized_snapshot(metadata, meta_path, normalized_status):
            return None

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return self._load_detail(meta_path)

    def update_case(
        self,
        dataset_id: str,
        case_id: str,
        question: str | None = None,
        persona: str | None = None,
        expected_answer: str | None = None,
        expected_outcome: str | None = None,
    ) -> CustomDatasetDetail | None:
        meta_path = self._dir / f"{dataset_id}.meta.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, TypeError, KeyError, json.JSONDecodeError):
            return None

        dataset_path = Path(str(metadata.get("path") or ""))
        if not dataset_path.exists() or not dataset_path.is_file():
            return None

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                stored_cases = json.load(f)
        except (OSError, TypeError, json.JSONDecodeError):
            return None

        if not isinstance(stored_cases, list):
            return None

        dataset_kind = str(metadata.get("dataset_kind") or "single_turn")
        updated = self._apply_case_edit(
            stored_cases=stored_cases,
            dataset_kind=dataset_kind,
            case_id=case_id,
            question=question,
            persona=persona,
            expected_answer=expected_answer,
            expected_outcome=expected_outcome,
        )
        if not updated:
            return None

        metadata["conversation_summary"] = (
            self._summarize_conversation_cases(stored_cases)
            if dataset_kind == "conversation"
            else None
        )
        metadata["dataset_tags"] = self._derive_dataset_tags(
            stored_cases=stored_cases,
            dataset_kind=dataset_kind,
            generation_mode=metadata.get("generation_mode", "generate_from_scratch"),
            source_attribution=metadata.get("source_attribution", {}),
            mutation_summary=metadata.get("mutation_summary", {}),
        )
        metadata["dataset_tag_summary"] = self._summarize_dataset_tag_coverage(
            stored_cases=stored_cases,
            dataset_kind=dataset_kind,
            generation_mode=metadata.get("generation_mode", "generate_from_scratch"),
            source_attribution=metadata.get("source_attribution", {}),
            mutation_summary=metadata.get("mutation_summary", {}),
        )

        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(stored_cases, f, ensure_ascii=False, indent=2)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return self._load_detail(meta_path)

    def promote_to_regression(self, dataset_id: str) -> CustomDatasetDetail | None:
        meta_path = self._dir / f"{dataset_id}.meta.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, TypeError, KeyError, json.JSONDecodeError):
            return None

        if self._resolve_review_status(metadata.get("review_status")) != "approved":
            raise ValueError("Only approved datasets can be promoted to regression")

        finalized_path_value = str(metadata.get("finalized_path") or "").strip()
        if not finalized_path_value:
            raise ValueError("Approved dataset is missing a finalized snapshot")

        finalized_path = Path(finalized_path_value)
        if not finalized_path.exists() or not finalized_path.is_file():
            raise ValueError("Finalized snapshot file does not exist")

        regression_dir = self._workspace_root / "eval_datasets" / "regression" / "promoted"
        regression_dir.mkdir(parents=True, exist_ok=True)
        regression_path = regression_dir / f"{dataset_id}.json"

        with open(finalized_path, "r", encoding="utf-8") as f:
            finalized_cases = json.load(f)
        with open(regression_path, "w", encoding="utf-8") as f:
            json.dump(finalized_cases, f, ensure_ascii=False, indent=2)

        metadata["promoted_to_regression_at"] = datetime.now(timezone.utc).isoformat()
        metadata["regression_dataset_path"] = str(regression_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return self._load_detail(meta_path)

    def resolve_dataset_path(self, dataset_id: str) -> Path | None:
        detail = self.get_dataset(dataset_id)
        if detail is None:
            return None
        path = Path(detail.path)
        if not path.exists() or not path.is_file():
            return None
        return path

    def list_workspace_source_files(self, limit: int = 30) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for path in self._workspace_root.rglob("*"):
            if len(candidates) >= limit * 3:
                break
            if not path.is_file():
                continue
            if self._is_ignored_source_path(path):
                continue
            if path.suffix.lower() not in DISCOVERABLE_SOURCE_EXTENSIONS:
                continue
            try:
                size_bytes = path.stat().st_size
            except OSError:
                continue
            if size_bytes > MAX_DISCOVERABLE_FILE_BYTES:
                continue
            relative_path = path.relative_to(self._workspace_root).as_posix()
            candidates.append(
                {
                    "path": relative_path,
                    "size_kb": round(size_bytes / 1024, 1),
                    "priority": self._score_source_file_candidate(relative_path),
                }
            )

        candidates.sort(key=lambda item: (-item["priority"], item["path"]))
        return [
            {"path": item["path"], "size_kb": item["size_kb"]}
            for item in candidates[:limit]
        ]

    def _load_detail(self, meta_path: Path) -> CustomDatasetDetail | None:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            dataset_path = Path(metadata["path"])
            if not dataset_path.exists():
                return None

            with open(dataset_path, "r", encoding="utf-8") as f:
                cases = json.load(f)

            conversation_summary = metadata.get("conversation_summary")
            if conversation_summary is None and metadata.get("dataset_kind", "single_turn") == "conversation":
                conversation_summary = self._summarize_conversation_cases(cases)
            dataset_tag_summary = metadata.get("dataset_tag_summary")
            if not isinstance(dataset_tag_summary, dict):
                dataset_tag_summary = self._summarize_dataset_tag_coverage(
                    stored_cases=cases,
                    dataset_kind=metadata.get("dataset_kind", "single_turn"),
                    generation_mode=metadata.get("generation_mode", "generate_from_scratch"),
                    source_attribution=metadata.get("source_attribution", {}),
                    mutation_summary=metadata.get("mutation_summary", {}),
                )
            dataset_tags = metadata.get("dataset_tags")
            if not isinstance(dataset_tags, list):
                dataset_tags = [tag for tag in DATASET_TAG_ORDER if dataset_tag_summary.get(tag, 0) > 0]
            review_status, review_role, reviewed_at = self._resolve_review_metadata(metadata)
            finalized_diff_summary = self._resolve_finalized_diff_summary(metadata, cases)
            finalized_at = self._parse_optional_datetime(metadata.get("finalized_at"))
            promoted_to_regression_at = self._parse_optional_datetime(metadata.get("promoted_to_regression_at"))

            return CustomDatasetDetail(
                dataset_id=metadata["dataset_id"],
                title=metadata["title"],
                generator_model=metadata["generator_model"],
                source_type=metadata.get("source_type", "generated"),
                source_label=metadata.get("source_label") or metadata.get("generator_model"),
                dataset_kind=metadata.get("dataset_kind", "single_turn"),
                generation_mode=metadata.get("generation_mode", "generate_from_scratch"),
                source_attribution=metadata.get("source_attribution", {}),
                sample_count=int(metadata.get("sample_count") or len(cases)),
                base_case_count=int(metadata.get("base_case_count") or len(cases)),
                created_at=datetime.fromisoformat(metadata["created_at"]),
                path=metadata["path"],
                mutation_summary=metadata.get("mutation_summary", {}),
                conversation_summary=conversation_summary,
                dataset_tags=dataset_tags,
                dataset_tag_summary=dataset_tag_summary,
                review_status=review_status,
                review_role=review_role,
                reviewed_at=reviewed_at,
                reusable_metric_candidate=bool(metadata.get("reusable_metric_candidate")),
                finalized_diff_summary=finalized_diff_summary,
                finalized_at=finalized_at,
                finalized_path=metadata.get("finalized_path"),
                finalized_case_count=int(metadata.get("finalized_case_count") or 0),
                promoted_to_regression_at=promoted_to_regression_at,
                regression_dataset_path=metadata.get("regression_dataset_path"),
                project_description=metadata.get("project_description", ""),
                focus_areas=metadata.get("focus_areas"),
                preview=cases[: min(8, len(cases))],
            )
        except (OSError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _build_prompt(
        self,
        request: CustomDatasetGenerateRequest,
        dataset_kind: str,
        generation_mode: str,
        source_chunks: list[dict[str, str]],
        source_material: str,
    ) -> str:
        if dataset_kind == "conversation":
            return self._build_conversation_prompt(request, generation_mode, source_chunks, source_material)

        source_block = self._build_source_block(request, generation_mode, source_chunks, source_material)
        return (
            "Create a custom evaluation dataset for the following product or project.\n\n"
            f"Dataset kind: {dataset_kind}\n"
            f"Generation mode: {generation_mode}\n"
            f"Project brief:\n{request.project_description.strip()}\n\n"
            f"Focus areas: {(request.focus_areas or 'general product quality, correctness, edge cases, and domain nuance').strip()}\n"
            f"Requested case count: {request.sample_count}\n"
            f"{source_block}"
            "Requirements:\n"
            "- Produce diverse, realistic user prompts that this product would face.\n"
            "- Mix straightforward, edge-case, failure-prone, and nuanced tasks.\n"
            "- Write deterministic expected answers so that an evaluator can compare model outputs.\n"
            "- Keep questions in Turkish if the brief is Turkish; otherwise match the brief language.\n"
            "- Use category labels like onboarding, support, policy, reasoning, extraction, or domain-specific topics.\n"
            "- Difficulty should be one of easy, medium, hard.\n"
            "- system_prompt is optional and should only be set when a special persona or instruction is required.\n"
            "- Return exactly the requested number of cases.\n"
            "- When source material is provided, ground the cases in that material instead of inventing unrelated scenarios.\n"
            "- When source chunks are provided, set source_case_id to the best matching chunk id for each grounded case.\n"
            "- Never use placeholder or vague expected answers like 'it depends', 'not specified', or 'consult documentation'.\n"
            "- Prefer deterministic answers that can be traced back to the brief or source material."
        )

    def _build_conversation_prompt(
        self,
        request: CustomDatasetGenerateRequest,
        generation_mode: str,
        source_chunks: list[dict[str, str]],
        source_material: str,
    ) -> str:
        source_block = self._build_source_block(request, generation_mode, source_chunks, source_material)
        template_block = self._build_conversation_template_block()
        return (
            "Create a multi-turn conversational evaluation dataset for the following product or project.\n\n"
            "Output format rules:\n"
            "- Each test case must describe a realistic conversation with 2-4 user turns.\n"
            "- turns must alternate between a user turn object (`role` + `content`) and an evaluator expectation turn object (`check` and optional `expected_actions`).\n"
            "- expected_outcome must describe the deterministic end-state the assistant should reach across the full conversation.\n"
            "- template_id must be one of the template ids from the conversation template library.\n"
            "- variation_type must state the main variation used in the scenario.\n"
            "- persona should capture the user's role, attitude, or profile in one short phrase.\n"
            "- escalation_needed should be true only when the conversation should end in handoff, refusal, or escalation.\n"
            "- Include adversarial or failure-prone follow-up turns where appropriate, but keep outcomes deterministic.\n\n"
            f"Generation mode: {generation_mode}\n"
            f"Project brief:\n{request.project_description.strip()}\n\n"
            f"Focus areas: {(request.focus_areas or 'context retention, instruction following, escalation handling, and realistic user journeys').strip()}\n"
            f"Requested conversation count: {request.sample_count}\n"
            f"{template_block}"
            f"{source_block}"
            "Requirements:\n"
            "- Keep conversations in Turkish if the brief is Turkish; otherwise match the brief language.\n"
            "- Use category labels like onboarding, support, policy, retention, escalation, or adversarial.\n"
            "- Difficulty should be one of easy, medium, hard.\n"
            "- Make the final expected outcome judgeable without requiring hidden knowledge.\n"
            "- At least one check should verify context retention or follow-up reasoning.\n"
            "- Cover multiple conversation archetypes from the template library instead of generating near-duplicate support chats.\n"
            "- Include at least one adversarial or pushback-style follow-up when it fits the brief.\n"
            "- Reuse the exact template_id string from the template library for every generated conversation.\n"
            "- When source material is provided, ground the scenario in that material instead of inventing unrelated policies.\n"
            "- When source chunks are provided, set source_case_id to the best matching chunk id for each grounded conversation.\n"
            "- Never use placeholder or vague outcomes like 'it depends', 'check the docs', or 'more information is needed' unless the deterministic expected outcome is to ask a specific clarification question."
        )

    def _build_conversation_template_block(self) -> str:
        lines = ["Conversation template library:"]
        for template in CONVERSATION_TEMPLATE_LIBRARY:
            lines.append(
                "- "
                f"{template['template_id']} | {template['category']} | {template['persona_pattern']} | "
                f"{template['scenario']} | variation: {template['variation']}"
            )
        return "\n".join(lines) + "\n\n"

    def _resolve_generation_mode(self, generation_mode: str | None) -> str:
        candidate = str(generation_mode or "generate_from_scratch").strip() or "generate_from_scratch"
        if candidate not in SUPPORTED_GENERATION_MODES:
            raise ValueError(f"Unsupported dataset generation mode: {candidate}")
        return candidate

    def _resolve_dataset_kind(self, dataset_kind: str | None) -> str:
        candidate = str(dataset_kind or "single_turn").strip() or "single_turn"
        if candidate not in SUPPORTED_DATASET_KINDS:
            raise ValueError(f"Unsupported dataset kind: {candidate}")
        return candidate

    def _build_source_block(
        self,
        request: CustomDatasetGenerateRequest,
        generation_mode: str,
        source_chunks: list[dict[str, str]],
        source_material: str,
    ) -> str:
        source_label = (request.source_label or "").strip()
        if generation_mode == "generate_from_scratch":
            return "Source material: none. Build the dataset from the product brief and focus areas only.\n\n"

        if not source_material:
            raise ValueError(f"{generation_mode} requires source material")

        intro = (
            "Use the following contextual snippets to derive grounded evaluation cases."
            if generation_mode == "generate_from_contexts"
            else "Use the following documentation or reference material to derive grounded evaluation cases."
        )
        label_line = f"Source label: {source_label}\n" if source_label else ""
        chunks_block = "\n".join(
            f"- {chunk['id']}: {chunk['text']}" for chunk in source_chunks
        )
        return (
            f"{intro}\n"
            f"{label_line}"
            f"Source material length: {len(source_material)} characters\n"
            f"Source chunks:\n{chunks_block}\n\n"
        )

    def _build_source_attribution(
        self,
        request: CustomDatasetGenerateRequest,
        dataset_kind: str,
        generation_mode: str,
        source_chunks: list[dict[str, str]],
        source_material: str,
        source_path_records: list[dict[str, str]],
        filtering_summary: dict[str, int],
    ) -> dict[str, Any]:
        source_label = (request.source_label or "").strip()
        source_attribution = {
            "kind": "project_brief" if generation_mode == "generate_from_scratch" else generation_mode,
            "dataset_kind": dataset_kind,
            "project_description": request.project_description,
            "focus_areas": request.focus_areas,
        }
        if source_label:
            source_attribution["source_label"] = source_label
        if source_material:
            source_attribution["source_excerpt"] = source_material[:500]
            source_attribution["source_length"] = len(source_material)
        if source_chunks:
            source_attribution["source_chunks"] = source_chunks
        if source_path_records:
            source_attribution["source_paths"] = [record["path"] for record in source_path_records]
        if dataset_kind == "conversation":
            source_attribution["conversation_templates"] = [
                {
                    "template_id": template["template_id"],
                    "category": template["category"],
                    "variation": template["variation"],
                }
                for template in CONVERSATION_TEMPLATE_LIBRARY
            ]
        if filtering_summary:
            source_attribution["filtering_summary"] = filtering_summary
        return source_attribution

    def _resolve_source_material(
        self,
        request: CustomDatasetGenerateRequest,
        generation_mode: str,
    ) -> tuple[str, list[dict[str, str]]]:
        inline_material = (request.source_material or "").strip()
        source_path_records = self._load_source_path_records(request.source_paths)

        if generation_mode == "generate_from_scratch":
            return inline_material, source_path_records

        combined_parts = []
        if inline_material:
            combined_parts.append(inline_material)
        combined_parts.extend(
            f"File: {record['path']}\n{record['content']}" for record in source_path_records
        )
        combined_material = "\n\n".join(part for part in combined_parts if part.strip()).strip()
        return combined_material, source_path_records

    def _load_source_path_records(self, source_paths: list[str]) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for raw_path in source_paths:
            relative_path = str(raw_path or "").strip()
            if not relative_path:
                continue
            resolved_path = self._resolve_workspace_path(relative_path)
            if not resolved_path.exists() or not resolved_path.is_file():
                raise ValueError(f"Source path does not exist or is not a file: {relative_path}")
            content = resolved_path.read_text(encoding="utf-8")
            records.append(
                {
                    "path": relative_path,
                    "content": content[:8000],
                }
            )
        return records

    def _resolve_workspace_path(self, relative_path: str) -> Path:
        candidate = (self._workspace_root / relative_path).resolve()
        if not candidate.is_relative_to(self._workspace_root):
            raise ValueError(f"Source path escapes workspace root: {relative_path}")
        return candidate

    def _is_ignored_source_path(self, path: Path) -> bool:
        return any(part in IGNORED_SOURCE_DIR_NAMES for part in path.parts)

    def _score_source_file_candidate(self, relative_path: str) -> int:
        score = 0
        lowered = relative_path.lower()
        if lowered.endswith("readme.md"):
            score += 6
        if "/docs/" in f"/{lowered}" or lowered.startswith("docs/"):
            score += 5
        if "/config/" in f"/{lowered}" or lowered.startswith("config/"):
            score += 3
        if any(token in lowered for token in ("guide", "policy", "handbook", "faq", "playbook", "overview")):
            score += 2
        return score

    def _build_source_chunks(self, source_material: str, generation_mode: str) -> list[dict[str, str]]:
        if generation_mode == "generate_from_scratch":
            return []

        text = source_material.strip()
        if not text:
            return []

        paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]
        if not paragraphs:
            paragraphs = [text]

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            current = self._append_paragraph_to_source_chunks(chunks, current, paragraph)
            if len(chunks) >= MAX_SOURCE_CHUNKS:
                break
        if current and len(chunks) < MAX_SOURCE_CHUNKS:
            chunks.append(current)

        return [
            {"id": f"src_{index:03d}", "text": chunk[:SOURCE_CHUNK_CHAR_LIMIT].strip()}
            for index, chunk in enumerate(chunks[:MAX_SOURCE_CHUNKS], start=1)
            if chunk.strip()
        ]

    def _append_paragraph_to_source_chunks(
        self,
        chunks: list[str],
        current: str,
        paragraph: str,
    ) -> str:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= SOURCE_CHUNK_CHAR_LIMIT:
            return candidate

        if current:
            chunks.append(current)

        if len(paragraph) <= SOURCE_CHUNK_CHAR_LIMIT:
            return paragraph

        self._append_long_source_paragraph(chunks, paragraph)
        return ""

    def _append_long_source_paragraph(self, chunks: list[str], paragraph: str) -> None:
        start = 0
        while start < len(paragraph) and len(chunks) < MAX_SOURCE_CHUNKS:
            chunks.append(paragraph[start:start + SOURCE_CHUNK_CHAR_LIMIT].strip())
            start += SOURCE_CHUNK_CHAR_LIMIT

    def _annotate_generated_cases_with_sources(
        self,
        cases: list[dict[str, Any]],
        source_chunks: list[dict[str, str]],
    ) -> None:
        if not source_chunks:
            return

        chunk_lookup = {chunk["id"]: chunk["text"] for chunk in source_chunks}
        for case in cases:
            source_case_id = self._coerce_optional_text(case.get("source_case_id"))
            if not source_case_id:
                continue
            chunk_text = chunk_lookup.get(source_case_id)
            if not chunk_text:
                continue
            mutation_metadata = case.get("mutation_metadata")
            if not isinstance(mutation_metadata, dict):
                mutation_metadata = {}
                case["mutation_metadata"] = mutation_metadata
            mutation_metadata["source_chunk_id"] = source_case_id
            mutation_metadata["source_excerpt"] = chunk_text[:240]

    def _annotate_generated_conversation_cases_with_sources(
        self,
        cases: list[dict[str, Any]],
        source_chunks: list[dict[str, str]],
    ) -> None:
        if not source_chunks:
            return

        chunk_lookup = {chunk["id"]: chunk["text"] for chunk in source_chunks}
        for case in cases:
            source_case_id = self._coerce_optional_text(case.get("source_case_id"))
            if not source_case_id:
                continue
            chunk_text = chunk_lookup.get(source_case_id)
            if not chunk_text:
                continue
            metadata = case.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                case["metadata"] = metadata
            metadata["source_chunk_id"] = source_case_id
            metadata["source_excerpt"] = chunk_text[:240]

    def _normalize_cases(self, cases: list[dict], expected_count: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
        normalized: list[dict] = []
        seen_questions: set[str] = set()
        invalid_removed = 0
        duplicate_removed = 0
        nondeterministic_removed = 0
        for index, case in enumerate(cases[:expected_count], start=1):
            if not isinstance(case, dict):
                invalid_removed += 1
                continue

            normalized_case = self._normalize_case(case, index)
            if not normalized_case["question"] or not normalized_case["expected_answer"]:
                invalid_removed += 1
                continue

            question_fingerprint = self._question_fingerprint(normalized_case["question"])
            if question_fingerprint in seen_questions:
                duplicate_removed += 1
                continue
            if self._is_nondeterministic_expected_answer(normalized_case["expected_answer"]):
                nondeterministic_removed += 1
                continue

            seen_questions.add(question_fingerprint)
            normalized.append(normalized_case)

        if len(normalized) < 3:
            raise ValueError("Dataset did not include enough valid test cases")
        filtering_summary = {
            "input_cases": min(len(cases), expected_count),
            "kept_cases": len(normalized),
            "invalid_removed": invalid_removed,
            "duplicate_removed": duplicate_removed,
            "nondeterministic_removed": nondeterministic_removed,
        }
        return normalized, filtering_summary

    def _normalize_conversation_cases(
        self,
        cases: list[dict[str, Any]],
        expected_count: int,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        normalized: list[dict[str, Any]] = []
        seen_conversations: set[str] = set()
        invalid_removed = 0
        duplicate_removed = 0
        nondeterministic_removed = 0

        for index, case in enumerate(cases[:expected_count], start=1):
            if not isinstance(case, dict):
                invalid_removed += 1
                continue

            normalized_case = self._normalize_conversation_case(case, index)
            if len(normalized_case["turns"]) < 4 or normalized_case["turn_count"] < 2 or not normalized_case["expected_outcome"]:
                invalid_removed += 1
                continue

            conversation_fingerprint = self._conversation_fingerprint(normalized_case)
            if conversation_fingerprint in seen_conversations:
                duplicate_removed += 1
                continue
            if self._is_nondeterministic_expected_answer(normalized_case["expected_outcome"]):
                nondeterministic_removed += 1
                continue

            seen_conversations.add(conversation_fingerprint)
            normalized.append(normalized_case)

        if len(normalized) < 3:
            raise ValueError("Conversation dataset did not include enough valid test cases")

        filtering_summary = {
            "input_cases": min(len(cases), expected_count),
            "kept_cases": len(normalized),
            "invalid_removed": invalid_removed,
            "duplicate_removed": duplicate_removed,
            "nondeterministic_removed": nondeterministic_removed,
        }
        return normalized, filtering_summary

    def _normalize_case(self, case: dict[str, Any], index: int) -> dict[str, Any]:
        return {
            "id": str(case.get("id") or f"custom_{index:03d}"),
            "category": str(case.get("category") or "custom"),
            "difficulty": str(case.get("difficulty") or "medium"),
            "question": str(case.get("question") or "").strip(),
            "expected_answer": str(case.get("expected_answer") or "").strip(),
            "system_prompt": self._coerce_optional_text(case.get("system_prompt")),
            "source_case_id": self._coerce_optional_text(case.get("source_case_id")),
            "mutation_type": self._coerce_optional_text(case.get("mutation_type")),
            "variant_label": self._coerce_optional_text(case.get("variant_label")),
            "risk_tags": self._coerce_string_list(case.get("risk_tags")),
            "mutation_metadata": case.get("mutation_metadata") if isinstance(case.get("mutation_metadata"), dict) else {},
        }

    def _normalize_conversation_case(self, case: dict[str, Any], index: int) -> dict[str, Any]:
        normalized_turns, user_turn_count = self._normalize_conversation_turns(case.get("turns"))
        resolved_template_id = self._resolve_conversation_template_id(case.get("template_id"), case.get("category"))

        return {
            "id": str(case.get("id") or f"conversation_{index:03d}"),
            "category": str(case.get("category") or "conversation"),
            "difficulty": str(case.get("difficulty") or "medium"),
            "persona": self._coerce_optional_text(case.get("persona")) or "general user",
            "template_id": resolved_template_id,
            "variation_type": self._resolve_conversation_variation_type(case.get("variation_type"), resolved_template_id),
            "expected_outcome": str(case.get("expected_outcome") or "").strip(),
            "escalation_needed": bool(case.get("escalation_needed", False)),
            "turn_count": user_turn_count,
            "source_case_id": self._coerce_optional_text(case.get("source_case_id")),
            "risk_tags": self._coerce_string_list(case.get("risk_tags")),
            "metadata": case.get("metadata") if isinstance(case.get("metadata"), dict) else {},
            "turns": normalized_turns,
        }

    def _resolve_conversation_template_id(self, template_id: Any, category: Any) -> str:
        candidate = self._coerce_optional_text(template_id)
        if candidate and candidate in CONVERSATION_TEMPLATE_MAP:
            return candidate

        normalized_category = self._coerce_optional_text(category) or ""
        for template in CONVERSATION_TEMPLATE_LIBRARY:
            if template["category"] == normalized_category:
                return str(template["template_id"])
        return str(CONVERSATION_TEMPLATE_LIBRARY[0]["template_id"])

    def _resolve_conversation_variation_type(self, variation_type: Any, template_id: Any) -> str:
        candidate = self._coerce_optional_text(variation_type)
        if candidate:
            return candidate

        template_key = self._coerce_optional_text(template_id)
        if template_key and template_key in CONVERSATION_TEMPLATE_MAP:
            return str(CONVERSATION_TEMPLATE_MAP[template_key]["variation"])
        return str(CONVERSATION_TEMPLATE_LIBRARY[0]["variation"])

    def _normalize_conversation_turns(self, raw_turns: Any) -> tuple[list[dict[str, Any]], int]:
        normalized_turns: list[dict[str, Any]] = []
        user_turn_count = 0
        if not isinstance(raw_turns, list):
            return normalized_turns, user_turn_count

        for turn in raw_turns:
            normalized_turn = self._normalize_conversation_turn(turn)
            if normalized_turn is None:
                continue
            if normalized_turn["role"] == "user" and normalized_turn["content"]:
                user_turn_count += 1
            normalized_turns.append(normalized_turn)

        return normalized_turns, user_turn_count

    def _normalize_conversation_turn(self, turn: Any) -> dict[str, Any] | None:
        if not isinstance(turn, dict):
            return None

        normalized_turn = {
            "role": self._coerce_optional_text(turn.get("role")),
            "content": self._coerce_optional_text(turn.get("content")),
            "expected_actions": self._coerce_string_list(turn.get("expected_actions")),
            "check": self._coerce_optional_text(turn.get("check")),
            "metadata": turn.get("metadata") if isinstance(turn.get("metadata"), dict) else {},
        }
        if any([
            normalized_turn["role"],
            normalized_turn["content"],
            normalized_turn["expected_actions"],
            normalized_turn["check"],
        ]):
            return normalized_turn
        return None

    def _coerce_optional_text(self, value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    def _question_fingerprint(self, question: str) -> str:
        return re.sub(r"\s+", " ", question.strip().casefold())

    def _conversation_fingerprint(self, case: dict[str, Any]) -> str:
        user_turns = [
            str(turn.get("content") or "").strip()
            for turn in case.get("turns", [])
            if isinstance(turn, dict) and turn.get("role") == "user"
        ]
        combined = " | ".join(user_turns + [str(case.get("expected_outcome") or "").strip()])
        return self._question_fingerprint(combined)

    def _is_nondeterministic_expected_answer(self, expected_answer: str) -> bool:
        normalized = re.sub(r"\s+", " ", expected_answer.strip().casefold())
        return any(pattern in normalized for pattern in NONDETERMINISTIC_ANSWER_PATTERNS)

    def _coerce_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _summarize_conversation_cases(self, stored_cases: list[dict[str, Any]]) -> dict[str, Any]:
        if not stored_cases:
            return ConversationCoverageSummary().model_dump()

        template_counts: Counter[str] = Counter()
        variation_counts: Counter[str] = Counter()
        escalation_count = 0
        total_user_turns = 0

        for case in stored_cases:
            template_id = self._coerce_optional_text(case.get("template_id"))
            if template_id:
                template_counts[template_id] += 1

            variation_type = self._coerce_optional_text(case.get("variation_type"))
            if variation_type:
                variation_counts[variation_type] += 1

            if bool(case.get("escalation_needed", False)):
                escalation_count += 1

            turn_count = case.get("turn_count")
            if isinstance(turn_count, int):
                total_user_turns += max(0, turn_count)

        summary = ConversationCoverageSummary(
            total_conversations=len(stored_cases),
            escalation_count=escalation_count,
            average_user_turns=round(total_user_turns / len(stored_cases), 2),
            template_counts=dict(sorted(template_counts.items())),
            variation_counts=dict(sorted(variation_counts.items())),
        )
        return summary.model_dump()

    def _derive_dataset_tags(
        self,
        *,
        stored_cases: list[dict[str, Any]],
        dataset_kind: str,
        generation_mode: str,
        source_attribution: dict[str, Any],
        mutation_summary: dict[str, int],
    ) -> list[str]:
        tag_summary = self._summarize_dataset_tag_coverage(
            stored_cases=stored_cases,
            dataset_kind=dataset_kind,
            generation_mode=generation_mode,
            source_attribution=source_attribution,
            mutation_summary=mutation_summary,
        )
        return [tag for tag in DATASET_TAG_ORDER if tag_summary.get(tag, 0) > 0]

    def _summarize_dataset_tag_coverage(
        self,
        *,
        stored_cases: list[dict[str, Any]],
        dataset_kind: str,
        generation_mode: str,
        source_attribution: dict[str, Any],
        mutation_summary: dict[str, int],
    ) -> dict[str, int]:
        tag_counts: Counter[str] = Counter()
        variation_count = sum(
            value
            for key, value in mutation_summary.items()
            if key != "base" and isinstance(value, int)
        )

        for case in stored_cases:
            for tag in self._derive_case_tags(
                case,
                dataset_kind=dataset_kind,
                generation_mode=generation_mode,
                source_attribution=source_attribution,
            ):
                tag_counts[tag] += 1

        if variation_count > 0 and tag_counts.get("variation", 0) == 0:
            tag_counts["variation"] = min(len(stored_cases), variation_count)

        return {
            tag: tag_counts[tag]
            for tag in DATASET_TAG_ORDER
            if tag_counts.get(tag, 0) > 0
        }

    def _derive_case_tags(
        self,
        case: dict[str, Any],
        *,
        dataset_kind: str,
        generation_mode: str,
        source_attribution: dict[str, Any],
    ) -> list[str]:
        tags: set[str] = {"standard"}

        if dataset_kind == "conversation" or self._case_has_variation(case):
            tags.add("variation")
        if self._case_is_edge_case(case):
            tags.add("edge_case")
        if self._case_is_adversarial(case):
            tags.add("adversarial")
        if self._case_is_policy(case):
            tags.add("policy")
        if self._case_uses_tool(case):
            tags.add("tool_use")
        if self._case_is_rag_grounded(case, generation_mode, source_attribution):
            tags.add("rag")
        if self._case_has_structured_output(case):
            tags.add("structured_output")

        return [tag for tag in DATASET_TAG_ORDER if tag in tags]

    def _case_has_variation(self, case: dict[str, Any]) -> bool:
        mutation_type = self._coerce_optional_text(case.get("mutation_type"))
        if mutation_type and mutation_type != "base":
            return True
        if self._coerce_optional_text(case.get("variant_label")):
            return True
        if self._coerce_optional_text(case.get("variation_type")):
            return True
        return False

    def _dataset_has_edge_cases(self, stored_cases: list[dict[str, Any]]) -> bool:
        return any(self._case_is_edge_case(case) for case in stored_cases)

    def _case_is_edge_case(self, case: dict[str, Any]) -> bool:
        difficulty = (self._coerce_optional_text(case.get("difficulty")) or "").casefold()
        category = (self._coerce_optional_text(case.get("category")) or "").casefold()
        risk_tags = {tag.casefold() for tag in self._coerce_string_list(case.get("risk_tags"))}
        return (
            difficulty == "hard"
            or any(token in category for token in ("edge", "exception", "failure"))
            or bool(risk_tags.intersection({"edge_case", "edge-case", "failure_mode"}))
        )

    def _dataset_has_adversarial_cases(self, stored_cases: list[dict[str, Any]]) -> bool:
        return any(self._case_is_adversarial(case) for case in stored_cases)

    def _case_is_adversarial(self, case: dict[str, Any]) -> bool:
        tokens = ("adversarial", "pushback", "jailbreak", "prompt_injection", "misuse")
        category = (self._coerce_optional_text(case.get("category")) or "").casefold()
        variation_type = (self._coerce_optional_text(case.get("variation_type")) or "").casefold()
        template_id = (self._coerce_optional_text(case.get("template_id")) or "").casefold()
        risk_tags = {tag.casefold() for tag in self._coerce_string_list(case.get("risk_tags"))}
        return (
            any(token in category for token in tokens)
            or any(token in variation_type for token in tokens)
            or any(token in template_id for token in tokens)
            or any(any(token in risk_tag for token in tokens) for risk_tag in risk_tags)
        )

    def _dataset_has_policy_cases(self, stored_cases: list[dict[str, Any]]) -> bool:
        return any(self._case_is_policy(case) for case in stored_cases)

    def _case_is_policy(self, case: dict[str, Any]) -> bool:
        category = (self._coerce_optional_text(case.get("category")) or "").casefold()
        template_id = (self._coerce_optional_text(case.get("template_id")) or "").casefold()
        risk_tags = {tag.casefold() for tag in self._coerce_string_list(case.get("risk_tags"))}
        return (
            "policy" in category
            or "policy" in template_id
            or any("policy" in risk_tag for risk_tag in risk_tags)
        )

    def _dataset_has_tool_use_cases(self, stored_cases: list[dict[str, Any]]) -> bool:
        return any(self._case_uses_tool(case) for case in stored_cases)

    def _case_uses_tool(self, case: dict[str, Any]) -> bool:
        category = (self._coerce_optional_text(case.get("category")) or "").casefold()
        risk_tags = {tag.casefold() for tag in self._coerce_string_list(case.get("risk_tags"))}
        return (
            "tool" in category
            or any("tool" in risk_tag for risk_tag in risk_tags)
            or self._case_has_tool_expectation(case)
        )

    def _case_has_tool_expectation(self, case: dict[str, Any]) -> bool:
        turns = case.get("turns") if isinstance(case.get("turns"), list) else []
        return any(
            isinstance(turn, dict)
            and any("tool" in action.casefold() for action in self._coerce_string_list(turn.get("expected_actions")))
            for turn in turns
        )

    def _dataset_has_rag_grounding(
        self,
        stored_cases: list[dict[str, Any]],
        generation_mode: str,
        source_attribution: dict[str, Any],
    ) -> bool:
        return any(
            self._case_is_rag_grounded(case, generation_mode, source_attribution)
            for case in stored_cases
        )

    def _case_is_rag_grounded(
        self,
        case: dict[str, Any],
        generation_mode: str,
        source_attribution: dict[str, Any],
    ) -> bool:
        if generation_mode in {"generate_from_docs", "generate_from_contexts"}:
            return True
        if isinstance(source_attribution.get("source_chunks"), list) and source_attribution.get("source_chunks"):
            return True
        if isinstance(source_attribution.get("source_paths"), list) and source_attribution.get("source_paths"):
            return True
        return bool(self._coerce_optional_text(case.get("source_case_id")))

    def _dataset_has_structured_output_cases(self, stored_cases: list[dict[str, Any]]) -> bool:
        return any(self._case_has_structured_output(case) for case in stored_cases)

    def _case_has_structured_output(self, case: dict[str, Any]) -> bool:
        category = (self._coerce_optional_text(case.get("category")) or "").casefold()
        risk_tags = {tag.casefold() for tag in self._coerce_string_list(case.get("risk_tags"))}
        if any(token in category for token in ("structured_output", "json", "schema", "extraction")):
            return True
        if any(any(token in risk_tag for token in ("structured_output", "json", "schema")) for risk_tag in risk_tags):
            return True
        candidate_text = ""
        if self._coerce_optional_text(case.get("expected_answer")):
            candidate_text = str(case.get("expected_answer") or "").strip()
        elif self._coerce_optional_text(case.get("expected_outcome")):
            candidate_text = str(case.get("expected_outcome") or "").strip()
        return self._looks_structured_output_text(candidate_text)

    def _looks_structured_output_text(self, text: str) -> bool:
        candidate = text.strip()
        if not candidate:
            return False
        return (
            (candidate.startswith("{") and candidate.endswith("}"))
            or (candidate.startswith("[") and candidate.endswith("]"))
        )

    def _resolve_review_status(self, review_status: Any) -> str:
        candidate = str(review_status or "draft").strip().casefold() or "draft"
        if candidate not in SUPPORTED_REVIEW_STATUSES:
            raise ValueError(f"Unsupported review status: {candidate}")
        return candidate

    def _resolve_review_role(self, review_role: Any) -> str:
        candidate = str(review_role or "qa").strip().casefold() or "qa"
        if candidate not in SUPPORTED_REVIEW_ROLES:
            raise ValueError(f"Unsupported review role: {candidate}")
        return candidate

    def _parse_optional_datetime(self, value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        return datetime.fromisoformat(value)

    def _resolve_review_metadata(self, metadata: dict[str, Any]) -> tuple[str, str | None, datetime | None]:
        review_status = self._resolve_review_status(metadata.get("review_status"))
        review_role_value = metadata.get("review_role")
        review_role = self._resolve_review_role(review_role_value) if review_role_value else None
        reviewed_at = self._parse_optional_datetime(metadata.get("reviewed_at"))
        return review_status, review_role, reviewed_at

    def _apply_case_edit(
        self,
        *,
        stored_cases: list[dict[str, Any]],
        dataset_kind: str,
        case_id: str,
        question: str | None,
        persona: str | None,
        expected_answer: str | None,
        expected_outcome: str | None,
    ) -> bool:
        normalized_case_id = str(case_id).strip()
        if not normalized_case_id:
            raise ValueError("Case id is required")

        for case in stored_cases:
            if not isinstance(case, dict) or str(case.get("id") or "").strip() != normalized_case_id:
                continue

            if dataset_kind == "conversation":
                self._apply_conversation_case_edit(case, persona, expected_outcome)
                return True

            self._apply_single_turn_case_edit(case, question, expected_answer)
            return True

        return False

    def _apply_conversation_case_edit(
        self,
        case: dict[str, Any],
        persona: str | None,
        expected_outcome: str | None,
    ) -> None:
        next_persona = self._coerce_optional_text(persona)
        next_outcome = self._coerce_optional_text(expected_outcome)
        if next_persona is None and next_outcome is None:
            raise ValueError("Conversation case edit requires persona or expected_outcome")
        if next_persona is not None:
            case["persona"] = next_persona
        if next_outcome is not None:
            case["expected_outcome"] = next_outcome

    def _apply_single_turn_case_edit(
        self,
        case: dict[str, Any],
        question: str | None,
        expected_answer: str | None,
    ) -> None:
        next_question = self._coerce_optional_text(question)
        next_answer = self._coerce_optional_text(expected_answer)
        if next_question is None and next_answer is None:
            raise ValueError("Single-turn case edit requires question or expected_answer")
        if next_question is not None:
            case["question"] = next_question
        if next_answer is not None:
            case["expected_answer"] = next_answer

    def _resolve_finalized_diff_summary(
        self,
        metadata: dict[str, Any],
        current_cases: list[dict[str, Any]],
    ) -> FinalizedDiffSummary | None:
        finalized_path_value = str(metadata.get("finalized_path") or "").strip()
        if not finalized_path_value:
            return None

        finalized_path = Path(finalized_path_value)
        if not finalized_path.exists() or not finalized_path.is_file():
            return None

        try:
            with open(finalized_path, "r", encoding="utf-8") as f:
                finalized_cases = json.load(f)
        except (OSError, TypeError, json.JSONDecodeError):
            return None

        if not isinstance(finalized_cases, list):
            return None

        current_index = {
            str(case.get("id") or f"current_{idx}"): json.dumps(case, sort_keys=True, ensure_ascii=False)
            for idx, case in enumerate(current_cases)
            if isinstance(case, dict)
        }
        finalized_index = {
            str(case.get("id") or f"finalized_{idx}"): json.dumps(case, sort_keys=True, ensure_ascii=False)
            for idx, case in enumerate(finalized_cases)
            if isinstance(case, dict)
        }
        current_keys = set(current_index)
        finalized_keys = set(finalized_index)
        shared_keys = current_keys.intersection(finalized_keys)
        changed_count = sum(1 for key in shared_keys if current_index[key] != finalized_index[key])
        unchanged_count = len(shared_keys) - changed_count

        return FinalizedDiffSummary(
            current_case_count=len(current_index),
            finalized_case_count=len(finalized_index),
            added_count=len(current_keys - finalized_keys),
            removed_count=len(finalized_keys - current_keys),
            changed_count=changed_count,
            unchanged_count=unchanged_count,
        )

    def _apply_review_metadata(
        self,
        metadata: dict[str, Any],
        normalized_status: str,
        reviewer_role: str | None,
        reusable_metric_candidate: bool | None,
    ) -> None:
        metadata["review_status"] = normalized_status
        if normalized_status == "draft":
            metadata["review_role"] = None
            metadata["reviewed_at"] = None
            metadata["reusable_metric_candidate"] = False
            return

        metadata["review_role"] = self._resolve_review_role(reviewer_role or metadata.get("review_role") or "qa")
        metadata["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        if reusable_metric_candidate is not None:
            metadata["reusable_metric_candidate"] = bool(reusable_metric_candidate)
        elif not isinstance(metadata.get("reusable_metric_candidate"), bool):
            metadata["reusable_metric_candidate"] = False

    def _sync_finalized_snapshot(self, metadata: dict[str, Any], meta_path: Path, normalized_status: str) -> bool:
        dataset_path = Path(str(metadata.get("path") or ""))
        finalized_path = self._finalized_snapshot_path(meta_path)
        if normalized_status != "approved":
            metadata["finalized_at"] = None
            metadata["finalized_path"] = None
            metadata["finalized_case_count"] = 0
            if finalized_path.exists():
                finalized_path.unlink()
            return True

        if not dataset_path.exists() or not dataset_path.is_file():
            return False

        with open(dataset_path, "r", encoding="utf-8") as f:
            stored_cases = json.load(f)
        with open(finalized_path, "w", encoding="utf-8") as f:
            json.dump(stored_cases, f, ensure_ascii=False, indent=2)
        metadata["finalized_at"] = datetime.now(timezone.utc).isoformat()
        metadata["finalized_path"] = str(finalized_path)
        metadata["finalized_case_count"] = len(stored_cases) if isinstance(stored_cases, list) else 0
        return True

    def _finalized_snapshot_path(self, meta_path: Path) -> Path:
        dataset_id = meta_path.name.removesuffix(".meta.json")
        return self._dir / f"{dataset_id}.finalized.json"

    def _persist_dataset(
        self,
        *,
        title: str,
        generator_model: str,
        source_type: str,
        source_label: str,
        dataset_kind: str,
        generation_mode: str,
        source_attribution: dict[str, Any],
        project_description: str,
        focus_areas: str | None,
        base_cases: list[dict[str, Any]] | None = None,
        stored_cases: list[dict[str, Any]] | None = None,
        mutation_summary: dict[str, int] | None = None,
        base_case_count: int | None = None,
    ) -> CustomDatasetDetail:
        if stored_cases is None:
            if base_cases is None:
                raise ValueError("Dataset persistence requires base_cases or stored_cases")
            stored_cases, mutation_summary = expand_stress_lab_cases(base_cases)
            base_case_count = len(base_cases)
        elif mutation_summary is None:
            mutation_summary = summarize_stress_lab_cases(stored_cases)

        if base_case_count is None:
            base_case_count = self._count_base_cases(stored_cases)

        conversation_summary = (
            self._summarize_conversation_cases(stored_cases)
            if dataset_kind == "conversation"
            else None
        )
        dataset_tags = self._derive_dataset_tags(
            stored_cases=stored_cases,
            dataset_kind=dataset_kind,
            generation_mode=generation_mode,
            source_attribution=source_attribution,
            mutation_summary=mutation_summary or {},
        )
        dataset_tag_summary = self._summarize_dataset_tag_coverage(
            stored_cases=stored_cases,
            dataset_kind=dataset_kind,
            generation_mode=generation_mode,
            source_attribution=source_attribution,
            mutation_summary=mutation_summary or {},
        )

        dataset_id = self._build_dataset_id(title)
        created_at = datetime.now(timezone.utc)
        dataset_path = self._dir / f"{dataset_id}.json"
        meta_path = self._dir / f"{dataset_id}.meta.json"

        metadata = {
            "dataset_id": dataset_id,
            "title": title,
            "generator_model": generator_model,
            "source_type": source_type,
            "source_label": source_label,
            "dataset_kind": dataset_kind,
            "generation_mode": generation_mode,
            "source_attribution": source_attribution,
            "sample_count": len(stored_cases),
            "base_case_count": base_case_count,
            "created_at": created_at.isoformat(),
            "path": str(dataset_path),
            "project_description": project_description,
            "focus_areas": focus_areas,
            "mutation_summary": mutation_summary or {},
            "conversation_summary": conversation_summary,
            "dataset_tags": dataset_tags,
            "dataset_tag_summary": dataset_tag_summary,
            "review_status": "draft",
            "review_role": None,
            "reviewed_at": None,
            "reusable_metric_candidate": False,
            "finalized_at": None,
            "finalized_path": None,
            "finalized_case_count": 0,
            "promoted_to_regression_at": None,
            "regression_dataset_path": None,
        }

        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(stored_cases, f, ensure_ascii=False, indent=2)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return CustomDatasetDetail(
            dataset_id=dataset_id,
            title=title,
            generator_model=generator_model,
            source_type=source_type,
            source_label=source_label,
            dataset_kind=dataset_kind,
            generation_mode=generation_mode,
            source_attribution=source_attribution,
            sample_count=len(stored_cases),
            base_case_count=base_case_count,
            created_at=created_at,
            path=str(dataset_path),
            mutation_summary=mutation_summary or {},
            conversation_summary=conversation_summary,
            dataset_tags=dataset_tags,
            dataset_tag_summary=dataset_tag_summary,
            review_status="draft",
            review_role=None,
            reviewed_at=None,
            reusable_metric_candidate=False,
            finalized_at=None,
            finalized_path=None,
            finalized_case_count=0,
            promoted_to_regression_at=None,
            regression_dataset_path=None,
            project_description=project_description,
            focus_areas=focus_areas,
            preview=stored_cases[: min(8, len(stored_cases))],
        )

    def _extract_cases_from_payload(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("test_cases", "cases", "items", "data"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError("Imported dataset must be a JSON array or an object with a test_cases/cases/items/data array")

    def _extract_title_from_payload(self, payload: Any) -> str | None:
        if isinstance(payload, dict):
            title = payload.get("title") or payload.get("name") or payload.get("dataset_name")
            if isinstance(title, str) and title.strip():
                return title.strip()
        return None

    def _infer_import_dataset_kind(self, payload: Any, cases: list[dict[str, Any]]) -> str:
        if isinstance(payload, dict):
            candidate = self._coerce_optional_text(payload.get("dataset_kind"))
            if candidate in SUPPORTED_DATASET_KINDS:
                return str(candidate)

        for case in cases:
            if not isinstance(case, dict):
                continue
            if isinstance(case.get("turns"), list):
                return "conversation"
            if self._coerce_optional_text(case.get("expected_outcome")):
                return "conversation"
        return "single_turn"

    def _looks_preexpanded(self, cases: list[dict[str, Any]]) -> bool:
        return any((case.get("mutation_type") or "").strip() not in ("", "base") for case in cases)

    def _count_base_cases(self, cases: list[dict[str, Any]]) -> int:
        base_cases = [case for case in cases if (case.get("mutation_type") or "base") == "base"]
        if base_cases:
            return len(base_cases)
        source_ids = {str(case.get("source_case_id") or case.get("id") or "").strip() for case in cases}
        source_ids.discard("")
        return len(source_ids)

    def _build_dataset_id(self, title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "custom-dataset"
        return f"{slug}-{uuid.uuid4().hex[:8]}"