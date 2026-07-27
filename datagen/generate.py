"""Synthetic dataset generation from source documents.

Converts source material (docs, guides, policies) into golden Q/A evaluation
datasets without requiring an existing test set — eliminating the cold-start
barrier for new LLM-powered products.

Core flow: source text → chunks → LLM prompt → Q/A pairs → JSON dataset.

The ``llm_fn`` parameter is injectable so the core logic can be tested without
a real LLM call.

CLI::

    python -m datagen.generate \\
        --source docs/guide.md [--source docs/policy.txt ...] \\
        --project "E-ticaret müşteri destek botu" \\
        --model gpt-4o \\
        [--sample-count 10] [--focus-areas "iade, kargo, hesap"] \\
        [--output eval_datasets/generated/my_dataset.json] \\
        [--format json|text]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

CHUNK_SIZE = 600
MAX_CHUNKS = 8

NONDETERMINISTIC_PATTERNS = (
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

GENERATION_SCHEMA: dict[str, Any] = {
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
                    "source_chunk_id": {"type": "string"},
                },
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You design high-signal evaluation datasets for LLM benchmarking. "
    "Return only valid JSON that matches the schema exactly. "
    "Every expected answer must be deterministic, concise, and verifiable from the source material."
)


def chunk_text(text: str) -> list[dict[str, str]]:
    """Split text into labeled chunks for source attribution.

    Splits on blank lines; merges short paragraphs; hard-cuts long ones.
    Matches the algorithm in ``CustomDatasetService._build_source_chunks``.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [seg.strip() for seg in re.split(r"\n\s*\n", text) if seg.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) <= CHUNK_SIZE:
                current = para
            else:
                start = 0
                while start < len(para) and len(chunks) < MAX_CHUNKS:
                    chunks.append(para[start : start + CHUNK_SIZE].strip())
                    start += CHUNK_SIZE
                current = ""
        if len(chunks) >= MAX_CHUNKS:
            break

    if current and len(chunks) < MAX_CHUNKS:
        chunks.append(current)

    return [
        {"id": f"src_{i:03d}", "text": chunk}
        for i, chunk in enumerate(chunks[:MAX_CHUNKS], start=1)
        if chunk
    ]


def _build_prompt(
    project_description: str,
    chunks: list[dict[str, str]],
    sample_count: int,
    focus_areas: str | None,
) -> str:
    chunks_block = "\n".join(f"- {c['id']}: {c['text']}" for c in chunks)
    source_section = (
        f"Source chunks:\n{chunks_block}\n\n"
        if chunks
        else "Source material: none. Build from project description only.\n\n"
    )
    return (
        "Create an evaluation dataset for the following product or project.\n\n"
        f"Project brief:\n{project_description.strip()}\n\n"
        f"Focus areas: {(focus_areas or 'correctness, edge cases, domain nuance').strip()}\n"
        f"Requested case count: {sample_count}\n\n"
        f"{source_section}"
        "Requirements:\n"
        "- Produce diverse, realistic user prompts.\n"
        "- Mix straightforward, edge-case, and nuanced questions.\n"
        "- Write deterministic expected answers traceable to the source.\n"
        "- Match the language of the project brief (Turkish brief → Turkish questions).\n"
        "- Use category labels like support, policy, reasoning, extraction, onboarding.\n"
        "- Difficulty: easy, medium, or hard.\n"
        "- When source chunks exist, set source_chunk_id to the best matching chunk id.\n"
        "- Never use vague answers like 'it depends' or 'check documentation'.\n"
        f"- Return exactly {sample_count} cases.\n"
    )


def _is_nondeterministic(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer.strip().casefold())
    return any(p in normalized for p in NONDETERMINISTIC_PATTERNS)


def _question_fingerprint(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().casefold())


def _normalize_cases(
    raw_cases: list[Any],
    sample_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    invalid = duplicate = nondeterministic = 0

    for i, case in enumerate(raw_cases[:sample_count], start=1):
        if not isinstance(case, dict):
            invalid += 1
            continue
        q = str(case.get("question") or "").strip()
        a = str(case.get("expected_answer") or "").strip()
        if not q or not a:
            invalid += 1
            continue
        fp = _question_fingerprint(q)
        if fp in seen:
            duplicate += 1
            continue
        if _is_nondeterministic(a):
            nondeterministic += 1
            continue
        seen.add(fp)
        kept.append(
            {
                "id": str(case.get("id") or f"gen_{i:03d}"),
                "category": str(case.get("category") or "custom"),
                "difficulty": str(case.get("difficulty") or "medium"),
                "question": q,
                "expected_answer": a,
                "system_prompt": (str(case.get("system_prompt") or "").strip() or None),
                "source_chunk_id": (str(case.get("source_chunk_id") or "").strip() or None),
                "mutation_type": "base",
                "risk_tags": [],
                "mutation_metadata": {},
            }
        )

    filtering_summary = {
        "input_cases": min(len(raw_cases), sample_count),
        "kept_cases": len(kept),
        "invalid_removed": invalid,
        "duplicate_removed": duplicate,
        "nondeterministic_removed": nondeterministic,
    }
    return kept, filtering_summary


def generate_qa_from_source(
    *,
    source_text: str,
    project_description: str,
    llm_fn: Callable[[list[dict[str, str]]], str],
    sample_count: int = 10,
    focus_areas: str | None = None,
) -> dict[str, Any]:
    """Core generation: source text + project description → golden Q/A dataset.

    Args:
        source_text: Raw document text (docs, guides, policies).
        project_description: What the LLM-powered product does (≥40 chars).
        llm_fn: Callable that takes an OpenAI-style messages list and returns
            the raw text response (string). Injectable for testing.
        sample_count: Number of Q/A pairs to generate.
        focus_areas: Comma-separated topic hints for the LLM.

    Returns:
        dict with keys ``title``, ``test_cases``, ``source_attribution``,
        ``generated_at``, and ``filtering_summary``.

    Raises:
        ValueError: If the LLM response cannot be parsed or yields < 3 valid cases.
    """
    chunks = chunk_text(source_text)
    prompt = _build_prompt(project_description, chunks, sample_count, focus_areas)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    raw_response = llm_fn(messages)
    parsed = _extract_json(raw_response)
    if parsed is None:
        raise ValueError("LLM response could not be parsed as JSON")

    raw_cases = parsed.get("test_cases") if isinstance(parsed, dict) else None
    if not isinstance(raw_cases, list):
        raise ValueError("LLM response missing 'test_cases' array")

    title = (
        parsed.get("title")
        if isinstance(parsed, dict) and isinstance(parsed.get("title"), str)
        else None
    ) or _default_title(project_description)

    cases, filtering_summary = _normalize_cases(raw_cases, sample_count)
    if len(cases) < 3:
        raise ValueError(
            f"Too few valid cases after filtering: {len(cases)} "
            f"(filtering_summary={filtering_summary})"
        )

    return {
        "title": title,
        "test_cases": cases,
        "source_attribution": {
            "project_description": project_description,
            "focus_areas": focus_areas,
            "source_length": len(source_text),
            "source_excerpt": source_text[:500],
            "source_chunks": chunks,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filtering_summary": filtering_summary,
    }


def _extract_json(text: str) -> Any:
    """Extract JSON from a string, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text, count=1)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _default_title(project_description: str) -> str:
    words = project_description.strip().split()
    snippet = " ".join(words[:6])
    return f"Eval Dataset — {snippet}" if snippet else "Eval Dataset"


def load_source_files(paths: list[str]) -> str:
    """Read and concatenate source files."""
    parts: list[str] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Source file not found: {raw}")
        parts.append(f"# {p.name}\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def _make_real_llm_fn(model_key: str) -> Callable[[list[dict[str, str]]], str]:
    """Build a real LLM callable using EvaluationPipeline.initialize_model."""
    from pipeline_runner import EvaluationPipeline

    pipeline = EvaluationPipeline()
    adapter = pipeline.initialize_model(model_key)

    def llm_fn(messages: list[dict[str, str]]) -> str:
        response = adapter.generate(messages, temperature=0.4, max_tokens=4000)
        return str(response.get("content") or "")

    return llm_fn


def _save_output(result: dict[str, Any], output: str) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = f"synthetic-{uuid.uuid4().hex[:8]}"
    payload = {
        "dataset_id": dataset_id,
        "title": result["title"],
        "generator_model": result.get("generator_model", "unknown"),
        "source_type": "generated",
        "generation_mode": "generate_from_docs",
        "source_attribution": result["source_attribution"],
        "generated_at": result["generated_at"],
        "filtering_summary": result["filtering_summary"],
        "test_cases": result["test_cases"],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_text(result: dict[str, Any]) -> None:
    cases = result["test_cases"]
    fs = result["filtering_summary"]
    print(f"Title: {result['title']}")
    print(f"Cases: {len(cases)} kept / {fs['input_cases']} input")
    if fs.get("invalid_removed"):
        print(f"  invalid removed: {fs['invalid_removed']}")
    if fs.get("duplicate_removed"):
        print(f"  duplicates removed: {fs['duplicate_removed']}")
    if fs.get("nondeterministic_removed"):
        print(f"  nondeterministic removed: {fs['nondeterministic_removed']}")
    print()
    for case in cases:
        print(f"[{case['id']}] ({case['category']}, {case['difficulty']})")
        print(f"  Q: {case['question']}")
        print(f"  A: {case['expected_answer']}")
        if case.get("source_chunk_id"):
            print(f"  src: {case['source_chunk_id']}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m datagen.generate",
        description="Generate golden Q/A evaluation datasets from source documents.",
    )
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        default=[],
        metavar="FILE",
        help="Source file (can be repeated for multiple docs).",
    )
    parser.add_argument(
        "--source-text",
        default="",
        metavar="TEXT",
        help="Inline source text (alternative to --source).",
    )
    parser.add_argument(
        "--project",
        required=True,
        metavar="DESCRIPTION",
        help="Product/project description (≥40 chars).",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL_KEY",
        help="Model key as configured in models.yaml (e.g. gpt-4o).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        metavar="N",
        help="Number of Q/A pairs to generate (default: 10).",
    )
    parser.add_argument(
        "--focus-areas",
        default=None,
        metavar="AREAS",
        help="Comma-separated focus areas (e.g. 'iade, kargo, hesap').",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save dataset JSON to this path.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text).",
    )

    args = parser.parse_args(argv)

    source_text = args.source_text
    if args.sources:
        try:
            file_text = load_source_files(args.sources)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        source_text = (source_text + "\n\n" + file_text).strip() if source_text else file_text

    if not source_text and len(args.project) < 40:
        print(
            "Error: --project must be ≥40 chars when no source material provided.",
            file=sys.stderr,
        )
        return 1

    try:
        llm_fn = _make_real_llm_fn(args.model)
    except Exception as exc:
        print(f"Error initializing model '{args.model}': {exc}", file=sys.stderr)
        return 1

    try:
        result = generate_qa_from_source(
            source_text=source_text,
            project_description=args.project,
            llm_fn=llm_fn,
            sample_count=args.sample_count,
            focus_areas=args.focus_areas,
        )
        result["generator_model"] = args.model
    except ValueError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1

    if args.output:
        _save_output(result, args.output)
        print(f"Saved to {args.output}")

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_text(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
