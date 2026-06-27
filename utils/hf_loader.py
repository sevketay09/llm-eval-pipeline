"""
HuggingFace dataset loader utilities.
"""
import json
import os
import random
import re
import time
from typing import Dict, Any, List, Optional
import requests
from datasets import load_dataset


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_LOCAL_DATASET_DIR = "eval_datasets"
TURKISH_FINANCE_LOCAL_RELATIVE = os.path.join("fintech", "turkish_finance_sft.jsonl")


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from model outputs."""
    if not text:
        return ""
    return THINK_PATTERN.sub("", text).strip()


def load_hf_dataset(
    dataset_id: str,
    config: Optional[str] = None,
    split: str = "train",
    sample_size: Optional[int] = None,
    seed: int = 42,
    revision: Optional[str] = None,
    max_retries: int = 5,
    retry_wait_seconds: float = 2.0
) -> Dict[str, Any]:
    """
    Load a HuggingFace dataset and return data + metadata.

    Returns:
        {
            "items": List[Dict],
            "meta": Dict[str, Any]
        }
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    local_dir = os.getenv("HF_LOCAL_DATASET_DIR", DEFAULT_LOCAL_DATASET_DIR)
    local_path = os.path.join(local_dir, TURKISH_FINANCE_LOCAL_RELATIVE)
    if dataset_id == "AlicanKiraz0/Turkish-Finance-SFT-Dataset" and os.path.exists(local_path):
        return _load_local_jsonl(local_path, sample_size=sample_size, seed=seed)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    load_kwargs = {"split": split, "revision": revision}
    if token:
        load_kwargs["token"] = token

    fallback_allowed = (
        dataset_id == "AlicanKiraz0/Turkish-Finance-SFT-Dataset"
        or os.getenv("HF_DATASETS_SERVER_FALLBACK") == "1"
    )

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if config:
                dataset = load_dataset(dataset_id, config, **load_kwargs)
            else:
                dataset = load_dataset(dataset_id, **load_kwargs)
            break
        except Exception as exc:
            last_error = exc
            if fallback_allowed:
                return _load_from_datasets_server(
                    dataset_id=dataset_id,
                    config=config,
                    split=split,
                    sample_size=sample_size,
                    max_retries=max_retries,
                    retry_wait_seconds=retry_wait_seconds
                )
            if attempt < max_retries - 1:
                time.sleep(retry_wait_seconds * (2 ** attempt))
            else:
                token_hint = "HF_TOKEN set" if token else "HF_TOKEN missing"
                raise RuntimeError(
                    f"HF dataset load failed after {max_retries} retries ({token_hint}): {exc}"
                ) from exc
    if dataset is None and last_error:
        raise RuntimeError(f"HF dataset load failed: {last_error}") from last_error
    if sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(min(sample_size, len(dataset))))

    items: List[Dict[str, Any]] = []
    for row in dataset:
        items.append(dict(row))

    meta = {
        "dataset_id": dataset_id,
        "config": config,
        "split": split,
        "revision": revision,
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "num_rows": len(dataset)
    }

    return {"items": items, "meta": meta}


def _load_local_jsonl(path: str, sample_size: Optional[int], seed: int) -> Dict[str, Any]:
    """Load rows from a local JSONL file."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if sample_size:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[: min(sample_size, len(rows))]

    meta = {
        "dataset_id": "local-jsonl",
        "config": None,
        "split": "train",
        "revision": None,
        "fingerprint": None,
        "num_rows": len(rows),
        "source": "local-jsonl",
        "file_path": path
    }

    return {"items": rows, "meta": meta}


def _load_from_datasets_server(
    dataset_id: str,
    config: Optional[str],
    split: str,
    sample_size: Optional[int],
    max_retries: int,
    retry_wait_seconds: float
) -> Dict[str, Any]:
    """Fallback loader using datasets-server rows endpoint."""
    rows: List[Dict[str, Any]] = []
    requested = sample_size or 100
    batch_size = min(100, requested)
    offset = 0
    cfg = config or "default"

    while len(rows) < requested:
        length = min(batch_size, requested - len(rows))
        params = {
            "dataset": dataset_id,
            "config": cfg,
            "split": split,
            "offset": offset,
            "length": length
        }

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = requests.get(DATASETS_SERVER_URL, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                batch = [item.get("row", {}) for item in payload.get("rows", [])]
                rows.extend(batch)
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(retry_wait_seconds * (2 ** attempt))
                else:
                    raise RuntimeError(
                        f"datasets-server load failed after {max_retries} retries: {exc}"
                    ) from exc

        if last_error:
            break
        if not payload.get("rows"):
            break

        offset += length

    meta = {
        "dataset_id": dataset_id,
        "config": cfg,
        "split": split,
        "revision": None,
        "fingerprint": None,
        "num_rows": len(rows),
        "source": "datasets-server",
        "server_url": DATASETS_SERVER_URL
    }

    return {"items": rows[:requested], "meta": meta}


def map_turkish_finance_sft(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Map the Turkish-Finance-SFT dataset into QA-style items.
    Columns: system, user, asistant (with <think>...</think>final)
    """
    mapped = []
    for idx, row in enumerate(rows):
        system_prompt = row.get("system", "") or ""
        user_prompt = row.get("user", "") or ""
        assistant_raw = row.get("asistant", "") or ""
        assistant_final = strip_think(assistant_raw)

        mapped.append({
            "id": row.get("id", f"hf_turkish_finance_{idx}"),
            "category": row.get("category", "fintech"),
            "system_prompt": system_prompt.strip(),
            "question": user_prompt.strip(),
            "expected_answer": assistant_final,
            "reference": assistant_final
        })

    return mapped
