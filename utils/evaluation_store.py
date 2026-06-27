"""
Unified evaluation store utilities.
Keeps all runs in a single JSON file and supports legacy migration.
"""

import json
import hashlib
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

STORE_VERSION = "2.0"
DEFAULT_STORE_PATH = "reports/evaluations_store.json"


def _build_default_store() -> Dict[str, Any]:
    return {
        "version": STORE_VERSION,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "runs": []
    }


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def load_store(store_path: str = DEFAULT_STORE_PATH) -> Dict[str, Any]:
    path = Path(store_path)
    data = _safe_read_json(path)
    if not data or "runs" not in data or not isinstance(data["runs"], list):
        return _build_default_store()
    data.setdefault("version", STORE_VERSION)
    data.setdefault("created_at", datetime.now().isoformat())
    data["updated_at"] = datetime.now().isoformat()
    return data


def save_store(store: Dict[str, Any], store_path: str = DEFAULT_STORE_PATH) -> None:
    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store["version"] = STORE_VERSION
    store.setdefault("created_at", datetime.now().isoformat())
    store["updated_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _extract_dataset_signature(run_results: Dict[str, Any]) -> Dict[str, Any]:
    test_names = set()
    for model_data in run_results.get("models", {}).values():
        if not isinstance(model_data, dict):
            continue
        tests = model_data.get("tests", {})
        if isinstance(tests, dict):
            test_names.update(tests.keys())

    tests_sorted = sorted(test_names)
    fingerprint_text = "|".join(tests_sorted)
    fingerprint = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16] if fingerprint_text else "unknown"

    return {
        "tests": tests_sorted,
        "test_count": len(tests_sorted),
        "dataset_fingerprint": fingerprint
    }


def _build_run_id(run_results: Dict[str, Any]) -> str:
    metadata = run_results.get("run_metadata", {}) if isinstance(run_results, dict) else {}
    ts = str(run_results.get("timestamp", ""))
    suite = str(metadata.get("test_suite", "unknown"))
    models = sorted(list((run_results.get("models") or {}).keys()))
    basis = f"{ts}|{suite}|{'|'.join(models)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def normalize_run(run_results: Dict[str, Any], source_file: Optional[str] = None) -> Dict[str, Any]:
    run = deepcopy(run_results)
    run.setdefault("timestamp", datetime.now().isoformat())
    run.setdefault("run_metadata", {})

    run_id = run["run_metadata"].get("run_id") or _build_run_id(run)
    run["run_metadata"]["run_id"] = run_id

    dataset_sig = _extract_dataset_signature(run)
    run["run_metadata"]["dataset_signature"] = dataset_sig

    if source_file:
        run["run_metadata"]["source_file"] = source_file

    return run


def upsert_run(run_results: Dict[str, Any], store_path: str = DEFAULT_STORE_PATH, source_file: Optional[str] = None) -> str:
    store = load_store(store_path)
    run = normalize_run(run_results, source_file=source_file)
    run_id = run["run_metadata"]["run_id"]

    existing_index = None
    for idx, item in enumerate(store["runs"]):
        item_id = (item.get("run_metadata") or {}).get("run_id")
        if item_id == run_id:
            existing_index = idx
            break

    if existing_index is None:
        store["runs"].append(run)
    else:
        store["runs"][existing_index] = run

    store["runs"] = sorted(
        store["runs"],
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )

    save_store(store, store_path)
    return run_id


def migrate_legacy_reports(
    reports_dir: str = "reports",
    store_path: str = DEFAULT_STORE_PATH,
    pattern: str = "eval_*.json"
) -> Dict[str, int]:
    reports_path = Path(reports_dir)
    if not reports_path.exists():
        return {"scanned": 0, "migrated": 0}

    store = load_store(store_path)
    existing_ids = {
        (run.get("run_metadata") or {}).get("run_id")
        for run in store.get("runs", [])
        if isinstance(run, dict)
    }

    scanned = 0
    migrated = 0

    for report_file in sorted(reports_path.glob(pattern)):
        scanned += 1
        data = _safe_read_json(report_file)
        if not data or "models" not in data:
            continue

        normalized = normalize_run(data, source_file=report_file.name)
        run_id = normalized.get("run_metadata", {}).get("run_id")
        if not run_id or run_id in existing_ids:
            continue

        store["runs"].append(normalized)
        existing_ids.add(run_id)
        migrated += 1

    if migrated > 0:
        store["runs"] = sorted(
            store["runs"],
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        save_store(store, store_path)

    return {"scanned": scanned, "migrated": migrated}


def get_runs(store_path: str = DEFAULT_STORE_PATH) -> List[Dict[str, Any]]:
    return load_store(store_path).get("runs", [])


def get_time_series_data(store_path: str = DEFAULT_STORE_PATH) -> List[Dict[str, Any]]:
    """
    Extract flat time-series rows from all runs in the store.

    Each row:
        {
            "timestamp": str (ISO),
            "run_id": str,
            "suite": str,
            "model": str,
            "test_name": str,
            "score": float
        }
    Rows with missing scores are skipped.
    """
    rows = []
    for run in get_runs(store_path):
        ts = run.get("timestamp", "")
        meta = run.get("run_metadata", {}) or {}
        run_id = meta.get("run_id", "")
        suite = meta.get("test_suite", "unknown")

        for model_key, model_data in (run.get("models") or {}).items():
            if not isinstance(model_data, dict):
                continue
            tests = model_data.get("tests") or {}
            for test_name, test_data in tests.items():
                if not isinstance(test_data, dict):
                    continue
                # score may be in summary.overall_score or directly as accuracy/score
                summary = test_data.get("summary") or {}
                score = summary.get("overall_score")
                if score is None:
                    score = test_data.get("accuracy")
                if score is None:
                    score = test_data.get("score")
                if score is None:
                    # look for avg_score inside summary
                    score = summary.get("avg_score")
                if not isinstance(score, (int, float)):
                    continue
                rows.append({
                    "timestamp": ts,
                    "run_id": run_id,
                    "suite": suite,
                    "model": model_key,
                    "test_name": test_name,
                    "score": float(score),
                })
    return rows
