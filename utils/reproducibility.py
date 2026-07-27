"""
Reproducibility Module - Config snapshot and result hashing.

Ensures evaluation runs are reproducible by:
1. Capturing a full config snapshot at run start (model params, env, versions)
2. Computing deterministic hashes of results for integrity verification
3. Linking each result file to its exact configuration
"""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def capture_config_snapshot(
    config_path: str = "config/models.yaml",
    test_config_path: str = "config/tests.yaml",
    registry_path: str = "config/task_registry.yaml",
    runtime_overrides: Optional[Dict[str, Any]] = None,
    model_keys: Optional[list] = None,
    suite: Optional[str] = None,
    selected_tests: Optional[list] = None,
) -> Dict[str, Any]:
    """Capture a complete configuration snapshot for reproducibility.

    Args:
        config_path: Path to models.yaml.
        test_config_path: Path to tests.yaml.
        registry_path: Path to task_registry.yaml.
        runtime_overrides: Any runtime parameter overrides (temp, top_p, etc).
        model_keys: Models being evaluated.
        suite: Test suite name.

    Returns:
        Dict containing full snapshot with timestamps, versions, and configs.
    """
    import yaml

    snapshot: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": _generate_run_id(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "packages": _get_package_versions(),
        "parameters": {
            "model_keys": model_keys or [],
            "suite": suite,
            "selected_tests": selected_tests or [],
            "runtime_overrides": runtime_overrides or {},
        },
        "configs": {},
        "config_hashes": {},
    }

    # Capture config file contents and hashes
    for name, path in [
        ("models", config_path),
        ("tests", test_config_path),
        ("task_registry", registry_path),
    ]:
        file_path = Path(path)
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            snapshot["configs"][name] = yaml.safe_load(content)
            snapshot["config_hashes"][name] = _hash_content(content)
        else:
            snapshot["configs"][name] = None
            snapshot["config_hashes"][name] = None

    # Capture relevant environment variables (keys only for security)
    env_keys = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME_PR",
        "AZURE_OPENAI_DEPLOYMENT_NAME_PTU",
        "AZURE_OPENAI_API_VERSION",
        "VLLM_BASE_URL",
    ]
    snapshot["env_vars_present"] = {k: k in os.environ for k in env_keys}

    return snapshot


def hash_results(results: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of evaluation results.

    Args:
        results: The full results dict from pipeline.

    Returns:
        Hex digest of the results hash.
    """
    # Remove non-deterministic fields before hashing
    hashable = _make_hashable(results)
    content = json.dumps(hashable, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify_results(results: Dict[str, Any], expected_hash: str) -> bool:
    """Verify result integrity against a stored hash.

    Args:
        results: Results dict to verify.
        expected_hash: Previously stored hash to compare against.

    Returns:
        True if hashes match (results not tampered with).
    """
    actual_hash = hash_results(results)
    return actual_hash == expected_hash


def save_reproducible_results(
    results: Dict[str, Any],
    snapshot: Dict[str, Any],
    output_path: str,
) -> Dict[str, str]:
    """Save results with config snapshot and integrity hash.

    Creates two files:
    - {output_path}: Results + embedded snapshot + hash
    - {output_path}.meta.json: Standalone metadata for quick verification

    Args:
        results: Evaluation results.
        snapshot: Config snapshot from capture_config_snapshot().
        output_path: Base path for the output file.

    Returns:
        Dict with paths to created files and the result hash.
    """
    result_hash = hash_results(results)

    # Embed reproducibility data into results
    results["_reproducibility"] = {
        "config_snapshot": snapshot,
        "result_hash": result_hash,
        "hash_algorithm": "sha256",
    }

    # Save main results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # Save standalone metadata
    meta_path = str(output) + ".meta.json"
    meta = {
        "run_id": snapshot["run_id"],
        "timestamp": snapshot["timestamp"],
        "result_hash": result_hash,
        "config_hashes": snapshot["config_hashes"],
        "parameters": snapshot["parameters"],
        "environment": snapshot["environment"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Reproducible results saved: {output_path} (hash: {result_hash[:12]}...)")
    return {
        "results_path": str(output),
        "meta_path": meta_path,
        "result_hash": result_hash,
    }


def _generate_run_id() -> str:
    """Generate a unique run ID based on timestamp + random suffix."""
    import uuid
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"run_{ts}_{short_uuid}"


def _get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = [
        "azure-ai-evaluation",
        "openai",
        "anthropic",
        "pyyaml",
        "numpy",
        "streamlit",
    ]
    versions = {}
    try:
        from importlib.metadata import version, PackageNotFoundError
        for pkg in packages:
            try:
                versions[pkg] = version(pkg)
            except PackageNotFoundError:
                versions[pkg] = "not installed"
    except ImportError:
        pass
    return versions


def _hash_content(content: str) -> str:
    """SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _make_hashable(obj: Any) -> Any:
    """Remove non-deterministic fields and prepare for hashing."""
    if isinstance(obj, dict):
        skip_keys = {"_reproducibility", "timestamp", "run_metadata"}
        return {k: _make_hashable(v) for k, v in sorted(obj.items()) if k not in skip_keys}
    elif isinstance(obj, list):
        return [_make_hashable(item) for item in obj]
    elif isinstance(obj, float):
        # Round floats to avoid floating-point noise
        return round(obj, 10)
    return obj
