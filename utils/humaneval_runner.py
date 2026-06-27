"""
HumanEval execution runner using Docker sandbox.
"""
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


DEFAULT_IMAGE = "your-registry/minimal-python:3.11"


def _build_main_file(solution_code: str, test_code: str, entry_point: Optional[str]) -> str:
    """Assemble a runnable Python file with solution + tests."""
    prefix = """
import sys
sys.setrecursionlimit(2000)
""".strip()

    lines = [prefix, "", solution_code.strip(), "", test_code.strip()]

    if "check(" in test_code and entry_point:
        if f"check({entry_point})" not in test_code:
            lines.append("")
            lines.append(f"check({entry_point})")

    return "\n".join(lines) + "\n"


def run_humaneval_in_docker(
    solution_code: str,
    test_code: str,
    entry_point: Optional[str],
    timeout_seconds: int = 5,
    docker_image: str = DEFAULT_IMAGE,
    disable_network: bool = True
) -> Dict[str, Any]:
    """Run HumanEval tests in a Docker sandbox."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        main_path = workdir / "main.py"
        main_path.write_text(_build_main_file(solution_code, test_code, entry_point), encoding="utf-8")

        cmd = [
            "docker", "run", "--rm",
            "--read-only",
            "--pids-limit", "64",
            "--memory", "512m",
            "--cpus", "1",
            "--user", "1000:1000",
            "--workdir", "/work",
            "-v", f"{workdir}:/work:rw",
            "-e", "PYTHONHASHSEED=0",
            "-e", "PYTHONIOENCODING=utf-8"
        ]

        if disable_network:
            cmd.extend(["--network", "none"])

        cmd.extend([docker_image, "python", "/work/main.py"])

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            return {
                "passed": completed.returncode == 0,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timeout": False
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "exit_code": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "timeout": True
            }
