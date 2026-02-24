"""Download and verify Turkish Finance SFT dataset via datasets-server + curl."""
import json
import os
import subprocess
from typing import Any, Dict, List
from urllib.parse import urlencode


DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"
DATASET_ID = "AlicanKiraz0/Turkish-Finance-SFT-Dataset"
SPLIT = "train"
CONFIG = "default"
LOCAL_PATH = os.path.join("eval_datasets", "fintech", "turkish_finance_sft.jsonl")


def _fetch_rows_via_curl(offset: int, length: int) -> List[Dict[str, Any]]:
    params = urlencode({
        "dataset": DATASET_ID,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": length,
    })
    url = f"{DATASETS_SERVER_URL}?{params}"
    result = subprocess.run(
        ["curl", "-s", "-X", "GET", url],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")
    payload = json.loads(result.stdout)
    return [item.get("row", {}) for item in payload.get("rows", [])]


def download_dataset(batch_size: int = 100) -> int:
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)

    offset = 0
    total = 0
    with open(LOCAL_PATH, "w", encoding="utf-8") as handle:
        while True:
            rows = _fetch_rows_via_curl(offset, batch_size)
            if not rows:
                break
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)
            offset += batch_size
            print(f"Downloaded {total} rows...")

    return total


def quick_verify() -> None:
    with open(LOCAL_PATH, "r", encoding="utf-8") as handle:
        first = handle.readline().strip()
    if not first:
        raise RuntimeError("Local dataset file is empty")
    row = json.loads(first)
    print("Sample keys:", sorted(row.keys()))
    print("Sample user:", str(row.get("user", ""))[:200])


def main() -> int:
    print(f"Downloading {DATASET_ID} split={SPLIT}...")
    total = download_dataset()
    print(f"Saved to {LOCAL_PATH} (rows={total})")
    quick_verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
