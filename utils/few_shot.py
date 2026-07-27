"""
Few-Shot Evaluation Support - Configurable few-shot examples per task.

Allows tasks to define few-shot examples that are prepended to the prompt,
improving evaluation consistency for tasks that benefit from demonstrations.

Configuration in task_registry.yaml:
    tasks:
      turkish_grammar:
        dataset: eval_datasets/benchmark/turkish_grammar.json
        runner: run_qa_test
        category: turkish
        few_shot:
          enabled: true
          count: 3
          source: eval_datasets/few_shot/turkish_grammar_examples.json

Few-shot example format (JSON file):
    [
      {
        "question": "Hangi cümle dilbilgisi açısından doğrudur?",
        "answer": "İkinci cümle doğrudur çünki..."
      },
      ...
    ]
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# Cache for loaded few-shot examples
_example_cache: Dict[str, List[Dict[str, str]]] = {}


def load_few_shot_examples(
    source_path: str,
    count: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load few-shot examples from a JSON file.

    Args:
        source_path: Path to JSON file containing examples.
        count: Number of examples to use (None = all).

    Returns:
        List of example dicts with 'question' and 'answer' keys.
    """
    cache_key = f"{source_path}:{count}"
    if cache_key in _example_cache:
        return _example_cache[cache_key]

    path = Path(source_path)
    if not path.exists():
        logger.warning(f"Few-shot source not found: {source_path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        if not isinstance(examples, list):
            logger.warning(f"Few-shot source must be a JSON array: {source_path}")
            return []

        if count is not None:
            examples = examples[:count]

        _example_cache[cache_key] = examples
        return examples
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load few-shot examples from {source_path}: {e}")
        return []


def build_few_shot_messages(
    examples: List[Dict[str, str]],
    system_prompt: str,
    user_question: str,
) -> List[Dict[str, str]]:
    """Build message list with few-shot examples prepended.

    Constructs: system → [example_user, example_assistant]* → actual_user

    Args:
        examples: List of {"question": ..., "answer": ...} dicts.
        system_prompt: The system message.
        user_question: The actual user question to evaluate.

    Returns:
        Full message list ready for LLM generation.
    """
    messages = [{"role": "system", "content": system_prompt}]

    for ex in examples:
        q = ex.get("question", ex.get("input", ""))
        a = ex.get("answer", ex.get("output", ex.get("expected_answer", "")))
        if q and a:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_question})
    return messages


def get_few_shot_config(task_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract few-shot configuration from a task registry entry.

    Args:
        task_config: Single task entry from task_registry.yaml.

    Returns:
        Dict with 'enabled', 'count', 'source' or None if not configured.
    """
    fs = task_config.get("few_shot")
    if not fs:
        return None
    if not isinstance(fs, dict):
        return None
    if not fs.get("enabled", False):
        return None
    return fs


def prepare_messages_with_few_shot(
    task_config: Dict[str, Any],
    system_prompt: str,
    user_question: str,
) -> List[Dict[str, str]]:
    """Prepare messages with optional few-shot examples based on task config.

    If the task has few_shot.enabled=true, loads examples and builds
    the full message list. Otherwise, returns standard [system, user] pair.

    Args:
        task_config: Task entry from task_registry.yaml.
        system_prompt: System message.
        user_question: User question.

    Returns:
        Message list for LLM generation.
    """
    fs_config = get_few_shot_config(task_config)
    if not fs_config:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]

    source = fs_config.get("source", "")
    count = fs_config.get("count")
    examples = load_few_shot_examples(source, count)

    if not examples:
        logger.debug(f"No few-shot examples loaded, falling back to standard messages.")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]

    return build_few_shot_messages(examples, system_prompt, user_question)
