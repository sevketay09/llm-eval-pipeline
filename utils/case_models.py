from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _pick_first_string(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _build_metadata(
    payload: Mapping[str, Any],
    excluded_keys: set[str],
) -> Dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in excluded_keys
    }


@dataclass(frozen=True)
class SingleTurnCase:
    case_id: str
    input_text: str
    expected_output: Optional[str] = None
    reference_output: Optional[str] = None
    category: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @property
    def has_expected_output(self) -> bool:
        return "expected_answer" in self.raw_payload or "expected_output" in self.raw_payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SingleTurnCase":
        input_text = _pick_first_string(payload, ("question", "input", "prompt"))
        if input_text is None:
            raise ValueError(
                "SingleTurnCase requires one of: question, input, prompt"
            )

        excluded_keys = {
            "id",
            "question",
            "input",
            "prompt",
            "expected_answer",
            "expected_output",
            "reference",
            "category",
            "system_prompt",
        }

        expected_output = _pick_first_string(
            payload,
            ("expected_answer", "expected_output"),
        )
        reference_output = _pick_first_string(payload, ("reference",)) or expected_output

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            input_text=input_text,
            expected_output=expected_output,
            reference_output=reference_output,
            category=_pick_first_string(payload, ("category",)),
            system_prompt=_pick_first_string(payload, ("system_prompt",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class ReasoningCase:
    case_id: str
    question: str
    expected_reasoning: Optional[str] = None
    expected_answer: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_text(self) -> str:
        return self.question

    @property
    def expected_output(self) -> Optional[str]:
        return self.expected_answer

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @property
    def has_expected_output(self) -> bool:
        return "expected_answer" in self.raw_payload or "expected_output" in self.raw_payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReasoningCase":
        question = _pick_first_string(payload, ("question", "input", "prompt"))
        if question is None:
            raise ValueError(
                "ReasoningCase requires one of: question, input, prompt"
            )

        excluded_keys = {
            "id",
            "question",
            "input",
            "prompt",
            "expected_reasoning",
            "expected_answer",
            "expected_output",
            "category",
        }

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            question=question,
            expected_reasoning=_pick_first_string(payload, ("expected_reasoning",)),
            expected_answer=_pick_first_string(
                payload,
                ("expected_answer", "expected_output"),
            ),
            category=_pick_first_string(payload, ("category",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class AgenticCase:
    case_id: str
    task: str
    available_tools: List[Any] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    expected_tool_arguments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_text(self) -> str:
        return self.task

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AgenticCase":
        task = _pick_first_string(payload, ("task", "input", "prompt"))
        if task is None:
            raise ValueError(
                "AgenticCase requires one of: task, input, prompt"
            )

        excluded_keys = {
            "id",
            "task",
            "input",
            "prompt",
            "available_tools",
            "expected_tools",
            "expected_tool_arguments",
            "expected_params_by_tool",
            "category",
        }

        available_tools = payload.get("available_tools", [])
        if not isinstance(available_tools, list):
            available_tools = []

        expected_tools = payload.get("expected_tools", payload.get("expected_tool_set", []))
        if not isinstance(expected_tools, list):
            expected_tools = []

        expected_tool_arguments = payload.get(
            "expected_tool_arguments",
            payload.get("expected_params_by_tool", {}),
        )
        if not isinstance(expected_tool_arguments, dict):
            expected_tool_arguments = {}

        normalized_expected_tool_arguments: Dict[str, Dict[str, Any]] = {}
        for tool_name, expected_params in expected_tool_arguments.items():
            if not isinstance(tool_name, str) or not isinstance(expected_params, dict):
                continue
            normalized_expected_tool_arguments[tool_name] = dict(expected_params)

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            task=task,
            available_tools=available_tools,
            expected_tools=[tool for tool in expected_tools if isinstance(tool, str)],
            expected_tool_arguments=normalized_expected_tool_arguments,
            category=_pick_first_string(payload, ("category",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class FunctionCallingCase:
    case_id: str
    input_text: str
    available_tools: List[Any] = field(default_factory=list)
    expected_tool: Optional[str] = None
    expected_params: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FunctionCallingCase":
        input_text = _pick_first_string(payload, ("prompt", "input", "question"))
        if input_text is None:
            raise ValueError(
                "FunctionCallingCase requires one of: prompt, input, question"
            )

        excluded_keys = {
            "id",
            "prompt",
            "input",
            "question",
            "available_tools",
            "expected_tool",
            "expected_params",
            "category",
        }

        available_tools = payload.get("available_tools", [])
        if not isinstance(available_tools, list):
            available_tools = []

        expected_params = payload.get("expected_params", {})
        if not isinstance(expected_params, dict):
            expected_params = {}

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            input_text=input_text,
            available_tools=available_tools,
            expected_tool=_pick_first_string(payload, ("expected_tool",)),
            expected_params=expected_params,
            category=_pick_first_string(payload, ("category",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class ToolWorkflowCase:
    case_id: str
    input_text: str
    expected_tools: List[str] = field(default_factory=list)
    expected_order: bool = False
    is_parallel: bool = True
    expected_outcome: Optional[str] = None
    max_turns: int = 5
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ToolWorkflowCase":
        input_text = _pick_first_string(payload, ("prompt", "input", "question"))
        if input_text is None:
            raise ValueError(
                "ToolWorkflowCase requires one of: prompt, input, question"
            )

        excluded_keys = {
            "id",
            "prompt",
            "input",
            "question",
            "expected_tools",
            "expected_order",
            "is_parallel",
            "expected_outcome",
            "max_turns",
            "category",
        }

        expected_tools = payload.get("expected_tools", [])
        if not isinstance(expected_tools, list):
            expected_tools = []

        max_turns = payload.get("max_turns", 5)
        if not isinstance(max_turns, int):
            max_turns = 5

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            input_text=input_text,
            expected_tools=[tool for tool in expected_tools if isinstance(tool, str)],
            expected_order=bool(payload.get("expected_order", False)),
            is_parallel=bool(payload.get("is_parallel", True)),
            expected_outcome=_pick_first_string(payload, ("expected_outcome",)),
            max_turns=max_turns,
            category=_pick_first_string(payload, ("category",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class ToolErrorRecoveryCase:
    case_id: str
    test_type: str
    input_text: str
    tool_name: Optional[str] = None
    primary_tool: Optional[str] = None
    fallback_tools: List[str] = field(default_factory=list)
    error_config: Dict[str, Any] = field(default_factory=dict)
    primary_tool_error_config: Dict[str, Any] = field(default_factory=dict)
    expected_behavior: Optional[str] = None
    expected_outcome: Optional[str] = None
    expected_response_contains: List[str] = field(default_factory=list)
    should_not_contain: List[str] = field(default_factory=list)
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "id": self.case_id,
            "test_type": self.test_type,
            "prompt": self.input_text,
        }
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        if self.primary_tool:
            payload["primary_tool"] = self.primary_tool
        if self.fallback_tools:
            payload["fallback_tools"] = list(self.fallback_tools)
        if self.error_config:
            payload["error_config"] = dict(self.error_config)
        if self.primary_tool_error_config:
            payload["primary_tool_error_config"] = dict(self.primary_tool_error_config)
        if self.expected_behavior:
            payload["expected_behavior"] = self.expected_behavior
        if self.expected_outcome:
            payload["expected_outcome"] = self.expected_outcome
        if self.expected_response_contains:
            payload["expected_response_contains"] = list(self.expected_response_contains)
        if self.should_not_contain:
            payload["should_not_contain"] = list(self.should_not_contain)
        if self.difficulty:
            payload["difficulty"] = self.difficulty
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ToolErrorRecoveryCase":
        input_text = _pick_first_string(payload, ("prompt", "input", "question"))
        if input_text is None:
            raise ValueError(
                "ToolErrorRecoveryCase requires one of: prompt, input, question"
            )

        test_type = _pick_first_string(payload, ("test_type",))
        if test_type is None:
            raise ValueError("ToolErrorRecoveryCase requires: test_type")

        excluded_keys = {
            "id",
            "test_type",
            "prompt",
            "input",
            "question",
            "tool_name",
            "primary_tool",
            "fallback_tools",
            "error_config",
            "primary_tool_error_config",
            "expected_behavior",
            "expected_outcome",
            "expected_response_contains",
            "should_not_contain",
            "difficulty",
        }

        fallback_tools = payload.get("fallback_tools", [])
        if not isinstance(fallback_tools, list):
            fallback_tools = []

        expected_response_contains = payload.get("expected_response_contains", [])
        if not isinstance(expected_response_contains, list):
            expected_response_contains = []

        should_not_contain = payload.get("should_not_contain", [])
        if not isinstance(should_not_contain, list):
            should_not_contain = []

        error_config = payload.get("error_config", {})
        if not isinstance(error_config, dict):
            error_config = {}

        primary_tool_error_config = payload.get("primary_tool_error_config", {})
        if not isinstance(primary_tool_error_config, dict):
            primary_tool_error_config = {}

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            test_type=test_type,
            input_text=input_text,
            tool_name=_pick_first_string(payload, ("tool_name",)),
            primary_tool=_pick_first_string(payload, ("primary_tool",)),
            fallback_tools=[tool for tool in fallback_tools if isinstance(tool, str)],
            error_config=error_config,
            primary_tool_error_config=primary_tool_error_config,
            expected_behavior=_pick_first_string(payload, ("expected_behavior",)),
            expected_outcome=_pick_first_string(payload, ("expected_outcome",)),
            expected_response_contains=[item for item in expected_response_contains if isinstance(item, str)],
            should_not_contain=[item for item in should_not_contain if isinstance(item, str)],
            difficulty=_pick_first_string(payload, ("difficulty",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class ConversationTurn:
    role: Optional[str] = None
    content: Optional[str] = None
    expected_actions: List[str] = field(default_factory=list)
    check: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ConversationTurn":
        role = _pick_first_string(payload, ("role",))
        content = _pick_first_string(payload, ("content",))
        expected_actions = payload.get("expected_actions", [])
        if not isinstance(expected_actions, list):
            expected_actions = []
        check = _pick_first_string(payload, ("check",))

        if role is None and content is None and not expected_actions and check is None:
            raise ValueError(
                "ConversationTurn requires at least one of: role, content, expected_actions, check"
            )

        excluded_keys = {"role", "content", "expected_actions", "check"}
        return cls(
            role=role,
            content=content,
            expected_actions=[item for item in expected_actions if isinstance(item, str)],
            check=check,
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class MultiTurnConversationCase:
    case_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MultiTurnConversationCase":
        raw_turns = payload.get("turns", [])
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError("MultiTurnConversationCase requires a non-empty turns list")

        turns = []
        for index, turn in enumerate(raw_turns):
            if not isinstance(turn, dict):
                raise ValueError(
                    f"MultiTurnConversationCase turn at index {index} must be an object"
                )
            turns.append(ConversationTurn.from_payload(turn))

        excluded_keys = {"id", "turns", "category", "difficulty"}
        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            turns=turns,
            category=_pick_first_string(payload, ("category",)),
            difficulty=_pick_first_string(payload, ("difficulty",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class RAGCase:
    case_id: str
    input_text: str
    context: str
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RAGCase":
        input_text = _pick_first_string(payload, ("question", "input", "prompt"))
        if input_text is None:
            raise ValueError("RAGCase requires one of: question, input, prompt")

        context = _pick_first_string(payload, ("context",))
        if context is None:
            raise ValueError("RAGCase requires: context")

        excluded_keys = {"id", "question", "input", "prompt", "context", "category"}
        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            input_text=input_text,
            context=context,
            category=_pick_first_string(payload, ("category",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class EdgeCase:
    case_id: str
    input_text: str
    category: Optional[str] = None
    instruction: Optional[str] = None
    expected_behavior: Optional[str] = None
    expected_output: Optional[str] = None
    severity: Optional[str] = None
    difficulty: Optional[str] = None
    test_type: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EdgeCase":
        input_text = _pick_first_string(payload, ("question", "input", "prompt"))
        if input_text is None:
            raise ValueError("EdgeCase requires one of: question, input, prompt")

        metrics = payload.get("metrics", [])
        if not isinstance(metrics, list):
            metrics = []

        excluded_keys = {
            "id",
            "question",
            "input",
            "prompt",
            "category",
            "instruction",
            "expected_behavior",
            "expected_answer",
            "expected_output",
            "severity",
            "difficulty",
            "test_type",
            "metrics",
        }

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            input_text=input_text,
            category=_pick_first_string(payload, ("category",)),
            instruction=_pick_first_string(payload, ("instruction",)),
            expected_behavior=_pick_first_string(payload, ("expected_behavior",)),
            expected_output=_pick_first_string(payload, ("expected_answer", "expected_output")),
            severity=_pick_first_string(payload, ("severity",)),
            difficulty=_pick_first_string(payload, ("difficulty",)),
            test_type=_pick_first_string(payload, ("test_type",)),
            metrics=[item for item in metrics if isinstance(item, str)],
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class PIIDetectionCase:
    case_id: str
    input_text: str
    expected_output: str
    pii_present: bool = False
    pii_types: List[str] = field(default_factory=list)
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PIIDetectionCase":
        input_text = _pick_first_string(payload, ("input", "question", "prompt"))
        if input_text is None:
            raise ValueError("PIIDetectionCase requires one of: input, question, prompt")

        expected_output = _pick_first_string(payload, ("expected_output",))
        if expected_output is None:
            raise ValueError("PIIDetectionCase requires: expected_output")

        pii_types = payload.get("pii_types", [])
        if not isinstance(pii_types, list):
            pii_types = []

        excluded_keys = {
            "id",
            "input",
            "question",
            "prompt",
            "expected_output",
            "pii_present",
            "pii_types",
            "category",
        }

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            input_text=input_text,
            expected_output=expected_output,
            pii_present=bool(payload.get("pii_present", False)),
            pii_types=[item for item in pii_types if isinstance(item, str)],
            category=_pick_first_string(payload, ("category",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class ConsistencyCase:
    case_id: str
    question: str
    category: Optional[str] = None
    complexity: Optional[str] = None
    expected_behavior: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_text(self) -> str:
        return self.question

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @property
    def resolved_complexity(self) -> str:
        return self.complexity or "unknown"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ConsistencyCase":
        question = _pick_first_string(payload, ("question", "prompt", "input"))
        if question is None:
            raise ValueError(
                "ConsistencyCase requires one of: question, prompt, input"
            )

        excluded_keys = {
            "id",
            "question",
            "prompt",
            "input",
            "category",
            "complexity",
            "expected_behavior",
        }

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            question=question,
            category=_pick_first_string(payload, ("category",)),
            complexity=_pick_first_string(payload, ("complexity",)),
            expected_behavior=_pick_first_string(payload, ("expected_behavior",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class PromptCompressionCase:
    case_id: str
    original_prompt: str
    compressed_prompts: Dict[str, str] = field(default_factory=dict)
    question_type: Optional[str] = None
    category: Optional[str] = None
    complexity: Optional[str] = None
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @property
    def resolved_complexity(self) -> str:
        return self.complexity or "unknown"

    @property
    def resolved_question_type(self) -> str:
        return self.question_type or "qa"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PromptCompressionCase":
        original_prompt = _pick_first_string(payload, ("original_prompt", "prompt", "input", "question"))
        if original_prompt is None:
            raise ValueError(
                "PromptCompressionCase requires one of: original_prompt, prompt, input, question"
            )

        compressed_prompts = {
            "75%": _pick_first_string(payload, ("compressed_75",)),
            "50%": _pick_first_string(payload, ("compressed_50",)),
            "25%": _pick_first_string(payload, ("compressed_25",)),
        }
        compressed_prompts = {
            label: prompt
            for label, prompt in compressed_prompts.items()
            if prompt is not None
        }
        if not compressed_prompts:
            raise ValueError(
                "PromptCompressionCase requires at least one compressed prompt variant"
            )

        excluded_keys = {
            "id",
            "original_prompt",
            "prompt",
            "input",
            "question",
            "compressed_75",
            "compressed_50",
            "compressed_25",
            "question_type",
            "category",
            "complexity",
            "expected_answer",
        }

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            original_prompt=original_prompt,
            compressed_prompts=compressed_prompts,
            question_type=_pick_first_string(payload, ("question_type",)),
            category=_pick_first_string(payload, ("category",)),
            complexity=_pick_first_string(payload, ("complexity",)),
            expected_answer=_pick_first_string(payload, ("expected_answer",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class NegativeConstraintCase:
    case_id: str
    prompt: str
    constraint_type: str
    constraint_params: Dict[str, Any] = field(default_factory=dict)
    expected_violation: bool = False
    category: Optional[str] = None
    complexity: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @property
    def resolved_complexity(self) -> str:
        return self.complexity or "unknown"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NegativeConstraintCase":
        prompt = _pick_first_string(payload, ("prompt", "input", "question"))
        if prompt is None:
            raise ValueError(
                "NegativeConstraintCase requires one of: prompt, input, question"
            )

        constraint_type = _pick_first_string(payload, ("constraint_type",))
        if constraint_type is None:
            raise ValueError("NegativeConstraintCase requires: constraint_type")

        constraint_params = payload.get("constraint_params", {})
        if not isinstance(constraint_params, dict):
            constraint_params = {}

        excluded_keys = {
            "id",
            "prompt",
            "input",
            "question",
            "constraint_type",
            "constraint_params",
            "expected_violation",
            "category",
            "complexity",
        }

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            prompt=prompt,
            constraint_type=constraint_type,
            constraint_params=constraint_params,
            expected_violation=bool(payload.get("expected_violation", False)),
            category=_pick_first_string(payload, ("category",)),
            complexity=_pick_first_string(payload, ("complexity",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class LanguageMixCase:
    case_id: str
    prompt: str
    expected_languages: List[str] = field(default_factory=list)
    mix_type: str = "code_switch"
    expected_response_language: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @property
    def resolved_difficulty(self) -> str:
        return self.difficulty or "medium"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LanguageMixCase":
        prompt = _pick_first_string(payload, ("prompt", "input", "question"))
        if prompt is None:
            raise ValueError(
                "LanguageMixCase requires one of: prompt, input, question"
            )

        expected_languages = payload.get("expected_languages", [])
        if not isinstance(expected_languages, list):
            expected_languages = []

        excluded_keys = {
            "id",
            "prompt",
            "input",
            "question",
            "expected_languages",
            "mix_type",
            "expected_response_language",
            "category",
            "difficulty",
        }

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            prompt=prompt,
            expected_languages=[lang for lang in expected_languages if isinstance(lang, str)],
            mix_type=_pick_first_string(payload, ("mix_type",)) or "code_switch",
            expected_response_language=_pick_first_string(payload, ("expected_response_language",)),
            category=_pick_first_string(payload, ("category",)),
            difficulty=_pick_first_string(payload, ("difficulty",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    prompt: str
    choices: List[Any] = field(default_factory=list)
    answer: Any = None
    label: Any = None
    best_answer: Optional[str] = None
    test_code: Optional[str] = None
    entry_point: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_text(self) -> str:
        return self.prompt

    @property
    def resolved_choices(self) -> List[Any]:
        return list(self.choices)

    @property
    def correct_index(self) -> Optional[int]:
        if isinstance(self.answer, int):
            return self.answer
        if isinstance(self.label, int):
            return self.label
        return None

    @property
    def reference_answer(self) -> Optional[str]:
        if self.best_answer:
            return self.best_answer
        if isinstance(self.answer, str):
            stripped = self.answer.strip()
            if stripped:
                return stripped
        return None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BenchmarkCase":
        prompt = _pick_first_string(payload, ("question", "prompt", "input", "ctx"))
        if prompt is None:
            raise ValueError(
                "BenchmarkCase requires one of: question, prompt, input, ctx"
            )

        choices = payload.get("choices") or payload.get("endings") or payload.get("options") or []
        if not isinstance(choices, list):
            choices = []

        answer = payload.get("answer")
        label = payload.get("label")

        mc1_targets = payload.get("mc1_targets")
        if isinstance(mc1_targets, Mapping):
            nested_choices = mc1_targets.get("choices")
            if isinstance(nested_choices, list) and not choices:
                choices = nested_choices
            if label is None:
                labels = mc1_targets.get("labels")
                if isinstance(labels, list) and labels:
                    label = labels[0]

        excluded_keys = {
            "id",
            "question",
            "prompt",
            "input",
            "ctx",
            "choices",
            "endings",
            "options",
            "answer",
            "label",
            "best_answer",
            "test",
            "tests",
            "unit_tests",
            "entry_point",
            "mc1_targets",
        }

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            prompt=prompt,
            choices=list(choices),
            answer=answer,
            label=label,
            best_answer=_pick_first_string(payload, ("best_answer",)),
            test_code=_pick_first_string(payload, ("test", "tests", "unit_tests")),
            entry_point=_pick_first_string(payload, ("entry_point",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class AdversarialCase:
    case_id: str
    input_text: str
    attack_type: str = "unknown"
    severity: str = "medium"
    expected_behavior: str = "Refuse unsafe or manipulative instructions"
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_category(self) -> str:
        return self.category or "general"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AdversarialCase":
        input_text = _pick_first_string(payload, ("attack_prompt", "question", "prompt"))
        if input_text is None:
            raise ValueError(
                "AdversarialCase requires one of: attack_prompt, question, prompt"
            )

        excluded_keys = {
            "id",
            "attack_prompt",
            "question",
            "prompt",
            "attack_type",
            "severity",
            "expected_behavior",
            "category",
            "difficulty",
        }

        return cls(
            case_id=_pick_first_string(payload, ("id",)) or "unknown",
            input_text=input_text,
            attack_type=_pick_first_string(payload, ("attack_type",)) or "unknown",
            severity=_pick_first_string(payload, ("severity",)) or "medium",
            expected_behavior=(
                _pick_first_string(payload, ("expected_behavior",))
                or "Refuse unsafe or manipulative instructions"
            ),
            category=_pick_first_string(payload, ("category",)),
            difficulty=_pick_first_string(payload, ("difficulty",)),
            metadata=_build_metadata(payload, excluded_keys),
            raw_payload=dict(payload),
        )