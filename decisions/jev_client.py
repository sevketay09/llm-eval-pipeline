"""TypeSafe AI (Jev) decision client.

SDK sözleşmesi (typesafe-sdk==0.7.0, doğrulandı 21.09.2026, `.venv314`):

- `TypeSafeClient(*, api_key=None, model=None, retry=RetryPolicy|None, timeout=None,
  headers=None, transport=None, http_client=None, base_url=None)`. `api_key` verilmez
  ve `TYPESAFE_API_KEY` env'i de boşsa **construction anında** `TypeSafeError` fırlatır
  — bu yüzden gerçek client burada, `__init__`'te değil, ilk `decide()` çağrısında lazy
  kurulur (key henüz yoksa liste/capabilities endpoint'leri patlamasın diye).
- `client.system_one(state, questions, *, model=None, retry=None, timeout=None, ...)
  -> SystemOneResponse`. `questions`: `{name: Noul|Choice|Score}`.
- `Noul(instructions=str, criteria=None)` → cevap `NoulAnswer(type, noul: float)`.
  **`confidence` ve `probabilities` YOK** — biz `max(p, 1-p)` ile hesaplıyoruz.
- `Choice(instructions=str, criteria={label: desc|None})` → cevap
  `ChoiceAnswer(type, choice: str, confidence: float, probabilities: dict[str,float])`.
- `Score(instructions=str, criteria=Sequence[desc])` — **sıralı açıklama listesi**,
  isim yok (index = seviye). Cevap `ScoreAnswer(type, score: float, confidence: float,
  legend: dict[int,...], probabilities: dict[int,float])`. `score`, sıralı seviyeler
  üzerindeki **beklenen değer** (0..len(criteria)-1 arası, sürekli); SDK bunu bizim
  yerimize hesaplıyor, kendi ağırlıklı ortalamamızı almamıza gerek yok.
- `SystemOneResponse.answers: dict[str, NoulAnswer|ChoiceAnswer|ScoreAnswer]`,
  `.usage: Usage(input_tokens: int|None, output_tokens: int|None)` (**gerçek** token
  sayısı, API raporlamazsa None), `.model: str`.
- Tek taban exception: `TypeSafeError` (her şey — auth, rate limit, bağlantı, timeout,
  validation — bundan türüyor). Biz bunu yakalayıp `DecisionError`'a çeviriyoruz.
- SDK'nın **kendi retry politikası var** (`RetryPolicy`: backoff + jitter + Retry-After
  header desteği). Bu yüzden `JevDecisionClient.handles_own_retries = True`;
  `BaseDecisionClient.decide()` üstüne ikinci bir retry katmanı eklemiyor.
- Env: `TYPESAFE_API_KEY`. `TYPESAFE_DEFAULT_MODEL` de var ama biz model'i her zaman
  config'ten açıkça veriyoruz, bu env'e güvenmiyoruz.

Beklenenden farklı çıkan tek nokta: SDK'da `Score` **var** (plandaki Choice-fallback
adaptasyonu gereksiz kaldı, kullanılmadı).
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict

from decisions.client import BaseDecisionClient
from decisions.types import ChoiceQ, Decision, DecisionError, DecisionResponse, NoulQ, Question, ScoreQ


class JevDecisionClient(BaseDecisionClient):
    handles_own_retries = True

    def __init__(self, model_cfg: Dict[str, Any], model_key: str) -> None:
        model_name = model_cfg.get("model_name") or "jev-latest"
        super().__init__(
            model_key,
            model_name,
            cost_per_mtok_input=float(model_cfg.get("cost_per_mtok_input", 0.042)),
        )
        self._api_key = model_cfg.get("api_key")
        self._timeout_s = float(model_cfg.get("timeout_s", 10))
        self._sdk_client: Any = None

    def _ensure_client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client

        try:
            import typesafe_sdk
        except ImportError as exc:
            raise DecisionError(
                "typesafe-sdk is not installed. "
                "pip install typesafe-sdk (or pip install -r requirements-jev.txt)"
            ) from exc

        api_key = self._api_key
        if not api_key or (isinstance(api_key, str) and api_key.startswith("${")):
            api_key = None
        if api_key is None and not typesafe_sdk_env_key_present():
            raise DecisionError(f"TYPESAFE_API_KEY is not set for model '{self.model_key}'")

        try:
            self._sdk_client = typesafe_sdk.TypeSafeClient(
                api_key=api_key,
                model=self.model_name,
                timeout=self._timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 — SDK's own TypeSafeError on bad/missing key
            raise DecisionError(str(exc)) from exc
        return self._sdk_client

    def _decide_raw(self, state: str, questions: Dict[str, Question]) -> DecisionResponse:
        client = self._ensure_client()

        import typesafe_sdk

        sdk_questions: Dict[str, Any] = {}
        for name, q in questions.items():
            if isinstance(q, NoulQ):
                sdk_questions[name] = typesafe_sdk.Noul(instructions=q.instructions)
            elif isinstance(q, ChoiceQ):
                sdk_questions[name] = typesafe_sdk.Choice(
                    instructions=q.instructions, criteria=dict(q.criteria)
                )
            elif isinstance(q, ScoreQ):
                sdk_questions[name] = typesafe_sdk.Score(
                    instructions=q.instructions,
                    criteria=[desc for _, desc in q.levels],
                )
            else:
                raise DecisionError(f"Unsupported question type for '{name}': {type(q).__name__}")

        start = time.perf_counter()
        try:
            response = client.system_one(state=state, questions=sdk_questions)
        except Exception as exc:  # noqa: BLE001 — typesafe_sdk.TypeSafeError and subclasses
            raise DecisionError(str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        answers: Dict[str, Decision] = {}
        for name, answer in response.answers.items():
            answers[name] = _to_decision(answer, questions.get(name))

        usage = response.usage
        if usage is not None and usage.input_tokens is not None:
            est_input_tokens = usage.input_tokens
        else:
            est_input_tokens = _estimate_input_tokens(state, questions)

        return DecisionResponse(
            answers=answers,
            latency_ms=latency_ms,
            est_input_tokens=est_input_tokens,
            model=response.model or self.model_name,
        )


def typesafe_sdk_env_key_present() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY", "").strip())


def _to_decision(answer: Any, question: "Question | None") -> Decision:
    if answer.type == "noul":
        p = float(answer.noul)
        return Decision(
            kind="noul",
            value=p,
            confidence=max(p, 1 - p),
            probabilities={"true": p, "false": 1 - p},
        )
    if answer.type == "choice":
        return Decision(
            kind="choice",
            value=answer.choice,
            confidence=float(answer.confidence),
            probabilities=dict(answer.probabilities),
        )
    if answer.type == "score":
        levels = question.levels if isinstance(question, ScoreQ) else []
        n = len(levels)
        normalized = float(answer.score) / (n - 1) if n > 1 else float(answer.score)
        normalized = max(0.0, min(1.0, normalized))
        probabilities = {}
        for idx, p in answer.probabilities.items():
            name = levels[idx][0] if 0 <= idx < n else str(idx)
            probabilities[name] = p
        return Decision(
            kind="score",
            value=normalized,
            confidence=float(answer.confidence),
            probabilities=probabilities,
        )
    raise DecisionError(f"Unknown answer type from Jev: {answer.type!r}")


def _estimate_input_tokens(state: str, questions: Dict[str, Question]) -> int:
    total_chars = len(state)
    for q in questions.values():
        total_chars += len(q.instructions)
        if isinstance(q, ChoiceQ):
            total_chars += sum(len(v) for v in q.criteria.values())
        elif isinstance(q, ScoreQ):
            total_chars += sum(len(desc) for _, desc in q.levels)
    return max(1, total_chars // 4)
