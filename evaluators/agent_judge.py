"""
Agent Judge Evaluator — replaces azure_agent.py.
LLM-as-judge for task adherence, tool call accuracy, response completeness, intent resolution.
No Azure dependency. Uses UnifiedLLMAdapter (same judge model as pipeline).
Output schema identical to AzureAgentEvaluator for backward compatibility.
"""
import json
from typing import Dict, Any, List, Optional, Union
from adapters.unified_adapter import UnifiedLLMAdapter
from utils.logger import get_logger

logger = get_logger(__name__)


def _flatten_messages(query: Union[str, List[dict]], response: Union[str, List[dict]]) -> List[dict]:
    """Build flat message list from query + response, same logic as AzureAgentEvaluator."""
    messages: List[dict] = []

    if isinstance(query, list):
        for msg in query:
            if isinstance(msg, dict) and msg.get("content"):
                messages.append(msg)
    elif isinstance(query, str) and query.strip():
        messages.append({"role": "user", "content": query.strip()})

    if isinstance(response, list):
        for msg in response:
            if isinstance(msg, dict):
                content = msg.get("content")
                if content and (isinstance(content, list) or (isinstance(content, str) and content.strip())):
                    messages.append(msg)
    elif isinstance(response, str) and response.strip():
        messages.append({"role": "assistant", "content": response.strip()})

    return messages


def _extract_query_str(query: Union[str, List[dict]]) -> str:
    if isinstance(query, str):
        return query.strip()
    for msg in (query or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _extract_final_response(response: Union[str, List[dict]]) -> str:
    if isinstance(response, str):
        return response.strip()
    last_text = ""
    for msg in (response or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                last_text = content.strip()
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        last_text = block.get("text", "").strip()
    return last_text


def _render_conversation(messages: List[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") in ("tool_call", "tool_use"):
                        tc = block.get("tool_call") or block
                        parts.append(f"[TOOL_CALL: {tc.get('name', '?')}({tc.get('arguments', '')})]")
                    elif block.get("type") == "tool_result":
                        parts.append(f"[TOOL_RESULT: {block.get('tool_result', '')}]")
            content = " ".join(parts)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class AgentJudgeEvaluator:
    """LLM-as-judge replacement for Azure AI agent evaluators.

    Evaluates agentic conversations across 4 dimensions:
    - task_adherence: Did the agent follow its instructions and complete the task?
    - tool_call_accuracy: Were tool calls chosen and parameterized correctly?
    - response_completeness: Does the final response address all parts of the query?
    - intent_resolution: Was the user's original intent resolved?

    Output schema matches AzureAgentEvaluator exactly.
    """

    _PROMPTS = {
        "task_adherence": """Aşağıdaki ajan konuşmasını değerlendir. Ajan verilen göreve sadık kaldı mı?

Konuşma:
{conversation}

Değerlendirme kriterleri:
- Ajan sistem talimatlarına uydu mu?
- Belirlenen görevi tamamladı mı?
- Görevden sapma var mı?

Puan skalası: 1 (hiç uymadı) → 5 (tam uyum)

Sadece JSON formatında yanıt ver:
{{"score": <1-5>, "reasoning": "<kısa gerekçe>", "result": "adherent" veya "non_adherent"}}""",

        "tool_call_accuracy": """Aşağıdaki ajan konuşmasında yapılan araç çağrılarını değerlendir.

Kullanıcı isteği: {query}

Konuşma:
{conversation}

{tool_definitions_section}

Değerlendirme kriterleri:
- Doğru araçlar seçildi mi?
- Parametreler doğru ve eksiksiz mi?
- Gereksiz veya hatalı araç çağrısı var mı?
- Araç çıktıları doğru yorumlandı mı?

Puan skalası: 1 (tamamen yanlış) → 5 (mükemmel doğruluk)

Sadece JSON formatında yanıt ver:
{{"score": <1-5>, "reasoning": "<kısa gerekçe>", "result": "accurate" veya "inaccurate"}}""",

        "response_completeness": """Aşağıdaki ajan konuşmasının son yanıtını değerlendir. Yanıt kullanıcının sorusunu tam olarak karşılıyor mu?

Kullanıcı isteği: {query}

Son ajan yanıtı:
{final_response}

Değerlendirme kriterleri:
- Kullanıcının tüm soruları yanıtlandı mı?
- Önemli bir bilgi eksik mi?
- Yanıt yeterince kapsamlı mı?

Puan skalası: 1 (çok eksik) → 5 (tam ve eksiksiz)

Sadece JSON formatında yanıt ver:
{{"score": <1-5>, "reasoning": "<kısa gerekçe>", "result": "complete" veya "incomplete"}}""",

        "intent_resolution": """Aşağıdaki ajan konuşmasını değerlendir. Kullanıcının asıl amacı çözüldü mü?

Konuşma:
{conversation}

Değerlendirme kriterleri:
- Kullanıcının temel amacı anlaşıldı mı?
- Bu amaç başarıyla karşılandı mı?
- Kullanıcı tatmin olmuş olur muydu?

Puan skalası: 1 (amaç hiç çözülmedi) → 5 (tam çözüm)

Sadece JSON formatında yanıt ver:
{{"score": <1-5>, "reasoning": "<kısa gerekçe>", "result": "resolved" veya "unresolved"}}""",
    }

    def __init__(self, judge_adapter: UnifiedLLMAdapter):
        self.judge = judge_adapter

    def _run(self, prompt: str, metric: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "Sen bir ajan değerlendirme uzmanısın. Verilen kriterlere göre ajan konuşmalarını objektif şekilde puanla. Sadece JSON formatında yanıt ver."},
            {"role": "user", "content": prompt},
        ]
        result = self.judge.generate(messages)
        content = (result.get("content") or "").strip()

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            parsed = json.loads(content)
            raw_score = float(parsed.get("score", 0))
            score = max(0.0, min(5.0, raw_score))
            normalized = round(score / 5.0, 4)
            reasoning = parsed.get("reasoning", "")
            res = parsed.get("result", "")
            logger.debug(f"[agent_judge] {metric} → score={score:.1f} result={res}")
            return {"score": normalized, "reasoning": reasoning, "result": res, "raw": parsed}
        except Exception as e:
            logger.warning(f"[agent_judge] {metric} parse failed: {e} | raw: {content[:200]!r}")
            return {"score": 0.0, "reasoning": "parse error", "result": "error", "raw": {}}

    def evaluate_task_adherence(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> Dict[str, Any]:
        messages = _flatten_messages(query, response)
        conversation = _render_conversation(messages)
        prompt = self._PROMPTS["task_adherence"].format(conversation=conversation)
        return self._run(prompt, "task_adherence")

    def evaluate_tool_call_accuracy(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        query_str = _extract_query_str(query)
        messages = _flatten_messages([], response)
        conversation = _render_conversation(messages)
        if tool_definitions:
            tool_section = "Tanımlı araçlar:\n" + json.dumps(tool_definitions, ensure_ascii=False, indent=2)
        else:
            tool_section = ""
        prompt = self._PROMPTS["tool_call_accuracy"].format(
            query=query_str or "N/A",
            conversation=conversation,
            tool_definitions_section=tool_section,
        )
        return self._run(prompt, "tool_call_accuracy")

    def evaluate_response_completeness(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> Dict[str, Any]:
        query_str = _extract_query_str(query)
        final_response = _extract_final_response(response)
        prompt = self._PROMPTS["response_completeness"].format(
            query=query_str or "N/A",
            final_response=final_response or "(boş yanıt)",
        )
        return self._run(prompt, "response_completeness")

    def evaluate_intent_resolution(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> Dict[str, Any]:
        messages = _flatten_messages(query, response)
        conversation = _render_conversation(messages)
        prompt = self._PROMPTS["intent_resolution"].format(conversation=conversation)
        return self._run(prompt, "intent_resolution")

    def evaluate_all(
        self,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run all 4 agent evaluations. Same return schema as AzureAgentEvaluator.evaluate_all()."""
        task = self.evaluate_task_adherence(query, response)
        tool = self.evaluate_tool_call_accuracy(query, response, tool_definitions=tool_definitions)
        completeness = self.evaluate_response_completeness(query, response)
        intent = self.evaluate_intent_resolution(query, response)

        valid_scores = [r["score"] for r in (task, tool, completeness, intent) if r["score"] > 0]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        logger.info(f"[agent_judge] evaluate_all → task={task['score']:.2f} tool={tool['score']:.2f} completeness={completeness['score']:.2f} intent={intent['score']:.2f} avg={avg:.2f}")

        return {
            "task_adherence": task,
            "tool_call_accuracy": tool,
            "response_completeness": completeness,
            "intent_resolution": intent,
            "aggregate_score": round(avg, 4),
        }

    def evaluate_simple(
        self,
        query: str,
        response: str,
    ) -> Dict[str, Any]:
        """Simplified evaluation for non-agent query/response pairs (no tool calls)."""
        completeness = self.evaluate_response_completeness(query, response)
        intent = self.evaluate_intent_resolution(query, response)

        valid_scores = [r["score"] for r in (completeness, intent) if r["score"] > 0]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return {
            "response_completeness": completeness,
            "intent_resolution": intent,
            "aggregate_score": round(avg, 4),
        }


def is_agent_eval_available() -> bool:
    """Always True — no external credentials needed."""
    return True
