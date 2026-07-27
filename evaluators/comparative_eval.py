"""
Comparative Evaluator
İki modelin yanıtlarını karşılaştırır ve hangisinin daha iyi olduğunu değerlendirir
"""
import json
from typing import Dict, Any, List, Tuple
from adapters.unified_adapter import UnifiedLLMAdapter


class ComparativeEvaluator:
    """Compare two model responses and determine which is better"""
    
    COMPARISON_PROMPT = """
Aşağıdaki soru için iki farklı modelin verdiği yanıtları karşılaştırın.

Soru: {question}

Model A Yanıtı:
{response_a}

Model B Yanıtı:
{response_b}

Hangisi daha iyi? Aşağıdaki kriterlere göre değerlendirin:
1. Doğruluk (accuracy)
2. Eksiksizlik (completeness)
3. Açıklık (clarity)
4. Alakalılık (relevance)

Sonuç olarak:
- "A" (Model A daha iyi)
- "B" (Model B daha iyi)
- "Tie" (Eşit)

JSON formatında yanıt verin:
{{"winner": "<A/B/Tie>", "reasoning": "<detaylı açıklama>", "score_difference": <1-10 arası fark>}}
"""

    def __init__(self, judge_adapter: UnifiedLLMAdapter):
        self.judge = judge_adapter
    
    def compare(
        self,
        question: str,
        response_a: str,
        response_b: str,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, Any]:
        """
        Compare two responses and determine which is better
        
        Args:
            question: The question asked
            response_a: First model's response
            response_b: Second model's response
            model_a_name: Name of first model
            model_b_name: Name of second model
        
        Returns:
            {
                "winner": str ("A", "B", or "Tie"),
                "reasoning": str,
                "score_difference": float,
                "model_a_name": str,
                "model_b_name": str
            }
        """
        prompt = self.COMPARISON_PROMPT.format(
            question=question,
            response_a=response_a,
            response_b=response_b
        )
        
        messages = [
            {"role": "system", "content": "Sen objektif bir karşılaştırma uzmanısın. Model yanıtlarını tarafsız şekilde değerlendirirsin."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.judge.generate(messages, temperature=0.0, max_tokens=500)
        
        try:
            content = result['content'].strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(content)
            
            return {
                "winner": evaluation.get("winner", "Tie"),
                "reasoning": evaluation.get("reasoning", ""),
                "score_difference": evaluation.get("score_difference", 0) / 10.0,
                "model_a_name": model_a_name,
                "model_b_name": model_b_name,
                "raw_response": result['content']
            }
        except Exception as e:
            return {
                "winner": "Tie",
                "reasoning": f"Parse error: {str(e)}",
                "score_difference": 0,
                "model_a_name": model_a_name,
                "model_b_name": model_b_name,
                "raw_response": result['content']
            }
    
    def compare_with_swap(
        self,
        question: str,
        response_a: str,
        response_b: str,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, Any]:
        """Position-bias-mitigated comparison: judge twice with A/B swapped.

        LLM judges favor whichever answer appears first. The verdict only
        stands if it survives swapping the presentation order; otherwise the
        match is scored as a Tie and flagged position_consistent=False.

        Returns the same schema as compare() plus:
            position_consistent: bool
            first_pass_winner / second_pass_winner: str (both in original A/B frame)
        """
        first = self.compare(question, response_a, response_b, model_a_name, model_b_name)
        swapped = self.compare(question, response_b, response_a, model_b_name, model_a_name)

        # Map the swapped verdict back into the original A/B frame
        swap_map = {"A": "B", "B": "A", "Tie": "Tie"}
        second_winner = swap_map.get(swapped["winner"], "Tie")

        consistent = first["winner"] == second_winner
        if consistent:
            winner = first["winner"]
            reasoning = first["reasoning"]
            score_difference = (first["score_difference"] + swapped["score_difference"]) / 2.0
        else:
            winner = "Tie"
            reasoning = (
                f"Pozisyon tutarsızlığı: sunum sırası değişince judge kararı değişti "
                f"({first['winner']} → {second_winner}); Tie sayıldı."
            )
            score_difference = 0.0

        return {
            "winner": winner,
            "reasoning": reasoning,
            "score_difference": score_difference,
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "position_consistent": consistent,
            "first_pass_winner": first["winner"],
            "second_pass_winner": second_winner,
            "raw_response": first.get("raw_response"),
        }

    def batch_compare(
        self,
        comparisons: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """
        Compare multiple question-response pairs
        
        Args:
            comparisons: List of (question, response_a, response_b) tuples
        
        Returns:
            {
                "results": List[Dict],
                "summary": {
                    "a_wins": int,
                    "b_wins": int,
                    "ties": int,
                    "win_rate_a": float,
                    "win_rate_b": float
                }
            }
        """
        results = []
        a_wins = 0
        b_wins = 0
        ties = 0
        
        for question, response_a, response_b in comparisons:
            result = self.compare(question, response_a, response_b)
            results.append(result)
            
            if result["winner"] == "A":
                a_wins += 1
            elif result["winner"] == "B":
                b_wins += 1
            else:
                ties += 1
        
        total = len(comparisons)
        
        return {
            "results": results,
            "summary": {
                "total_comparisons": total,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "win_rate_a": a_wins / total if total > 0 else 0,
                "win_rate_b": b_wins / total if total > 0 else 0,
                "tie_rate": ties / total if total > 0 else 0
            }
        }
