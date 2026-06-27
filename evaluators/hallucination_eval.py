"""
Hallucination Detection Evaluator
Modelin uydurma/gerçek dışı bilgi üretip üretmediğini tespit eder
"""
import json
from typing import Dict, Any, Optional, List
from adapters.unified_adapter import UnifiedLLMAdapter


class HallucinationEvaluator:
    """Detect hallucinations in model outputs"""
    
    HALLUCINATION_PROMPT = """
Aşağıdaki soru, referans bilgi ve modelin verdiği cevabı değerlendirin.

Soru: {question}

Referans Bilgi (Güvenilir Kaynak): {reference}

Model Cevabı: {answer}

Modelin cevabında halüsinasyon (uydurma bilgi) var mı değerlendirin:
- 1-3: Ciddi halüsinasyonlar, referansa aykırı yanlış bilgiler
- 4-6: Bazı doğrulanmamış iddialar, referansla kısmen uyumsuz
- 7-8: Çoğunlukla doğru, küçük spekülatif ifadeler olabilir
- 9-10: Tamamen doğru ve referansla uyumlu, halüsinasyon yok

JSON formatında yanıt verin:
{{"score": <1-10>, "reasoning": "<açıklama>", "hallucinations": ["<tespit edilen halüsinasyonlar listesi>"]}}
"""

    FACTUAL_CONSISTENCY_PROMPT = """
Aşağıdaki iki metni karşılaştırın ve faktüel tutarlılığı değerlendirin.

Kaynak Metin: {source}

Üretilen Metin: {generated}

Üretilen metnin kaynak metne ne kadar sadık olduğunu değerlendirin:
- 1-3: Kaynakta olmayan birçok bilgi eklemiş
- 4-6: Bazı bilgiler kaynak dışı eklenmiş
- 7-8: Çoğunlukla kaynak metne sadık
- 9-10: Tamamen kaynak metinle tutarlı

JSON formatında yanıt verin:
{{"score": <1-10>, "reasoning": "<açıklama>", "added_info": ["<kaynak dışı eklenen bilgiler>"]}}
"""

    def __init__(self, judge_adapter: UnifiedLLMAdapter):
        self.judge = judge_adapter
    
    def check_hallucination(
        self,
        question: str,
        answer: str,
        reference: str
    ) -> Dict[str, Any]:
        """
        Check if model's answer contains hallucinations
        
        Args:
            question: The question asked
            answer: Model's answer
            reference: Reliable reference information
        
        Returns:
            {
                "score": float (0-1),
                "reasoning": str,
                "hallucinations": List[str],
                "has_hallucination": bool
            }
        """
        prompt = self.HALLUCINATION_PROMPT.format(
            question=question,
            reference=reference,
            answer=answer
        )
        
        messages = [
            {"role": "system", "content": "Sen bir fact-checking uzmanısın. Metinlerdeki halüsinasyonları tespit edersin."},
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
            score = evaluation.get("score", 0) / 10.0
            
            return {
                "score": score,
                "reasoning": evaluation.get("reasoning", ""),
                "hallucinations": evaluation.get("hallucinations", []),
                "has_hallucination": score < 0.7,
                "raw_response": result['content']
            }
        except Exception as e:
            return {
                "score": 0.5,
                "reasoning": f"Parse error: {str(e)}",
                "hallucinations": [],
                "has_hallucination": True,
                "raw_response": result['content']
            }
    
    def check_factual_consistency(
        self,
        source: str,
        generated: str
    ) -> Dict[str, Any]:
        """
        Check if generated text is consistent with source
        
        Args:
            source: Source/reference text
            generated: Generated text to check
        
        Returns:
            {
                "score": float (0-1),
                "reasoning": str,
                "added_info": List[str],
                "is_consistent": bool
            }
        """
        prompt = self.FACTUAL_CONSISTENCY_PROMPT.format(
            source=source,
            generated=generated
        )
        
        messages = [
            {"role": "system", "content": "Sen bir metin tutarlılığı analisti olarak çalışırsın."},
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
            score = evaluation.get("score", 0) / 10.0
            
            return {
                "score": score,
                "reasoning": evaluation.get("reasoning", ""),
                "added_info": evaluation.get("added_info", []),
                "is_consistent": score >= 0.7,
                "raw_response": result['content']
            }
        except Exception as e:
            return {
                "score": 0.5,
                "reasoning": f"Parse error: {str(e)}",
                "added_info": [],
                "is_consistent": False,
                "raw_response": result['content']
            }
