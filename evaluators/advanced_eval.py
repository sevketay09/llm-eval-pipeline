"""
Advanced Evaluators
Chain-of-Thought, RAG, Instruction Following evaluators
"""
import json
import re
from typing import Dict, Any, List, Optional
from adapters.unified_adapter import UnifiedLLMAdapter


class ChainOfThoughtEvaluator:
    """Evaluate reasoning quality in chain-of-thought responses"""
    
    COT_PROMPT = """
Aşağıdaki problemin çözümünde kullanılan adım adım muhakemeyi değerlendirin.

Problem: {question}

Çözüm (Adım Adım Muhakeme):
{answer}

Değerlendirme kriterleri:
1. Her adım mantıklı mı?
2. Adımlar birbirini takip ediyor mu?
3. Sonuca doğru şekilde ulaşılmış mı?
4. Gereksiz veya yanlış adımlar var mı?

Muhakeme kalitesini puanlayın (1-10):
- 1-3: Mantık hataları, adımlar birbirini tutmuyor
- 4-6: Temel muhakeme var ama eksik veya hatalı
- 7-8: İyi muhakeme, adımlar mantıklı
- 9-10: Mükemmel muhakeme, tüm adımlar doğru ve net

JSON formatında yanıt verin:
{{"score": <1-10>, "reasoning": "<açıklama>", "logical_errors": ["<tespit edilen mantık hataları>"], "step_count": <adım sayısı>}}
"""

    def __init__(self, judge_adapter: UnifiedLLMAdapter):
        self.judge = judge_adapter
    
    def evaluate(
        self,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate chain-of-thought reasoning quality
        
        Returns:
            {
                "score": float (0-1),
                "reasoning": str,
                "logical_errors": List[str],
                "step_count": int,
                "has_steps": bool
            }
        """
        prompt = self.COT_PROMPT.format(
            question=question,
            answer=answer
        )
        
        messages = [
            {"role": "system", "content": "Sen bir mantıksal muhakeme uzmanısın."},
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
            
            # Detect if answer has step-by-step structure
            has_steps = bool(re.search(r'(adım|step|\d+\.|\d+\))', answer.lower()))
            
            return {
                "score": score,
                "reasoning": evaluation.get("reasoning", ""),
                "logical_errors": evaluation.get("logical_errors", []),
                "step_count": evaluation.get("step_count", 0),
                "has_steps": has_steps,
                "raw_response": result['content']
            }
        except Exception as e:
            return {
                "score": 0.5,
                "reasoning": f"Parse error: {str(e)}",
                "logical_errors": [],
                "step_count": 0,
                "has_steps": False,
                "raw_response": result['content']
            }


class RAGEvaluator:
    """Evaluate Retrieval-Augmented Generation responses"""
    
    RAG_PROMPT = """
Aşağıdaki soru, verilen context ve modelin yanıtını değerlendirin.

Soru: {question}

Verilen Context:
{context}

Model Yanıtı:
{answer}

Değerlendirme kriterleri:
1. Yanıt context'e dayanıyor mu?
2. Context dışı bilgi eklemiş mi?
3. Context'teki bilgiyi doğru kullanmış mı?
4. Soruya context bazlı cevap vermiş mi?

RAG kalitesini puanlayın (1-10):
- 1-3: Context'i görmezden gelmiş, alakasız bilgi
- 4-6: Kısmen context kullanmış ama context dışı ekleme yapmış
- 7-8: Çoğunlukla context'e sadık kalmış
- 9-10: Tamamen context'e dayalı, doğru bilgi kullanımı

JSON formatında yanıt verin:
{{"score": <1-10>, "reasoning": "<açıklama>", "context_adherence": <0-1>, "added_info": ["<context dışı eklenen bilgiler>"]}}
"""

    def __init__(self, judge_adapter: UnifiedLLMAdapter):
        self.judge = judge_adapter
    
    def evaluate(
        self,
        question: str,
        context: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate RAG response quality
        
        Returns:
            {
                "score": float (0-1),
                "reasoning": str,
                "context_adherence": float,
                "added_info": List[str],
                "is_grounded": bool
            }
        """
        prompt = self.RAG_PROMPT.format(
            question=question,
            context=context,
            answer=answer
        )
        
        messages = [
            {"role": "system", "content": "Sen bir RAG sistemi değerlendirme uzmanısın."},
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
            context_adherence = evaluation.get("context_adherence", score)
            
            return {
                "score": score,
                "reasoning": evaluation.get("reasoning", ""),
                "context_adherence": context_adherence,
                "added_info": evaluation.get("added_info", []),
                "is_grounded": score >= 0.7,
                "raw_response": result['content']
            }
        except Exception as e:
            return {
                "score": 0.5,
                "reasoning": f"Parse error: {str(e)}",
                "context_adherence": 0.5,
                "added_info": [],
                "is_grounded": False,
                "raw_response": result['content']
            }


class InstructionFollowingEvaluator:
    """Evaluate how well model follows specific instructions"""
    
    INSTRUCTION_PROMPT = """
Aşağıdaki talimat ve modelin yanıtını değerlendirin.

Talimat: {instruction}

Model Yanıtı:
{answer}

Model talimata uymuş mu? Kontrol edilecekler:
1. Format doğru mu? (JSON, liste, tablo vb.)
2. Uzunluk kısıtlamasına uymuş mu?
3. Dil kısıtlamasına uymuş mu?
4. Özel gereksinimlere uymuş mu?

Talimat uyumu puanı verin (1-10):
- 1-3: Talimatları görmezden gelmiş
- 4-6: Kısmen uymuş ama eksikler var
- 7-8: Çoğunlukla uymuş
- 9-10: Tüm talimatlara harfiyen uymuş

JSON formatında yanıt verin:
{{"score": <1-10>, "reasoning": "<açıklama>", "violations": ["<ihlal edilen talimatlar>"]}}
"""

    def __init__(self, judge_adapter: UnifiedLLMAdapter):
        self.judge = judge_adapter
    
    def evaluate(
        self,
        instruction: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate instruction following
        
        Returns:
            {
                "score": float (0-1),
                "reasoning": str,
                "violations": List[str],
                "follows_instructions": bool
            }
        """
        prompt = self.INSTRUCTION_PROMPT.format(
            instruction=instruction,
            answer=answer
        )
        
        messages = [
            {"role": "system", "content": "Sen bir talimat uyumu değerlendirme uzmanısın."},
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
                "violations": evaluation.get("violations", []),
                "follows_instructions": score >= 0.7,
                "raw_response": result['content']
            }
        except Exception as e:
            return {
                "score": 0.5,
                "reasoning": f"Parse error: {str(e)}",
                "violations": [],
                "follows_instructions": False,
                "raw_response": result['content']
            }
