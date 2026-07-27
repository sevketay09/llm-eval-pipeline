"""
Language Mix Evaluator
Tests model's ability to handle Turkish-English mixed queries
"""
import re
from typing import Dict, Any, List, Tuple
from adapters.unified_adapter import UnifiedLLMAdapter


class LanguageMixEvaluator:
    """
    Evaluates model's handling of Turkish-English language mixing scenarios.
    
    Tests:
    - Code-switching (switching between languages mid-sentence)
    - Bilingual queries (questions in multiple languages)
    - Technical terms in one language, context in another
    - Response language consistency
    """
    
    def __init__(self, judge_adapter: UnifiedLLMAdapter = None):
        """
        Initialize evaluator.
        
        Args:
            judge_adapter: Optional judge model for quality assessment
        """
        self.judge_adapter = judge_adapter
        
        # Turkish language indicators
        self.turkish_patterns = [
            r'\b(ve|ile|için|olan|olarak|bir|bu|şu|nasıl|ne|nedir|midir|mıdır)\b',
            r'\b(değil|değildir|var|yok|gibi|kadar|daha|çok|az)\b',
            r'\b(ben|sen|o|biz|siz|onlar|benim|senin|onun|bizim)\b',
            r'[çğıöşü]',  # Turkish-specific characters
        ]
        
        # English language indicators
        self.english_patterns = [
            r'\b(and|or|the|is|are|was|were|been|have|has|had)\b',
            r'\b(what|where|when|who|why|how|which|that|this|these)\b',
            r'\b(can|could|should|would|will|shall|may|might|must)\b',
            r'\b(not|no|yes|but|because|if|then|than|very|more|most)\b',
        ]
        
        # Technical/domain terms that are commonly mixed
        self.technical_terms = [
            r'\b(API|REST|JSON|XML|HTTP|HTTPS|SQL|NoSQL|database|server)\b',
            r'\b(function|class|method|variable|parameter|return|exception)\b',
            r'\b(machine learning|deep learning|neural network|AI|model|training)\b',
            r'\b(blockchain|cryptocurrency|bitcoin|ethereum|smart contract)\b',
            r'\b(cloud|SaaS|PaaS|IaaS|microservice|container|kubernetes)\b',
            r'\b(IBAN|SWIFT|EFT|TCMB|BKM|kredi|hesap|bakiye|transfer)\b',
        ]
    
    def evaluate_language_mix(
        self,
        model: UnifiedLLMAdapter,
        prompt: str,
        expected_languages: List[str],
        mix_type: str,
        expected_response_language: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate model's handling of language-mixed prompt.
        
        Args:
            model: The LLM adapter to test
            prompt: Mixed-language prompt
            expected_languages: Languages present in prompt (e.g., ["tr", "en"])
            mix_type: Type of mixing (code_switch, bilingual, technical_mix, domain_specific)
            expected_response_language: Expected primary language of response
            
        Returns:
            Evaluation results with language analysis
        """
        # Get model response
        generated = model.generate([
            {"role": "user", "content": prompt}
        ])
        if generated.get("error"):
            # Infrastructure failure, not the model's answer — the caller
            # must check `generation_error` and exclude this item instead of
            # scoring an empty string as a language-mix response.
            return {
                "generation_error": generated["error"],
                "response": "",
            }
        response = generated.get("content") or ""
        usage = generated.get("usage", {})
        
        # Analyze prompt languages
        prompt_lang_analysis = self._analyze_text_languages(prompt)
        
        # Analyze response languages
        response_lang_analysis = self._analyze_text_languages(response)
        
        # Check if model understood the mixed query
        understood_mix = self._check_understanding(prompt, response, mix_type)
        
        # Check response language appropriateness
        response_lang_appropriate = self._check_response_language(
            response_lang_analysis,
            expected_response_language
        )
        
        # Check for language consistency in response
        response_consistency = self._check_response_consistency(response, response_lang_analysis)
        
        # Evaluate with judge if available
        judge_scores = {}
        if self.judge_adapter:
            judge_scores = self._judge_quality(prompt, response, mix_type)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            understood_mix=understood_mix,
            lang_appropriate=response_lang_appropriate,
            consistency=response_consistency,
            judge_scores=judge_scores
        )
        
        return {
            'prompt': prompt,
            'response': response,
            'mix_type': mix_type,
            'expected_languages': expected_languages,
            'expected_response_language': expected_response_language,
            
            # Language analysis
            'prompt_languages': prompt_lang_analysis,
            'response_languages': response_lang_analysis,
            
            # Evaluation metrics
            'understood_mix': understood_mix,
            'response_language_appropriate': response_lang_appropriate,
            'response_consistency': response_consistency,
            'overall_score': overall_score,
            
            # Judge evaluation
            'judge_scores': judge_scores,
            
            # Metadata
            'latency': generated.get('latency', 0),
            'tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
        }
    
    def _analyze_text_languages(self, text: str) -> Dict[str, Any]:
        """
        Analyze which languages are present in text.
        
        Returns:
            Dict with language presence and ratios
        """
        text_lower = text.lower()
        
        # Count Turkish indicators
        turkish_count = sum(
            len(re.findall(pattern, text_lower, re.IGNORECASE))
            for pattern in self.turkish_patterns
        )
        
        # Count English indicators
        english_count = sum(
            len(re.findall(pattern, text_lower, re.IGNORECASE))
            for pattern in self.english_patterns
        )
        
        # Count technical terms
        technical_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in self.technical_terms
        )
        
        total_indicators = turkish_count + english_count
        
        # Calculate ratios
        turkish_ratio = turkish_count / max(total_indicators, 1)
        english_ratio = english_count / max(total_indicators, 1)
        
        # Determine primary language
        if turkish_ratio > 0.6:
            primary_language = "tr"
        elif english_ratio > 0.6:
            primary_language = "en"
        else:
            primary_language = "mixed"
        
        # Check if text is truly mixed (both languages present significantly)
        is_mixed = turkish_ratio > 0.2 and english_ratio > 0.2
        
        return {
            'primary_language': primary_language,
            'is_mixed': is_mixed,
            'turkish_ratio': turkish_ratio,
            'english_ratio': english_ratio,
            'turkish_indicators': turkish_count,
            'english_indicators': english_count,
            'technical_terms': technical_count,
            'has_technical_terms': technical_count > 0
        }
    
    def _check_understanding(self, prompt: str, response: str, mix_type: str) -> bool:  # noqa: ARG002
        """
        Check if model understood the mixed-language query.
        
        Returns:
            True if model appears to have understood
        """
        # Basic checks
        response_lower = response.lower()
        
        # Check for confusion indicators
        confusion_indicators = [
            "anlamadım", "anlayamadım", "i don't understand", "i didn't understand",
            "unclear", "belirsiz", "ne demek istediğinizi", "what do you mean",
            "could you clarify", "açıklayabilir misiniz"
        ]
        
        if any(indicator in response_lower for indicator in confusion_indicators):
            return False
        
        # Check if response is too short (might indicate confusion)
        if len(response.split()) < 5:
            return False
        
        # Check if response contains relevant content
        # Extract key terms from prompt
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
        
        # Some overlap in content words suggests understanding
        overlap = len(prompt_words & response_words)
        
        return overlap >= 1  # At least one significant word overlap
    
    def _check_response_language(
        self,
        response_analysis: Dict[str, Any],
        expected_language: str = None
    ) -> bool:
        """
        Check if response language is appropriate.
        
        Args:
            response_analysis: Language analysis of response
            expected_language: Expected primary language (None = any)
            
        Returns:
            True if language is appropriate
        """
        if expected_language is None:
            # No specific expectation - just check it's not too confused
            return response_analysis['primary_language'] != 'unknown'
        
        if expected_language == 'mixed':
            return response_analysis['is_mixed']
        
        return response_analysis['primary_language'] == expected_language
    
    def _check_response_consistency(
        self,
        response: str,
        response_analysis: Dict[str, Any]  # noqa: ARG002
    ) -> float:
        """
        Check language consistency within response.
        
        Returns:
            Consistency score (0-1)
        """
        # Split response into sentences
        sentences = re.split(r'[.!?]+', response)
        
        if len(sentences) <= 1:
            return 1.0  # Single sentence is always consistent
        
        # Analyze each sentence
        sentence_langs = []
        for sentence in sentences:
            if sentence.strip():
                analysis = self._analyze_text_languages(sentence)
                sentence_langs.append(analysis['primary_language'])
        
        if not sentence_langs:
            return 1.0
        
        # Count language switches
        switches = sum(
            1 for i in range(len(sentence_langs) - 1)
            if sentence_langs[i] != sentence_langs[i + 1] and 
               sentence_langs[i] != 'mixed' and 
               sentence_langs[i + 1] != 'mixed'
        )
        
        # More switches = lower consistency (unless it's a mixed prompt)
        # Allow some switching for mixed prompts
        max_allowed_switches = len(sentence_langs) // 3  # Allow switching in 1/3 of boundaries
        
        if switches <= max_allowed_switches:
            return 1.0
        else:
            # Penalty for excessive switching
            return max(0.5, 1.0 - (switches - max_allowed_switches) * 0.1)
    
    def _judge_quality(
        self,
        prompt: str,
        response: str,
        mix_type: str
    ) -> Dict[str, float]:
        """
        Use judge model to evaluate response quality.
        
        Returns:
            Dict with judge scores
        """
        if not self.judge_adapter:
            return {}
        
        judge_prompt = f"""Aşağıdaki karma dilde (Türkçe-İngilizce) sorulan soruya verilen cevabı değerlendir.

Soru Tipi: {mix_type}
Soru: {prompt}

Cevap: {response}

Lütfen şu kriterleri 0-10 arasında puanla:
1. Relevance (İlgililik): Cevap soruyla ne kadar ilgili?
2. Accuracy (Doğruluk): Cevap ne kadar doğru ve güvenilir?
3. Language Handling (Dil Kullanımı): Karma dil kullanımı ne kadar iyi yönetilmiş?
4. Clarity (Açıklık): Cevap ne kadar açık ve anlaşılır?
5. Completeness (Tamlık): Cevap soruyu tam olarak yanıtlıyor mu?

JSON formatında sadece puanları ver:
{{"relevance": X, "accuracy": X, "language_handling": X, "clarity": X, "completeness": X}}"""

        try:
            judge_generated = self.judge_adapter.generate([
                {"role": "user", "content": judge_prompt}
            ])
            judge_response = judge_generated.get("content") or ""
            
            # Extract JSON scores
            import json
            # Try to find JSON in response
            json_match = re.search(r'\{[^}]+\}', judge_response)
            if json_match:
                scores = json.loads(json_match.group())
                # Normalize to 0-1
                return {k: v / 10.0 for k, v in scores.items()}
        except Exception as e:
            print(f"Judge evaluation failed: {e}")
        
        return {}
    
    def _calculate_overall_score(
        self,
        understood_mix: bool,
        lang_appropriate: bool,
        consistency: float,
        judge_scores: Dict[str, float]
    ) -> float:
        """
        Calculate overall language mix handling score.
        
        Returns:
            Score between 0 and 1
        """
        # Base score from boolean checks
        base_score = (
            (1.0 if understood_mix else 0.0) * 0.3 +
            (1.0 if lang_appropriate else 0.0) * 0.2 +
            consistency * 0.2
        )
        
        # Add judge scores if available
        if judge_scores:
            judge_avg = sum(judge_scores.values()) / len(judge_scores)
            base_score += judge_avg * 0.3
        else:
            # If no judge, redistribute weight
            base_score = base_score / 0.7
        
        return min(1.0, max(0.0, base_score))
