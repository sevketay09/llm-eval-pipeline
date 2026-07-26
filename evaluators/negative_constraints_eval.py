"""
Negative Constraints Evaluator
Tests model's ability to follow "do NOT do X" instructions
"""
import re
import json
from typing import Dict, List, Any, Optional
from adapters.unified_adapter import UnifiedLLMAdapter


class NegativeConstraintsEvaluator:
    """
    Evaluator for testing model compliance with negative constraints.
    
    Tests scenarios like:
    - "Do NOT use JSON format"
    - "Do NOT mention specific words"
    - "Do NOT exceed X words"
    - "Do NOT use numbers"
    - "Do NOT use English words"
    """
    
    def __init__(self, judge_adapter: Optional[UnifiedLLMAdapter] = None):
        """
        Initialize the evaluator.
        
        Args:
            judge_adapter: Optional LLM for semantic evaluation
        """
        self.judge_adapter = judge_adapter
    
    def evaluate_negative_constraint(
        self,
        model: UnifiedLLMAdapter,
        prompt: str,
        constraint_type: str,
        constraint_params: Dict[str, Any],
        expected_violation: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single negative constraint test.
        
        Args:
            model: The LLM adapter to test
            prompt: The prompt with negative constraint
            constraint_type: Type of constraint (format, word_limit, content, language, etc.)
            constraint_params: Parameters for constraint validation
            expected_violation: Whether violation is expected (for testing edge cases)
        
        Returns:
            Evaluation result with compliance metrics
        """
        # Generate response via unified adapter contract (messages -> dict)
        generated = model.generate([
            {"role": "user", "content": prompt}
        ])
        if generated.get("error"):
            # Infrastructure failure, not the model's answer — the caller
            # must check `generation_error` and exclude this item instead of
            # scoring an empty string as a (non-)compliant response.
            return {
                "generation_error": generated["error"],
                "response": "",
                "constraint_type": constraint_type,
            }
        response = generated.get("content") or ""

        # Check constraint compliance based on type
        violation_detected = False
        violation_details = []
        
        if constraint_type == "format":
            violation_detected, violation_details = self._check_format_constraint(
                response, constraint_params
            )
        elif constraint_type == "word_limit":
            violation_detected, violation_details = self._check_word_limit_constraint(
                response, constraint_params
            )
        elif constraint_type == "forbidden_words":
            violation_detected, violation_details = self._check_forbidden_words_constraint(
                response, constraint_params
            )
        elif constraint_type == "forbidden_content":
            violation_detected, violation_details = self._check_forbidden_content_constraint(
                response, constraint_params
            )
        elif constraint_type == "language":
            violation_detected, violation_details = self._check_language_constraint(
                response, constraint_params
            )
        elif constraint_type == "numeric":
            violation_detected, violation_details = self._check_numeric_constraint(
                response, constraint_params
            )
        elif constraint_type == "structure":
            violation_detected, violation_details = self._check_structure_constraint(
                response, constraint_params
            )
        else:
            violation_detected = False
            violation_details = [f"Unknown constraint type: {constraint_type}"]
        
        # Calculate compliance score
        compliant = not violation_detected if not expected_violation else violation_detected
        compliance_score = 1.0 if compliant else 0.0
        
        # Severity of violation (0-1, higher = worse)
        severity = len(violation_details) / max(1, constraint_params.get("max_violations", 10))
        severity = min(1.0, severity)
        
        return {
            "compliant": compliant,
            "compliance_score": compliance_score,
            "violation_detected": violation_detected,
            "violation_details": violation_details,
            "violation_count": len(violation_details),
            "severity": severity,
            "response": response,
            "constraint_type": constraint_type,
            "expected_violation": expected_violation
        }
    
    def _check_format_constraint(
        self,
        response: str,
        params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Check format-related constraints (e.g., 'do NOT use JSON')"""
        violations = []
        forbidden_format = params.get("forbidden_format", "")
        
        if forbidden_format == "json":
            # Check for JSON-like structures
            if re.search(r'\{.*".*".*:.*\}', response, re.DOTALL):
                violations.append("Response contains JSON-like structure with curly braces and colons")
            if re.search(r'\[.*\]', response):
                violations.append("Response contains array-like structure with square brackets")
        
        elif forbidden_format == "list":
            # Check for list formatting
            if re.search(r'^\s*[-*•]\s+', response, re.MULTILINE):
                violations.append("Response contains bullet point list")
            if re.search(r'^\s*\d+[\.)]\s+', response, re.MULTILINE):
                violations.append("Response contains numbered list")
        
        elif forbidden_format == "table":
            # Check for table formatting
            if re.search(r'\|.*\|', response):
                violations.append("Response contains table with pipe separators")
            if re.search(r'─|═|│|║', response):
                violations.append("Response contains table drawing characters")
        
        elif forbidden_format == "code":
            # Check for code blocks
            if re.search(r'```', response):
                violations.append("Response contains code block markers")
            if re.search(r'^\s{4,}', response, re.MULTILINE):
                violations.append("Response contains indented code-like structure")
        
        return len(violations) > 0, violations
    
    def _check_word_limit_constraint(
        self,
        response: str,
        params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Check word count constraints"""
        violations = []
        word_count = len(response.split())
        
        max_words = params.get("max_words")
        min_words = params.get("min_words")
        
        if max_words is not None and word_count > max_words:
            violations.append(f"Response has {word_count} words, exceeding max of {max_words}")
        
        if min_words is not None and word_count < min_words:
            violations.append(f"Response has {word_count} words, below min of {min_words}")
        
        return len(violations) > 0, violations
    
    def _check_forbidden_words_constraint(
        self,
        response: str,
        params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Check for forbidden specific words"""
        violations = []
        forbidden_words = params.get("forbidden_words", [])
        
        response_lower = response.lower()
        for word in forbidden_words:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(word.lower()) + r'\b'
            if re.search(pattern, response_lower):
                violations.append(f"Response contains forbidden word: '{word}'")
        
        return len(violations) > 0, violations
    
    def _check_forbidden_content_constraint(
        self,
        response: str,
        params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Check for forbidden content types"""
        violations = []
        forbidden_content = params.get("forbidden_content", "")
        
        if forbidden_content == "examples":
            if re.search(r'örnek|example|mesela|şöyle ki', response, re.IGNORECASE):
                violations.append("Response contains examples despite instruction not to")
        
        elif forbidden_content == "explanations":
            if re.search(r'çünkü|because|nedeni|sebep|açıklama', response, re.IGNORECASE):
                violations.append("Response contains explanations despite instruction not to")
        
        elif forbidden_content == "definitions":
            if re.search(r'tanım|definition|anlamı|demek ki', response, re.IGNORECASE):
                violations.append("Response contains definitions despite instruction not to")
        
        elif forbidden_content == "personal_opinions":
            if re.search(r'bence|bana göre|in my opinion|i think|i believe', response, re.IGNORECASE):
                violations.append("Response contains personal opinions despite instruction not to")
        
        return len(violations) > 0, violations
    
    def _check_language_constraint(
        self,
        response: str,
        params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Check language constraints (e.g., 'do NOT use English words')"""
        violations = []
        forbidden_language = params.get("forbidden_language", "")
        
        if forbidden_language == "english":
            # Simple heuristic: look for common English words
            english_words = [
                'the', 'and', 'or', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
                'can', 'this', 'that', 'these', 'those', 'for', 'with', 'from', 'about'
            ]
            response_words = set(re.findall(r'\b[a-zA-Z]+\b', response.lower()))
            found_english = response_words.intersection(english_words)
            if found_english:
                violations.append(f"Response contains English words: {', '.join(list(found_english)[:5])}")
        
        elif forbidden_language == "turkish":
            # Look for Turkish characters/patterns
            turkish_chars = set('çğıöşüÇĞİÖŞÜ')
            if any(char in response for char in turkish_chars):
                violations.append("Response contains Turkish characters despite instruction not to")
        
        return len(violations) > 0, violations
    
    def _check_numeric_constraint(
        self,
        response: str,
        params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Check numeric constraints (e.g., 'do NOT use numbers')"""
        violations = []
        forbidden_numeric = params.get("forbidden_numeric", False)
        
        if forbidden_numeric:
            # Check for digits
            numbers = re.findall(r'\d+', response)
            if numbers:
                violations.append(f"Response contains {len(numbers)} numeric values: {', '.join(numbers[:5])}")
        
        return len(violations) > 0, violations
    
    def _check_structure_constraint(
        self,
        response: str,
        params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Check structural constraints (e.g., 'do NOT use multiple sentences')"""
        violations = []
        forbidden_structure = params.get("forbidden_structure", "")
        
        if forbidden_structure == "multiple_sentences":
            # Count sentences (simple heuristic)
            sentence_count = len(re.findall(r'[.!?]+', response))
            if sentence_count > 1:
                violations.append(f"Response contains {sentence_count} sentences, only 1 allowed")
        
        elif forbidden_structure == "paragraphs":
            # Check for paragraph breaks
            paragraph_count = len(re.split(r'\n\s*\n', response.strip()))
            if paragraph_count > 1:
                violations.append(f"Response contains {paragraph_count} paragraphs, only 1 allowed")
        
        return len(violations) > 0, violations


def evaluate_negative_constraints(
    model: UnifiedLLMAdapter,
    test_cases: List[Dict[str, Any]],
    judge_adapter: Optional[UnifiedLLMAdapter] = None
) -> Dict[str, Any]:
    """
    Evaluate multiple negative constraint test cases.
    
    Args:
        model: The LLM adapter to test
        test_cases: List of test case dictionaries
        judge_adapter: Optional LLM for semantic evaluation
    
    Returns:
        Aggregate results with compliance metrics
    """
    evaluator = NegativeConstraintsEvaluator(judge_adapter=judge_adapter)
    results = []
    
    for test_case in test_cases:
        result = evaluator.evaluate_negative_constraint(
            model=model,
            prompt=test_case["prompt"],
            constraint_type=test_case["constraint_type"],
            constraint_params=test_case["constraint_params"],
            expected_violation=test_case.get("expected_violation", False)
        )
        result["test_id"] = test_case.get("id", "unknown")
        result["category"] = test_case.get("category", "unknown")
        results.append(result)
    
    # Calculate aggregate metrics
    total_tests = len(results)
    compliant_tests = sum(1 for r in results if r["compliant"])
    total_violations = sum(r["violation_count"] for r in results)
    avg_severity = sum(r["severity"] for r in results) / total_tests if total_tests > 0 else 0
    
    compliance_rate = compliant_tests / total_tests if total_tests > 0 else 0
    
    # Group by constraint type
    by_type = {}
    for result in results:
        ctype = result["constraint_type"]
        if ctype not in by_type:
            by_type[ctype] = []
        by_type[ctype].append(result)
    
    type_compliance = {}
    for ctype, type_results in by_type.items():
        type_compliant = sum(1 for r in type_results if r["compliant"])
        type_compliance[ctype] = type_compliant / len(type_results) if type_results else 0
    
    return {
        "results": results,
        "summary": {
            "total_tests": total_tests,
            "compliant_tests": compliant_tests,
            "compliance_rate": compliance_rate,
            "total_violations": total_violations,
            "avg_violations_per_test": total_violations / total_tests if total_tests > 0 else 0,
            "avg_severity": avg_severity,
            "compliance_by_type": type_compliance,
            "overall_score": compliance_rate
        }
    }
