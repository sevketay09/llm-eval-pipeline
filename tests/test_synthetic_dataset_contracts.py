"""Contract tests for datasets.generate — offline, no real LLM required."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from datagen.generate import (
    _default_title,
    _extract_json,
    _is_nondeterministic,
    _normalize_cases,
    chunk_text,
    generate_qa_from_source,
    load_source_files,
)


def _make_llm_fn(cases: list[dict], title: str = "Test Dataset") -> object:
    """Return a fake llm_fn that yields deterministic JSON with the given cases."""
    payload = {"title": title, "test_cases": cases}

    def llm_fn(messages):
        return json.dumps(payload, ensure_ascii=False)

    return llm_fn


def _valid_case(i: int = 1) -> dict:
    return {
        "id": f"case_{i:03d}",
        "category": "support",
        "difficulty": "medium",
        "question": f"Ürün iadesi nasıl yapılır? (case {i})",
        "expected_answer": f"İade için müşteri hizmetlerini aramanız gerekir. (case {i})",
    }


class ChunkTextContractTests(unittest.TestCase):
    def test_empty_text_returns_empty_list(self):
        self.assertEqual(chunk_text(""), [])

    def test_whitespace_only_returns_empty_list(self):
        self.assertEqual(chunk_text("   \n  "), [])

    def test_single_short_paragraph_yields_one_chunk(self):
        chunks = chunk_text("Hello world")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["id"], "src_001")
        self.assertEqual(chunks[0]["text"], "Hello world")

    def test_two_paragraphs_merged_when_short_enough(self):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertIn("First paragraph", chunks[0]["text"])
        self.assertIn("Second paragraph", chunks[0]["text"])

    def test_long_paragraph_split_at_chunk_size(self):
        long_para = "x" * 1800
        chunks = chunk_text(long_para)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"]), 600)

    def test_max_chunks_capped_at_8(self):
        # 9 distinct paragraphs each > CHUNK_SIZE would exceed limit
        paragraphs = "\n\n".join(["y" * 700 for _ in range(10)])
        chunks = chunk_text(paragraphs)
        self.assertLessEqual(len(chunks), 8)

    def test_chunk_ids_are_sequential(self):
        text = "\n\n".join([f"Paragraph {i}" for i in range(5)])
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk["id"], f"src_{i:03d}")


class NormalizeCasesContractTests(unittest.TestCase):
    def test_valid_cases_are_kept(self):
        cases = [_valid_case(i) for i in range(1, 6)]
        kept, fs = _normalize_cases(cases, 10)
        self.assertEqual(len(kept), 5)
        self.assertEqual(fs["kept_cases"], 5)
        self.assertEqual(fs["invalid_removed"], 0)

    def test_non_dict_cases_are_removed(self):
        cases = [_valid_case(1), "bad", 42, None, _valid_case(2), _valid_case(3)]
        kept, fs = _normalize_cases(cases, 10)
        self.assertEqual(len(kept), 3)
        self.assertEqual(fs["invalid_removed"], 3)

    def test_missing_question_or_answer_removes_case(self):
        cases = [
            {"id": "x1", "category": "a", "difficulty": "easy", "question": "Q?", "expected_answer": ""},
            {"id": "x2", "category": "a", "difficulty": "easy", "question": "", "expected_answer": "A"},
            _valid_case(3),
            _valid_case(4),
            _valid_case(5),
        ]
        kept, fs = _normalize_cases(cases, 10)
        self.assertEqual(fs["invalid_removed"], 2)
        self.assertEqual(len(kept), 3)

    def test_duplicate_questions_are_deduped(self):
        same = _valid_case(1)
        cases = [same, same.copy(), _valid_case(2), _valid_case(3)]
        kept, fs = _normalize_cases(cases, 10)
        self.assertEqual(fs["duplicate_removed"], 1)
        self.assertEqual(len(kept), 3)

    def test_nondeterministic_answer_is_removed(self):
        bad = {
            "id": "nd1",
            "category": "support",
            "difficulty": "easy",
            "question": "Ne zaman teslimat yapılır?",
            "expected_answer": "It depends on your location.",
        }
        cases = [bad, _valid_case(2), _valid_case(3), _valid_case(4)]
        kept, fs = _normalize_cases(cases, 10)
        self.assertEqual(fs["nondeterministic_removed"], 1)
        self.assertFalse(any(c["id"] == "nd1" for c in kept))

    def test_sample_count_is_respected(self):
        cases = [_valid_case(i) for i in range(1, 20)]
        kept, fs = _normalize_cases(cases, 5)
        self.assertEqual(fs["input_cases"], 5)
        self.assertLessEqual(len(kept), 5)

    def test_output_cases_have_required_fields(self):
        cases = [_valid_case(i) for i in range(1, 4)]
        kept, _ = _normalize_cases(cases, 10)
        required = {"id", "category", "difficulty", "question", "expected_answer", "mutation_type", "risk_tags", "mutation_metadata"}
        for case in kept:
            self.assertTrue(required.issubset(case.keys()))


class GenerateQaFromSourceContractTests(unittest.TestCase):
    def test_returns_expected_top_level_keys(self):
        cases = [_valid_case(i) for i in range(1, 11)]
        result = generate_qa_from_source(
            source_text="Ürün iade politikası belgesinden alınmış metin.",
            project_description="E-ticaret müşteri destek botu için iade ve kargo yönetimi. Bu bot müşterilere yardım eder.",
            llm_fn=_make_llm_fn(cases),
            sample_count=10,
        )
        for key in ("title", "test_cases", "source_attribution", "generated_at", "filtering_summary"):
            self.assertIn(key, result)

    def test_source_attribution_contains_chunks_when_source_provided(self):
        source = "Ürünlerimiz yüksek kalitelidir.\n\nİade politikamız 30 gündür."
        cases = [_valid_case(i) for i in range(1, 6)]
        result = generate_qa_from_source(
            source_text=source,
            project_description="Müşteri destek botu iade ve ürün kalitesi hakkında kullanıcılara yardım eder.",
            llm_fn=_make_llm_fn(cases),
        )
        attr = result["source_attribution"]
        self.assertGreater(attr["source_length"], 0)
        self.assertIn("source_chunks", attr)
        self.assertGreater(len(attr["source_chunks"]), 0)

    def test_invalid_json_response_raises_value_error(self):
        def bad_llm(messages):
            return "this is not json"

        with self.assertRaises(ValueError, msg="Should raise on unparseable response"):
            generate_qa_from_source(
                source_text="",
                project_description="Bir ürün için test dataset oluşturulacak ve bu çok önemli.",
                llm_fn=bad_llm,
            )

    def test_too_few_valid_cases_after_filtering_raises(self):
        def llm_fn(messages):
            return json.dumps({"title": "T", "test_cases": [
                {"id": "x1", "category": "a", "difficulty": "e", "question": "Q1?", "expected_answer": "it depends"},
                {"id": "x2", "category": "a", "difficulty": "e", "question": "Q2?", "expected_answer": "not specified"},
            ]})

        with self.assertRaises(ValueError):
            generate_qa_from_source(
                source_text="",
                project_description="Ürün değerlendirme botu için test dataset. Çok önemli bir sistem.",
                llm_fn=llm_fn,
            )

    def test_focus_areas_passed_to_llm_in_prompt(self):
        captured = []

        def recording_llm(messages):
            captured.extend(messages)
            return json.dumps({"title": "T", "test_cases": [_valid_case(i) for i in range(1, 5)]})

        generate_qa_from_source(
            source_text="",
            project_description="Müşteri destek sistemi, iade ve ödeme konularını yönetir ve çok önemli.",
            llm_fn=recording_llm,
            focus_areas="iade, kargo",
        )
        user_prompt = next(m["content"] for m in captured if m["role"] == "user")
        self.assertIn("iade, kargo", user_prompt)

    def test_empty_source_text_generates_without_chunks(self):
        cases = [_valid_case(i) for i in range(1, 5)]
        result = generate_qa_from_source(
            source_text="",
            project_description="Sıfır kaynak materyalle oluşturulan bir test dataset. Bu test çok önemli.",
            llm_fn=_make_llm_fn(cases),
        )
        self.assertEqual(result["source_attribution"]["source_length"], 0)
        self.assertEqual(result["source_attribution"]["source_chunks"], [])


class ExtractJsonContractTests(unittest.TestCase):
    def test_plain_json_parsed(self):
        result = _extract_json('{"a": 1}')
        self.assertEqual(result, {"a": 1})

    def test_markdown_fenced_json_parsed(self):
        result = _extract_json('```json\n{"a": 1}\n```')
        self.assertEqual(result, {"a": 1})

    def test_invalid_json_returns_none(self):
        result = _extract_json("not json at all")
        self.assertIsNone(result)

    def test_json_embedded_in_prose_extracted(self):
        result = _extract_json('Here is the result: {"title": "T", "test_cases": []}')
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "T")


class IsNondeterministicContractTests(unittest.TestCase):
    def test_deterministic_answer_passes(self):
        self.assertFalse(_is_nondeterministic("İade için müşteri hizmetlerini arayın."))

    def test_it_depends_detected(self):
        self.assertTrue(_is_nondeterministic("It depends on your location."))

    def test_turkish_phrase_detected(self):
        self.assertTrue(_is_nondeterministic("Duruma göre değişir."))

    def test_case_insensitive(self):
        self.assertTrue(_is_nondeterministic("NOT SPECIFIED for this case"))


class LoadSourceFilesContractTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_source_files(["/nonexistent/path/file.txt"])

    def test_single_file_loaded(self, tmp_path=None):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Guide\nBu bir rehber dosyasıdır.")
            tmp = f.name
        try:
            text = load_source_files([tmp])
            self.assertIn("Bu bir rehber dosyasıdır.", text)
            self.assertIn("Guide", text)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_multiple_files_concatenated(self):
        import tempfile
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write(f"Content {i}")
                files.append(f.name)
        try:
            text = load_source_files(files)
            for i in range(3):
                self.assertIn(f"Content {i}", text)
        finally:
            for fp in files:
                Path(fp).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
