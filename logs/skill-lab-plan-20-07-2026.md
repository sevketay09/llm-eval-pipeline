# Skill Quality Lab — Geliştirme Planı (20-07-2026)

Branch: `feature/skill-quality-lab` (base: feature/eval-improvements)
Kural: sıfır yeni bağımlılık, her adımda test + commit + push + onay.

## Amaç
SKILL.md dosyalarını değerlendiren yapı: hazır/elde yazılmış bir skill,
kullanıcının yaptırmak istediği iş için kaliteli ve yeterli mi?
Farklılaştırıcı: piyasa araçları genel skor verir; biz **task-fit** (göreve
uygunluk) skoru ekliyoruz.

## Adımlar

| Adım | İçerik | Dosyalar | Durum |
|------|--------|----------|-------|
| S1 | Statik linter: frontmatter/name/description, gövde boyutu, boş bölüm, progressive disclosure, 6 güvenlik deseni; 0-100 skor | `analysis/skill_lint.py`, `tests/test_skill_lint_contracts.py` | ✅ `a3cc301` (20 test, 582/582) |
| S2 | Task-fit judge: 5 kriterli rubric (kapsam örtüşmesi, talimat netliği, eksik adımlar, konvansiyon uyumu, verimlilik riski), her kriter 0-1 + kanıt alıntısı | `evaluators/skill_fit_judge.py`, `tests/test_skill_fit_judge_contracts.py` | ✅ (13 test, 595/595) |
| S4 | API router: `/skill-eval/lint`, `/fit`, `/full`, `/reports`; rapor `reports/skill_eval_<ts>.json` | `api/routers/skill_eval.py`, router testleri | ⬜ |
| S5 | Web sayfası: SkillLab.tsx — paste/upload + görev tanımı + skor kartı + judge alıntıları | `web/src/pages/SkillLab.tsx`, route/sidebar | ⬜ |
| S6 | CLI + dokümantasyon: `run_skill_eval.py`, README/README.tr bölümü, badge | `run_skill_eval.py`, README'ler | ⬜ |
| S3 | (SONA ALINDI — kullanıcı kararı) Trigger simülasyonu: tetiklemeli/tetiklememeli/belirsiz prompt'larla routing precision/recall | `analysis/skill_trigger.py`, testler | ⬜ |

## Referanslar (fikrin dayanağı)
- philschmid — Testing Skills rehberi: 4 boyut (triggering, output kalitesi, konvansiyon, verimlilik), 10-20 prompt'luk test seti → https://www.philschmid.de/testing-skills
- Anthropic — Improving skill-creator: eval pass rate + süre + token metrikleri, skill'li vs skill'siz A/B → https://claude.com/blog/improving-skill-creator-test-measure-and-refine-agent-skills
- anthropics/skills skill-creator: SKILL.md format/frontmatter spesifikasyonu (name ≤64, description ≤1024) → https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md
- Skill Bench: kanıt-alıntılı judge (skor + alıntı) → https://skill-bench.dev/
- mdskills.ai Skill Advisor: kalite + güvenlik birlikte skorlama → https://www.mdskills.ai/docs/skill-advisor
- skill-quality-benchmarker: statik analiz kontrolleri → https://mcpmarket.com/tools/skills/skill-quality-benchmarker
- Promptfoo — Test Agent Skills → https://www.promptfoo.dev/docs/guides/test-agent-skills/

## Altyapı yeniden kullanımı
- `evaluators/judge_utils.py` (request_judge_json, parse-fail → None, asla sahte 0.0)
- `evaluators/nlp_metrics.py` (S3 trigger benzerlik ölçümü için)
- Mock provider (API key'siz test/demo)
