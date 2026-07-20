## Test suite onarımı — 90 kırık test sıfırlandı

Kullanıcı, yeni geliştirme fikri öncesinde fail olan tüm testlerin sebeplerinin araştırılıp çözülmesini istedi.
tests/ altındaki 8 test dosyası (path ve stale-mock düzeltmeleri), conftest.py (sklearn.metrics, scipy.stats.ttest_1samp saf Python mock, event-loop guard), evaluate_api.py, pipeline_runner.py, README.md ve README.tr.md dosyalarına dokunuldu.
Kök nedenler: test dosyaları root'tan tests/'e taşınırken kırılan Path(__file__).parent yolları, evaluate_api.evaluate_single'daki gerçek NameError bug'ı (tanımsız run değişkeni), test_evaluate_api_smoke'un sys.modules'e fake pipeline_runner sızdırması, Python 3.9'da asyncio.run sonrası event loop kalmaması, scipy mock'unun ttest_1samp'ı unpack edilemez döndürmesi ve run_qa_test'in as_completed ile sonuçları sırasız raporlaması (dataset sırası korunacak şekilde production'da düzeltildi).
Sonuç: 42 fail + 48 error'dan 562/562 pass'e ulaşıldı, sabit ve rastgele test sırasında 4 koşuda stabil doğrulandı, rozet 562'ye güncellendi.

## Skill Quality Lab — S1 static linter approval ve implementation

The user approved step S1 of the Skill Quality Lab plan: a static SKILL.md linter with no LLM calls and no new dependencies, on new branch feature/skill-quality-lab.
Files touched: analysis/skill_lint.py (new) and tests/test_skill_lint_contracts.py (new).
Implemented frontmatter/name/description validation, body size and dead-section checks, progressive-disclosure detection, and six security red-flag patterns (pipe-to-shell, destructive rm, base64-exec, sudo, secret-file access, env exfiltration), producing a 0-100 score with per-check severity.
Outcome: 20 new contract tests pass and the full suite is 582/582 green; committed and pushed to feature/skill-quality-lab.
