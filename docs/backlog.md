**Birleşik Backlog**

Bu backlog, Azure Evaluate incelemesi ile DeepEval incelemesinin tek bir ürün/yol haritası çıktısıdır. Ana hedef, ikinci bir Azure katmanı kurmak değil; mevcut framework içinde ortak evaluation sözleşmesi, trace-first ölçümleme, dataset üretimi, review workspace ve ürünleşmiş raporlama yüzeylerini inşa etmektir.

**Faz 1: Çekirdek Evaluation Mimarisi**

1. `BE-001` Ortak Metric Sözleşmesi ve Registry
Amaç: Tüm evaluator çıktılarının tek formatta toplanması.
Kapsam: `MetricDefinition`, `MetricResult`, `MetricGroup`, `MetricProvider`, `MetricExecutionContext` şemalarının tanımlanması.
Alt işler: Azure quality, Azure groundedness, judge skorları, safety skorları ve mevcut özel skorların yeni sözleşmeye map edilmesi.
Alt işler: `score`, `normalized_score`, `success`, `reason`, `provider`, `cost`, `latency`, `metadata`, `raw_payload` alanlarının standartlaştırılması.
Alt işler: Test bazlı ve run bazlı aggregate helper’ların eklenmesi.
Çıktı: Pipeline içinde tüm metric’lerin aynı JSON şekline oturması.
2. `BE-002` Provider-agnostic Evaluator Katmanı
Amaç: Azure dahil tüm evaluator’ları framework içi sağlayıcı olarak çalıştırmak.
Kapsam: `AzureQualityEvaluator`, `AzureAgentEvaluator`, `FaithfulnessEvaluator`, `SafetyEvaluator`, judge evaluator’larının ortak adapter arayüzüne geçirilmesi.
Alt işler: Her evaluator için `can_run`, `evaluate`, `normalize`, `serialize` davranışlarının netleştirilmesi.
Alt işler: Fail-soft davranışı tanımlanması; evaluator patladığında run’ın düşmemesi.
Alt işler: Provider capability matrix oluşturulması.
Çıktı: “Metric family” mantığında çalışan evaluator altyapısı.
3. `BE-003` Test Case Tiplerinin Standardizasyonu
Amaç: Evaluation mantığını test adı yerine veri tipi üzerinden kurmak.
Kapsam: `SingleTurnCase`, `ConversationalCase`, `AgentTraceCase`, `RagCase`, `StructuredOutputCase` veri modellerinin tanımlanması.
Alt işler: Mevcut dataset loader’ların bu case tiplerine dönüştürülmesi.
Alt işler: Tool çağrıları, retrieval context, expected tools, expected JSON schema, metadata alanlarının birinci sınıf alanlar olarak eklenmesi.
Alt işler: Case serialization/deserialization desteği eklenmesi.
Çıktı: Metric’lerin case tipine göre tekrar kullanılabilir hale gelmesi.
4. `BE-004` Run Sonuç Modelinin Yeniden Tasarımı
Amaç: Sonuçların raporlama, trend analizi ve review workspace için hazır olması.
Kapsam: `RunResult`, `TestResult`, `CaseResult`, `MetricResult`, `TraceResult` veri modellerinin tasarlanması.
Alt işler: Mevcut reproducibility, hash, cache ve trend verilerinin yeni modele taşınması.
Alt işler: Schema compliance, latency, cost, disagreement, evaluator errors alanlarının standart hale getirilmesi.
Alt işler: HTML/JSON/Markdown export formatlarının tek kaynaktan üretilmesi.
Çıktı: Run verisinin UI, API ve offline analiz için ortak kullanılması.

**Faz 2: Agent ve Multi-turn Evaluation Derinliği**

1. `AG-001` Agentic Testlerde Full Evaluation’a Geçiş
Amaç: Agent benchmark’larda sade değil tam agent değerlendirmesi yapmak.
Kapsam: `evaluate_simple` yerine koşula bağlı `evaluate_all` kullanımı.
Alt işler: Tool bilgisi olan case’lerde task adherence ve tool correctness’i aktifleştirmek.
Alt işler: Tool trace olmayan senaryolarda fallback stratejisi tanımlamak.
Alt işler: Aggregate score hesaplama mantığını yeniden kalibre etmek.
Çıktı: Agent görevlerinde eksik kalan ölçüm boyutlarının kapanması.
2. `AG-002` Agent Trace Modeli
Amaç: Agent akışını ölçülebilir bir trace ağacına dönüştürmek.
Kapsam: `agent`, `tool`, `llm`, `retriever`, `system` span tiplerinin tanımlanması.
Alt işler: Her span için süre, input/output özeti, status, metric sonuçları ve hata bilgisinin tutulması.
Alt işler: Run sırasında trace capture noktalarının yerleştirilmesi.
Alt işler: Trace ile case sonuçlarının bağlanması.
Çıktı: UI’da ve raporda gösterilebilir trace-first evaluation verisi.
3. `AG-003` Agent Metric Pack v1
Amaç: Agent davranışını reusable metric ailesiyle ölçmek.
Kapsam: `task_completion`, `tool_correctness`, `plan_adherence`, `step_efficiency`, `response_completeness`, `intent_resolution`.
Alt işler: Mevcut Azure ve judge tabanlı skorların bu metric ailelerine bağlanması.
Alt işler: Her metric için threshold, strict mode, reason üretimi ve normalization tanımlanması.
Alt işler: Metric bazlı aggregate ve pass/fail mantığı eklenmesi.
Çıktı: Agent testleri için tutarlı metric paketi.
4. `AG-004` Multi-turn Metric Pack v1
Amaç: Multi-turn testleri tek context-retention prompt’undan çıkarmak.
Kapsam: `conversation_completeness`, `turn_faithfulness`, `turn_relevancy`, `context_retention`, `knowledge_retention`.
Alt işler: Turn windowing mantığının eklenmesi.
Alt işler: Retrieval context varsa turn-level groundedness hesaplanması.
Alt işler: User intention extraction ve unresolved intent tespiti yapılması.
Çıktı: Konuşma bazlı evaluation’ın kurumsal seviyeye çıkması.
5. `AG-005` MCP ve Tool Kullanımı Ölçümü
Amaç: Tool ve MCP kullanımını ayrıca ölçmek.
Kapsam: `mcp_task_completion`, `tool_selection`, `argument_correctness`, `tool_use_efficiency`.
Alt işler: Tool registry ve expected tool set bilgisinin case modeline bağlanması.
Alt işler: Yanlış tool seçimi, eksik tool çağrısı ve argüman hatalarının reason bazlı raporlanması.
Alt işler: UI’da tool misuse görünümü için veri üretimi.
Çıktı: Agent/tool davranışı daha net okunabilir hale gelir.

**Faz 3: Structured Output ve Prompt Disiplini**

1. `SO-001` JSON Correctness Metric
Amaç: Schema fail rate’i sadece aggregate oran değil case-level metric haline getirmek.
Kapsam: Expected schema ile response doğrulama metric’inin eklenmesi.
Alt işler: Pydantic veya JSON Schema tabanlı validator standardı belirlenmesi.
Alt işler: Parse error, schema error, missing field, type mismatch reason’larının ayrıştırılması.
Alt işler: Test summary’de schema compliance ve error taxonomy üretilmesi.
Çıktı: Structured output güvenilirliğinin doğrudan yönetilebilir hale gelmesi.
2. `SO-002` Prompt Alignment Metric
Amaç: Modelin sistem ve görev talimatına ne kadar uyduğunu ölçmek.
Kapsam: Prompt instruction listesi ile actual output uyumu değerlendirmesi.
Alt işler: Instruction extraction formatının tanımlanması.
Alt işler: Judge tabanlı alignment score ve reason mantığının eklenmesi.
Alt işler: Alignment failure’ları UI’da gösterilecek şekilde sınıflandırılması.
Çıktı: Prompt drift ve instruction violation görünür olur.
3. `SO-003` Structured Output Reliability Dashboard Verisi
Amaç: Ürün yüzeyinde structured output kalitesini görünür kılmak.
Kapsam: Model bazlı schema compliance, parser failure, invalid JSON trendleri.
Alt işler: Mevcut `schema_fail_rate_mean` metriğinin case-level histogram ile genişletilmesi.
Alt işler: Model, test, dataset ve schema tipine göre breakdown üretilmesi.
Çıktı: Structured output kararlarını yönlendiren raporlama yüzeyi.

**Faz 4: Dataset Studio 2.0**

1. `DS-001` Synthetic Dataset Generation Motoru
Amaç: Dataset Studio’yu sadece tek seferlik üretici değil, kontrollü evaluation asset üreticisi yapmak.
Kapsam: `generate_from_docs`, `generate_from_contexts`, `generate_from_scratch` akışlarının eklenmesi.
Alt işler: Chunking, context extraction, generation, evolution, filtering aşamalarının ayrılması.
Alt işler: Goldens için deterministic expected output üretme kurallarının sıkılaştırılması.
Alt işler: Source attribution bilgisinin saklanması.
Çıktı: Dataset üretimi daha kontrollü ve tekrar üretilebilir hale gelir.
2. `DS-002` Conversational Dataset Generation
Amaç: Çok turlu test setlerini otomatik üretmek.
Kapsam: Docs ve context’ten multi-turn senaryo üretimi.
Alt işler: Persona, expected outcome, turn count, difficulty, escalation ihtiyacı gibi alanların eklenmesi.
Alt işler: User intention zinciri ve adversarial turn varyasyonlarının üretilmesi.
Alt işler: Conversation template kütüphanesi oluşturulması.
Çıktı: Multi-turn eval kapsamı hızla büyütülebilir.
3. `DS-003` Dataset Taxonomy ve Tagging
Amaç: Dataset kalitesini ve kapsama alanını sistematik yönetmek.
Kapsam: `standard`, `variation`, `edge_case`, `adversarial`, `policy`, `tool_use`, `rag`, `structured_output` etiketleri.
Alt işler: Generator çıktılarının otomatik tag edilmesi.
Alt işler: Import edilen dataset’lerin normalize edilmesi.
Alt işler: Tag bazlı coverage görünümü hazırlanması.
Çıktı: Dataset portföyü yönetilebilir hale gelir.
4. `DS-004` Dataset Review ve Finalize Akışı
Amaç: Dataset üretiminden sonra insan onayını ürünün parçası yapmak.
Kapsam: Preview, edit, approve, reject, promote-to-regression akışları.
Alt işler: Generated case ile finalized case farkının saklanması.
Alt işler: SME/QA/PM review rollerinin eklenmesi.
Alt işler: “Bu failure’dan reusable metric üret” işaretleme alanı eklenmesi.
Çıktı: Dataset Studio, değerlendirme üretim hattının kalıcı parçası olur.

**Faz 5: Review Workspace ve HITL Ürünleşmesi**

1. `RV-001` Review Queue V2
Amaç: Mevcut auto-HITL queue’yu gerçek çalışma alanına çevirmek.
Kapsam: disagreement, low confidence, safety risk, schema fail, tool misuse gibi queue reason’larının eklenmesi.
Alt işler: Queue item veri modeli genişletilmesi.
Alt işler: Priority, owner, status, SLA alanlarının eklenmesi.
Alt işler: Batch triage desteği eklenmesi.
Çıktı: Review akışı mühendislik sonrası operasyonel bir yüzeye dönüşür.
2. `RV-002` Annotation Workspace
Amaç: İnceleme ve karar süreçlerini tek ekranda toplamak.
Kapsam: Case detayları, trace, metric reasons, expected vs actual, reviewer notes, label setleri.
Alt işler: Case-level diff görünümü eklenmesi.
Alt işler: Reviewer kararları için standard verdict şeması tanımlanması.
Alt işler: Disagreement export ve calibration raporları bağlanması.
Çıktı: İnsan değerlendirmesi dağınık değil, ürünleşmiş hale gelir.
3. `RV-003` Calibration ve Judge Quality Yönetimi
Amaç: Judge evaluator’ların güvenilirliğini ölçmek.
Kapsam: Human vs judge uyumu, false positive/false negative analizi.
Alt işler: Calibration sample seti oluşturulması.
Alt işler: Judge disagreement reason taxonomy oluşturulması.
Alt işler: Prompt version bazlı judge kıyaslaması yapılması.
Çıktı: LLM judge sonuçlarına daha kontrollü güvenilirlik katmanı gelir.
4. `RV-004` Failure’dan Metric Üretim Akışı
Amaç: Review çıktısını doğrudan evaluation sistemine geri beslemek.
Kapsam: Review item’dan yeni metric adayı, dataset adayı, prompt rule adayı çıkarılması.
Alt işler: “Convert to metric candidate” workflow eklenmesi.
Alt işler: Metric backlog entegrasyonu oluşturulması.
Alt işler: Failure cluster ekranı hazırlanması.
Çıktı: Review sistemi sadece gözlem değil iyileştirme motoru olur.

**Faz 6: Raporlama, Export ve Trend Analizi**

1. `RP-001` Unified Console ve File Report Katmanı
Amaç: Tüm run sonuçlarını ortak raporlayıcıdan üretmek.
Kapsam: Terminal summary, JSON export, Markdown export, HTML export.
Alt işler: Case paneli, aggregate metrics, fail-first görünüm ve truncation kuralları eklenmesi.
Alt işler: Run metadata, config snapshot, hash, cost, latency alanlarının görünür hale getirilmesi.
Alt işler: Report renderer’ların API ve CLI tarafından ortak kullanılması.
Çıktı: Raporlama parçalı olmaktan çıkar.
2. `RP-002` Trace-aware Report
Amaç: Report içinde trace-level metric’leri göstermek.
Kapsam: Span metric aggregate, tool failure breakdown, retriever quality, agent path summary.
Alt işler: Trace metric aggregate modelinin oluşturulması.
Alt işler: Report export katmanında trace section eklenmesi.
Alt işler: Case-level trace detail linkleme mantığı eklenmesi.
Çıktı: Agent davranışı rapordan okunabilir hale gelir.
3. `RP-003` Trend ve Regression Analizi
Amaç: Mevcut trend altyapısını ürün kararına bağlamak.
Kapsam: Metric bazlı zaman serisi, dataset bazlı drift, prompt version karşılaştırması.
Alt işler: Regression thresholds tanımlanması.
Alt işler: Baseline run seçme ve kıyaslama ekranı hazırlanması.
Alt işler: “New failures introduced” görünümü eklenmesi.
Çıktı: Trend analizi operasyonel karar aracı olur.
4. `RP-004` Cost ve Efficiency Reporting
Amaç: Kaliteyi maliyet ve gecikme ile birlikte yönetmek.
Kapsam: Metric cost, evaluator cost, model latency, quality-per-latency, quality-per-cost.
Alt işler: Provider bazlı cost normalization eklenmesi.
Alt işler: Run summary’ye efficiency metrics eklenmesi.
Alt işler: UI’da bottleneck paneli hazırlanması.
Çıktı: Model seçimi kalite kadar verimle de yönetilir.

**Faz 7: Ürün Arayüzü**

1. `UI-001` Run Summary Ekranı
Amaç: Bir run’ın sağlık durumunu ilk bakışta göstermek.
Kapsam: passed/failed/skipped, avg score, schema compliance, cost, latency, top failure reasons.
Alt işler: Metric table, aggregate cards, failure distribution ve timeline panelleri.
Alt işler: Dataset, model, test type ve prompt version filtreleri.
Çıktı: Yönetici ve mühendis için ortak özet ekran.
2. `UI-002` Agent Trace Terminal
Amaç: Agent çalışmasını terminal benzeri ama okunabilir bir yüzeyde göstermek.
Kapsam: Tree yapı, step türü badge’leri, süre, metric score, pass/fail durumu.
Alt işler: AGENT, TOOL, LLM, RETRIEVER, SYSTEM satır tipleri.
Alt işler: Step expand/collapse ve raw payload drawer.
Alt işler: Failed step highlight ve top reasoning görünümü.
Çıktı: Agent debugging ve review çok hızlanır.
3. `UI-003` Dataset Studio Ekranları
Amaç: Dataset generation ve dataset review’ü görsel iş akışına çevirmek.
Kapsam: Generation pipeline görünümü, source docs paneli, generated cases listesi, tag görünümü.
Alt işler: Stage progress paneli.
Alt işler: Generated vs finalized case diff görünümü.
Alt işler: Promote-to-regression aksiyonu.
Çıktı: Dataset Studio ürünün görünür güçlü alanlarından biri olur.
4. `UI-004` Multi-turn Conversation Explorer
Amaç: Multi-turn case sonuçlarını okunur ve karşılaştırılabilir yapmak.
Kapsam: Turn transcript, per-turn score, unresolved intent, retrieval context, faithfulness reason.
Alt işler: Sliding window score visualization.
Alt işler: Turn-level failure işaretleme.
Alt işler: Reviewer notes paneli.
Çıktı: Multi-turn kalite sorunları teşhis edilebilir hale gelir.
5. `UI-005` Review Workspace
Amaç: HITL sürecini tam ürün yüzeyine taşımak.
Kapsam: Queue, detail panel, action bar, assignment, annotation, escalation.
Alt işler: Reviewer persona bazlı filtreler.
Alt işler: Disagreement-only ve high-risk-only görünümleri.
Alt işler: Case-to-metric suggestion paneli.
Çıktı: Review operasyonu ölçeklenebilir hale gelir.

**Faz 8: Güvenlik ve Politika Ölçümleri**

1. `SF-001` Safety Metric Pack v2
Amaç: Güvenliği mevcut framework içinde standart metric ailesi yapmak.
Kapsam: PII leakage, toxicity, misuse, refusal quality, prompt injection resistance.
Alt işler: Mevcut safety evaluator çıktılarının metric registry’ye bağlanması.
Alt işler: Risk seviyesi ve category alanlarının standardize edilmesi.
Alt işler: Review queue entegrasyonu yapılması.
Çıktı: Güvenlik ayrı katman değil, evaluation sistemi içinde ölçülür.
2. `SF-002` Policy-aware Review ve Reporting
Amaç: Policy ihlallerini sadece skor değil operasyon konusu yapmak.
Kapsam: Safety failure queue, severity, reviewer workflow, audit trail.
Alt işler: Policy type taxonomy eklenmesi.
Alt işler: False positive düzeltme akışı eklenmesi.
Alt işler: Export’larda güvenlik özeti oluşturulması.
Çıktı: Güvenlik değerlendirmesi ürün seviyesinde yönetilir.

**Faz 9: Geliştirici Deneyimi ve Operasyon**

1. `DX-001` CLI ve API Uyumlaştırması
Amaç: Aynı evaluation çekirdeğinin CLI, API ve UI’dan çağrılması.
Kapsam: Run config, dataset selection, report export, trace toggles, metric pack selection.
Alt işler: CLI argümanları ile API request modelinin hizalanması.
Alt işler: Async run lifecycle standardının çıkarılması.
Alt işler: Error reporting standardı oluşturulması.
Çıktı: Çekirdek mantık üç farklı entry point’te ayrışmaz.
2. `DX-002` Prompt ve Metric Versioning
Amaç: Judge prompt’ları ve metric tanımlarını versiyonlanabilir hale getirmek.
Kapsam: Prompt version, metric version, schema version alanlarının run metadata’ya yazılması.
Alt işler: Regression analiziyle versiyon bağının kurulması.
Alt işler: Backward compatibility kuralları tanımlanması.
Çıktı: Sonuçların neden değiştiği izlenebilir hale gelir.
3. `DX-003` Test ve Doğrulama Kapsamı
Amaç: Yeni evaluation çekirdeğinin güvenli taşınması.
Kapsam: Contract tests, evaluator adapter tests, export tests, dataset normalization tests.
Alt işler: Golden sample setleri hazırlanması.
Alt işler: Snapshot test stratejisi oluşturulması.
Alt işler: End-to-end smoke eval pipeline kurulması.
Çıktı: Refactor güvenle ilerletilebilir.

**Önceliklendirme**

1. P1: `BE-001`, `BE-002`, `BE-003`, `AG-001`, `AG-002`, `SO-001`, `RV-001`
2. P2: `AG-003`, `AG-004`, `DS-001`, `DS-002`, `RP-001`, `UI-001`, `UI-002`
3. P3: `RV-002`, `RV-003`, `RP-002`, `UI-003`, `UI-004`, `SF-001`
4. P4: `RV-004`, `RP-003`, `RP-004`, `UI-005`, `DX-001`, `DX-002`, `DX-003`

**İlk Uygulama Dalgası**

1. Evaluation çekirdeği: ortak metric sözleşmesi, provider adapter katmanı, yeni result modeli
2. Agent derinliği: full agent evaluation, trace modeli, task/tool metric pack
3. Structured output: JSON correctness metric ve schema reliability görünümü
4. Review ürünü: queue v2 ve case-level reason yapısı
5. UI başlangıcı: run summary ve agent trace terminal

----

SESSION HANDOVER — LLM Eval Pipeline G-Roadmap

Proje: /home/sevketa/agents/llm-eval-pipeline
Tarih: 26-06-2026

---
Okunması gereken dosyalar (session başında, bu sırayla)

──────────────────────────────┬───────────────────────────────────────────────────────────────────────────────┐
│              Dosya              │                                    Ne için                                    │
├─────────────────────────────────┼──────────────────────────────────────────────────┤
──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ logs/development-26-06-2026.md  │ Bu session'da yapılanların tamamı, her G'nin detayı                           │
├─────────────────────────────────┼──────────────────────────────────────────────────┤
│ docs/plan.md                    │ Genel mimari plan ve karar geçmişi                                            │
├─────────────────────────────────┼──────────────────────────────────────────────────┤
│ docs/backlog.md                 │ İç mimari borç ve konsolidasyon maddeleri (G roadmap'e ek)                    │
├─────────────────────────────────┼──────────────────────────────────────────────────┤
ome/sevketa/.claude/CLAUDE.md │ Global çalışma kuralları (caveman mode, token disiplini, daily log protokolü) │
└─────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────┘

▎ docs/EMBEDDING_EVALUATION.md — yalnızca embedding evaluator'larına dokunuluyorsa oku.
▎ logs/plan-26-06-2026.md — 3 satır, atlayabilirsin.

---
Tamamlanan maddeler (bu session)

┌─────────────────────────────┬─────────────────────────────────┬──────┐
│            Madde            │              Dosya              │ Test │
├─────────────────────────────┼─────────────────────────────────┼─────
│ G2 — Eval-as-CI             │ ci/gate.py, ci/pytest_plugin.py │ 16   │
├─────────────────────────────┼─────────────────────────────────┼─────
│ G6 — İstatistiksel sıkılık  │ analysis/significance.py        │ 10   │
├─────────────────────────────┼─────────────────────────────────┼─────
│ G4 — Sentetik dataset       │ datagen/generate.py             │ 31   │
├─────────────────────────────┼─────────────────────────────────┼──────┤
│ G10 — Failure clustering    │ analysis/failure_clustering.py  │ 15
├─────────────────────────────┼─────────────────────────────────┼──────┤
│ G12 — Run/prompt diff       │ analysis/run_diff.py            │ 14   │
├─────────────────────────────┼─────────────────────────────────┼──────┤
│ G3 — Özel metrik yazımı     │ evaluators/custom_metric.py     │ 18
├─────────────────────────────┼─────────────────────────────────┼──────┤
│ G7 — Arena/Elo leaderboard  │ analysis/arena_elo.py           │ 16
├─────────────────────────────┼─────────────────────────────────┼──────┤
│ G8 — RAG bileşen-level eval │ analysis/rag_eval.py            │ 18
├─────────────────────────────┼─────────────────────────────────┼──────┤
│ Toplam                      │                                 │ 140
└─────────────────────────────┴─────────────────────────────────┴──────┘

Tüm testler:
pytest test_ci_gate_contracts.py test_significance_contracts.py test_dataset_normalization_contracts.py test_synthetic_dataset_contracts.py test_failure_clustering_contracts.py test_run_diff_contracts.py test_custom_metric_contracts.py test_arena_elo_contracts.py test_rag_eval_contracts.py
→ 140 passed

---
Kalan maddeler (öncelik sırası)

┌───────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────┐
│ Madde │                                                        Açıkl                              │ Zorluk │
├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ G1 ⭐ │ Online eval + trace ingest — @eval.trace decorator + OTLP endpoint. API layer'a dokunuyor. En yüksek retention değeri. │ Yüksek │
├───────┼───────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ G9    │ Conversation simulator / sentetik kullanıcı. Multi-turn agent eval. Standalone olabilir.                               │ Orta   │
├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ G11   │ Live agent trace terminal (UI hero feature). Frontend ağırlıklı.                                                       │ Orta   │
├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ G13   │ Paylaşılabilir public rapor linki + embed leaderboard.                                    │ Orta   │
├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ G14   │ Prompt playground + Experiments.                                                          │ Yüksek │
├───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ G15   │ Otomatik red-team / adversarial evrim.                                                    │ Yüksek │
└───────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────┘

---
Kritik pattern (bozma!)

Report contract — tüm modüller buna uyar:
report["models"][model_key]["tests"][test_name]["results"] → list of cases
each case: {"case_id", "scores": dict, "latency", "category", "question"/"input_text"/"prompt", "error"(optional)}
report["models"][model_key]["overall_metrics"]["weighted_score"]
report["summary"]["model_comparison"][model_key]

Standalone module pattern — tüm yeni modüller:
- api/, utils/, adapters/ importu YOK
- Injectable callables: llm_fn(messages) -> str, embed_fn(texts) -> li
- Argparse CLI: python -m analysis.<module> ...
- Offline contract tests: test_<module>_contracts.py

Namespace: datagen/ (HuggingFace datasets çakışması nedeniyle datasets/ → datagen/ rename yapıldı)

---
Çalışma kuralları (CLAUDE.md'den)

- Token disiplini: Broad repo scan yok. codegraph context "..." kullan
- Haiku subagent: Küçük araştırma & geliştirme için model=haiku. Ana agent sadece test & onay.
- Daily log: Her onaylı değişikten sonra logs/development-26-06-2026.md'ye ekle (4 cümle).
- Caveman mode: Kısa, dense, fluff yok.
- Kırılma yok: Her G sonrası tam suite çalıştır, 140 test geçmeli.

---
Sonraki adım

En yüksek değer: G1 (online trace ingest). Ama API layer'a dokunuyor.
Daha güvenli başlangıç: G9 (conversation simulator) — standalone pattern'e oturuyor.

Kullanıcı "devam et" derse G9'dan başla. G1 için önce api/ mimarisini codegraph context "trace ingest OTLP" ile incele, sonra Haiku'ya delege et.