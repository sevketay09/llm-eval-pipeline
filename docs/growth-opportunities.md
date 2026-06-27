# Geliştirme Fırsatları — LLM Eval Pipeline

> Amaç: teknik borç değil. Bu ürünü **insanların her gün açtığı** bir eval platformuna çevirecek yetenekler.
> Mevcut backlog (`docs/backlog.md`) çoğunlukla iç mimari konsolidasyonu. Bu doküman onun **üstüne**, çekicilik/adoption odaklı yönleri ekler.

---

## 1. Şu an ne var (özet)

- **Çok sağlayıcılı adapter**: Azure / OpenAI / Anthropic / vLLM / Ollama / LM Studio.
- **~25 evaluator**: judge (kategorik), G-Eval, NLP (ROUGE/BLEU/BERTScore), Azure agentic pack, embedding (STS/retrieval/clustering), RAG faithfulness, function calling, multi-turn, PII, adversarial, benchmark (MMLU/GSM8K/HumanEval...).
- **Olgun çekirdek**: ortak metric sözleşmesi, typed case/result modelleri, trace-span modeli, agent/multi-turn/tool metric pack'leri, JSON correctness + prompt alignment.
- **HITL**: review queue, annotation, policy audit trail.
- **Trend/regression + version metadata + reproducibility hash + cache.**
- **Ön yüz (React)**: Dashboard, RunEvaluation (WS progress), Results, HitlReview, Models, DatasetStudio.
- **CLI + API + UI** aynı çekirdeği çağırıyor.

**Tek cümle teşhis:** Güçlü bir **offline, batch, model-vs-model** karşılaştırma motoru. Eksik olan: kendi **LLM uygulamanı** (RAG/agent) değerlendirme, **canlı** kullanım, ve **viral/yapışkan** yüzeyler.

---

## 2. En yüksek kaldıraçlı yönler (öncelik sırası)

### TIER 1 — Yapışkanlık & aktivasyon (önce bunlar)

**G1. Online eval + trace ingestion (observability'ye geçiş)** ⭐ en kritik
- Şu an: yalnız batch. Kullanıcı bir kez koşturup gidiyor → retention sıfır.
- Ekle: hafif Python SDK / decorator (`@eval.trace`) + OpenTelemetry/OTLP ingest endpoint. Kullanıcı **kendi prod uygulamasını** instrument eder, canlı trace'leri gönderir, örnekleme ile online skorlar.
- Neden yapışkan: araç "tek seferlik test"ten "her gün bakılan dashboard"a döner. Langfuse/Phoenix/Braintrust'ın tüm büyümesi bu eksende.

**G2. Eval-as-CI: pytest plugin + GitHub Action + badge**
- Regression altyapısı zaten var; geliştiricinin **workflow'una** koymuyor.
- Ekle: `pytest` assertion API (`assert_eval(score > 0.8)`), hazır GitHub Action, PR yorumu olarak skor delta + "yeni başarısızlıklar", README badge.
- Neden: eval dev döngüsüne girince acquisition + retention birlikte gelir. Viral (badge).

**G3. UI içinde özel metrik yazımı (no-code + code)**
- En çok istenen şey "BENİM metriğim". Şu an metrikler sabit.
- Ekle: doğal dil → judge prompt üretimi ("yanıtın empatik olup olmadığını 0-1 ver"), birkaç etiketli örnekle otokalibrasyon, kaydet & suite'e ekle. (G-Eval-by-description.)
- Neden: kullanıcı kendi domain'ini ölçebilince ürünü sahiplenir.

**G4. Sentetik dataset üretimi (docs → golden Q/A)** — cold-start öldürücü
- Backlog'da DS-001 var ama ürünleşmemiş. En büyük onboarding sürtünmesi: "elimde test seti yok".
- Ekle: doküman yükle → chunk → otomatik soru/beklenen-cevap üretimi, kaynak attribution, deterministic golden kuralları, multi-turn senaryo üretimi (DS-002).
- Neden: 0 dakikada ilk değerli sonuç. Aktivasyon metriğini uçurur.

### TIER 2 — Güven & farklılaşma

**G5. Judge kalibrasyonu & meta-eval** (RV-003'ü ürünleştir)
- LLM-judge'a güvensizlik #1 itiraz. Şu an judge skoru "kara kutu".
- Ekle: judge-vs-human uyumu (Cohen's kappa), pozisyon/verbosity/self-preference bias tespiti, prompt versiyonları arası judge kıyası, otomatik judge-prompt iyileştirme önerisi.
- Neden: "judge'ına neden güveneyim?" sorusunu cevaplayan ilk Türkçe araç olursun.

**G6. İstatistiksel sıkılık: güven aralığı + anlamlılık testi**
- Şu an "gpt-4o, qwen'i %3 geçti" deniyor — anlamlı mı bilinmiyor.
- Ekle: bootstrap CI, skor farkı için significance testi, örneklem boyutu uyarısı.
- Neden: ciddi değerlendiriciler (araştırma, ML ekipleri) için kredibilite eşiği.

**G7. Arena / pairwise Elo modu** (viral format)
- `run_arena.py` var ama ürün değil.
- Ekle: kör A/B, LLM veya insan judge, Bradley-Terry/Elo leaderboard, kategori bazlı.
- Neden: Chatbot Arena en viral eval formatı; paylaşılabilir leaderboard acquisition getirir.

**G8. RAG bileşen-seviyesi eval (retriever vs generator ayrıştırma)**
- Şu an RAG quality tek skor. RAG ekipleri en büyük eval alıcısı.
- Ekle: context precision/recall, chunk attribution, "hata retriever'da mı generator'da mı" fault isolation, contextual relevancy.
- Neden: RAG borç ödeyen segment; net teşhis = net değer.

**G9. Konuşma simülatörü / sentetik kullanıcı (agent eval ölçekleme)**
- Multi-turn datasetleri elle yazmak pahalı.
- Ekle: persona tanımlı simüle kullanıcı, agent ile N tur konuşur, tüm trajectory değerlendirilir (goal completion, tur sayısı, sapma).
- Neden: agent-eval'in frontier'ı; elle dataset yazmadan derin multi-turn kapsama.

### TIER 3 — İçgörü & ön yüz cazibesi

**G10. Failure clustering / otomatik taksonomi**
- Şu an başarısızlıklar liste. İçgörü yok.
- Ekle: fail case'leri embed → kümele → LLM ile etiketle: "başarısızlıkların %37'si sayısal hesap hatası".
- Neden: ham skoru aksiyona çeviren "aha" anı. Standout analytics.

**G11. Live agent trace terminal'i (UI-002) — hero feature yap**
- Span ağacı, step badge (AGENT/TOOL/LLM/RETRIEVER), expand/collapse, raw payload drawer, failed-step highlight, replay.
- Neden: agent'ı görsel debug etmek mıknatıs; demo'da "wow" yaratır.

**G12. Run/prompt diff görünümü (case-level kırmızı/yeşil)**
- İki run veya iki prompt versiyonu arasında git-diff hissi; hangi case düzeldi/bozuldu.
- Neden: iterasyonu hızlandırır → günlük kullanım.

**G13. Paylaşılabilir public rapor linki + embed leaderboard**
- Read-only share URL, social-card preview, README'ye gömülebilir leaderboard.
- Neden: insanlar leaderboard'unu paylaşır = bedava acquisition.

**G14. Prompt playground + Experiments (versiyonlama + bağlı eval)**
- Prompt v1 vs v2'yi aynı dataset'te koştur, yan yana diff, hangi case'i kim regress etti.
- Neden: Braintrust'ın çekirdek kancası; prompt yönetimi + eval = günlük araç.

**G15. Otomatik red-team / adversarial evrim**
- Şu an statik adversarial set. Ekle: başarılı jailbreak'leri evrimleştiren auto red-teaming.
- Neden: güvenlik ekipleri sever, PR/demo değeri yüksek.

---

## 3. Konumlandırma hamlesi (ücretsiz ama büyük)

- Şu anki çerçeve: "Türkçe odaklı model karşılaştırma".
- Önerilen: **"her LLM uygulaması için eval + observability"** — Türkçe-native eval'i **niş üstünlük** olarak koru (gerçekten nadir), ama yatay genişle.
- Türkçe-native eval (Türkçe judge rubric'leri, dilbilgisi/nüans/regülasyon datasetleri) = pazarda **savunulabilir tek farklılaşma**. Bunu öne çıkar, gömme.

---

## 4. Önerilen ilk dalga (3 hamle)

1. **G4 (sentetik dataset)** + **G2 (CI/pytest)** → aktivasyon + workflow girişi. Cold-start'ı öldürür.
2. **G1 (online trace ingest)** → tek-seferlik araçtan platforma. Retention'ın anahtarı.
3. **G5 (judge kalibrasyon)** + **G13 (paylaşılabilir rapor)** → güven + viralite.

> G3 (özel metrik) ve G10 (failure clustering) hızlı "wow" için ikinci dalga.

**G6 (istatistiksel sıkılık)** artık hayata geçirildi: `analysis/significance.py` + CLI + `test_significance_contracts.py` (10 test).
