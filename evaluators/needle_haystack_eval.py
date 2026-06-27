"""
Needle in Haystack Evaluator
Uzun context'lerde kritik bilginin bulunmasını test eder.
Paul Graham'ın "Needle in Haystack" testinin Türkçe adaptasyonu.
"""
import json
import random
from typing import Dict, Any, List, Optional, Tuple
from adapters.unified_adapter import UnifiedLLMAdapter


class NeedleInHaystackEvaluator:
    """
    Evaluate model's ability to find specific information (needle) 
    in long context (haystack).
    """
    
    # Context templates for different domains
    CONTEXT_TEMPLATES = {
        "financial_services_info": [
            "Bankamız müşterilerine geniş bir ürün yelpazesi sunmaktadır. Bireysel bankacılık hizmetlerimiz arasında hesap işlemleri, kredi kartları, tüketici kredileri ve yatırım ürünleri bulunmaktadır.",
            "Dijital bankacılık platformumuz 7/24 hizmet vermektedir. Mobil uygulamamız üzerinden para transferi, fatura ödeme, kredi başvurusu gibi işlemler kolaylıkla yapılabilir.",
            "Kurumsal müşterilerimize özel finansman çözümleri sunuyoruz. Ticari krediler, işletme kredileri, proje finansmanı ve dış ticaret finansmanı hizmetlerimizden yararlanabilirsiniz.",
            "Müşteri memnuniyeti bizim için önceliklidir. Şubelerimiz ve çağrı merkezimiz her zaman hizmetinizdedir. Sorun ve önerilerinizi bize iletebilirsiniz.",
            "Yatırım danışmanlığı hizmetlerimiz ile tasarruflarınızı değerlendirebilirsiniz. Uzman kadromuz piyasa analizi ve portföy yönetimi konularında size destek sağlar."
        ],
        "annual_report": [
            "Şirketimiz 2025 yılını başarılı bir şekilde tamamlamıştır. Gelir artışımız bir önceki yıla göre %18 olmuştur.",
            "Operasyonel verimliliğimizi artırmaya yönelik yatırımlar yapmaya devam ediyoruz. Dijital dönüşüm projelerimiz planlandığı gibi ilerlemektedir.",
            "İnsan kaynaklarımıza önem veriyoruz. Çalışan sayımız 1.250'ye ulaşmış ve eğitim programlarımız genişletilmiştir.",
            "Pazarda rekabet avantajımızı koruyoruz. Yeni ürün lansmanlarımız müşterilerimiz tarafından olumlu karşılanmıştır.",
            "Sürdürülebilirlik ilkelerimize bağlı kalarak çevre dostu uygulamalar geliştiriyoruz. Karbon ayak izimizi azaltmak için çeşitli projeler yürütüyoruz."
        ],
        "security_manual": [
            "Güvenlik prosedürleri tüm personel tarafından eksiksiz uygulanmalıdır. Bilgi güvenliği politikalarımız ISO 27001 standartlarına uygundur.",
            "Fiziksel güvenlik önlemleri 24 saat aktiftir. Giriş çıkışlar kayıt altına alınır ve yetkilendirme sistemleri kullanılır.",
            "Siber güvenlik ekibimiz sürekli olarak sistemlerimizi izlemektedir. Olası tehditler anında tespit edilip müdahale edilir.",
            "Veri yedekleme işlemleri günlük olarak gerçekleştirilir. Felaket kurtarma planlarımız düzenli olarak test edilmektedir.",
            "Personel eğitimleri periyodik olarak yapılır. Güvenlik farkındalığı seminerleri ve simülasyon tatbikatları düzenlenmektedir."
        ],
        "project_documentation": [
            "Proje kapsamı ve hedefleri paydaşlar ile birlikte belirlenmiştir. İş planı detaylı olarak hazırlanmış ve onaylanmıştır.",
            "Ekip üyeleri görevlendirilmiş ve sorumluluklar tanımlanmıştır. Proje yöneticisi Sayın Kemal Şahin olarak atanmıştır.",
            "Bütçe planlaması tamamlanmış ve kaynak tahsisleri yapılmıştır. Mali kontrol mekanizmaları devreye alınmıştır.",
            "Risk analizi gerçekleştirilmiş ve azaltma stratejileri geliştirilmiştir. Acil durum planları hazır tutulmaktadır.",
            "Kalite standartları belirlenmiş ve test süreçleri tanımlanmıştır. Düzenli raporlama yapılacaktır."
        ],
        "marketing_campaign": [
            "Yeni kampanyamız tüm segmentlerde büyük ilgi görmektedir. Sosyal medya erişimimiz %45 artmıştır.",
            "Hedef kitlemiz 25-45 yaş arası kentli profesyonellerdir. Dijital kanallar üzerinden aktif iletişim kurmaktayız.",
            "Kampanya bütçesi optimize edilmiş ve ROI hedeflerimiz netleştirilmiştir. Performans metrikleri günlük takip edilmektedir.",
            "İçerik stratejimiz müşteri geri bildirimleri doğrultusunda şekillendirilmektedir. Video içerikler öne çıkmaktadır.",
            "Influencer işbirlikleri ve partnerlikler devam etmektedir. Marka bilinirliğimiz artış göstermektedir."
        ],
        "branch_information": [
            "Şubelerimiz tüm Türkiye'de stratejik lokasyonlarda hizmet vermektedir. Modern ve konforlu ortamlarımızda müşterilerimizi ağırlıyoruz.",
            "Çalışma saatlerimiz hafta içi 09:00-17:00, Cumartesi 09:00-13:00 şeklindedir. Özel randevu sistemi mevcuttur.",
            "Şube personelimiz profesyonel eğitim almıştır. Müşteri temsilcilerimiz her türlü bankacılık işleminizde size yardımcı olur.",
            "Self servis ATM ve kiosk cihazlarımız 7/24 hizmet verir. Para çekme, yatırma ve fatura ödeme işlemlerini kolayca yapabilirsiniz.",
            "Engelli erişimine uygun altyapımız mevcuttur. Tüm müşterilerimize eşit hizmet sunmayı hedefliyoruz."
        ],
        "company_directory": [
            "Organizasyon yapımız fonksiyonel ve çevik bir şekilde tasarlanmıştır. Departmanlarımız birbirleriyle koordineli çalışmaktadır.",
            "Yönetim kurulumuz deneyimli profesyonellerden oluşmaktadır. Stratejik kararlar şeffaf bir süreçle alınır.",
            "İnsan Kaynakları departmanımız işe alım ve eğitim süreçlerini yönetir. Kariyer gelişim programlarımız mevcuttur.",
            "Bilgi Teknolojileri ekibimiz sistemlerimizin kesintisiz çalışmasını sağlar. Yeni teknolojileri takip eder ve entegre ederiz.",
            "Müşteri Hizmetleri ekibimiz geri bildirimleri değerlendirir ve çözüm üretir. Memnuniyet oranımız %92'nin üzerindedir."
        ],
        "emergency_procedures": [
            "Acil durum eylem planları tüm çalışanlara bildirilmiştir. Düzenli tatbikatlar yapılmaktadır.",
            "Yangın söndürme sistemleri ve ekipmanları düzenli kontrolden geçirilir. Çıkış yolları işaretlenmiştir.",
            "Tıbbi acil durumlarda ilk yardım ekibimiz müdahale eder. Ambulans çağrılması için prosedürler tanımlıdır.",
            "Doğal afet durumunda toplanma noktaları belirlenmiştir. İletişim zinciri oluşturulmuştur.",
            "Güvenlik görevlilerimiz 24 saat nöbettedir. Yetkili merciler ile koordinasyon sağlanır."
        ],
        "organizational_chart": [
            "Organizasyon şemasında hiyerarşik yapı net olarak belirlenmiştir. Raporlama ilişkileri tanımlanmıştır.",
            "Üst yönetim stratejik planlama ve koordinasyondan sorumludur. Haftalık yönetim toplantıları yapılır.",
            "Orta kademe yöneticiler operasyonel süreçleri yönetir. Ekip liderleri ile düzenli iletişim halindedirler.",
            "Destek departmanları tüm birimlere hizmet verir. İşbirliği kültürü teşvik edilmektedir.",
            "Performans değerlendirme sistemimiz objektif kriterlere dayanır. Yıllık değerlendirmeler yapılır."
        ],
        "confidential_documents": [
            "Bu doküman gizlilik derecesi taşımaktadır. Yetkisiz erişim yasaktır ve yasal sonuçları vardır.",
            "Proje detayları stratejik öneme sahiptir. Bilgilerin korunması kritik önem taşımaktadır.",
            "Finansal veriler ve tahminler hassas bilgiler içermektedir. Sadece yetkili personel erişebilir.",
            "Teknolojik yenilikler ve patent başvuruları gizli tutulmalıdır. Bilgi sızıntısı önlenmelidir.",
            "Rakip analizi ve pazar stratejileri özel niteliktedir. İç güvenlik prosedürleri uygulanmalıdır."
        ]
    }
    
    # Filler text paragraphs for creating longer contexts
    FILLER_PARAGRAPHS = [
        "Modern iş dünyasında dijital dönüşüm kaçınılmaz bir gereklilik haline gelmiştir. Şirketler rekabet avantajı elde etmek için teknolojiye yatırım yapmaktadır.",
        "Müşteri deneyimi odaklı yaklaşımlar başarının anahtarıdır. Müşteri memnuniyetini artırmak için sürekli iyileştirme çalışmaları yapılmaktadır.",
        "Sürdürülebilir büyüme için inovasyon şarttır. Ar-Ge yatırımları ile yeni ürün ve hizmetler geliştirilmektedir.",
        "Veri analitiği ile karar alma süreçleri güçlendirilmektedir. Büyük veri ve yapay zeka teknolojileri kullanılmaktadır.",
        "Küresel pazarlarda rekabet edebilmek için yerel ve uluslararası normları bilmek önemlidir. Uyum ve şeffaflık esastır.",
        "İnsan kaynağı şirketlerin en değerli varlığıdır. Çalışan mutluluğu ve gelişimi önceliklidir.",
        "Operasyonel mükemmellik için süreç optimizasyonu yapılmaktadır. Verimlilik artışı hedeflenmektedir.",
        "Paydaş ilişkileri sağlam temeller üzerine kurulmalıdır. Karşılıklı güven ve işbirliği önemlidir.",
        "Risk yönetimi proaktif bir yaklaşım gerektirmektedir. Olası tehditler önceden belirlenmeli ve önlem alınmalıdır.",
        "Kalite standartlarına uygunluk sürekli denetim gerektirir. Belgelendirme ve akreditasyon süreçleri takip edilmektedir."
    ]
    
    def __init__(self, judge_adapter: Optional[UnifiedLLMAdapter] = None):
        self.judge = judge_adapter
    
    def generate_haystack(
        self,
        needle: str,
        position: str,
        target_length: str,
        template_name: str = "financial_services_info",
        target_tokens_override: Optional[int] = None
    ) -> Tuple[str, int]:
        """
        Generate a haystack (context) with needle at specified position.
        
        Args:
            needle: The critical information to hide
            position: beginning, early_middle, middle, late_middle, end, deep_middle, scattered
            target_length: short, medium, long, very_long, extreme
            template_name: Template to use for context generation
        
        Returns:
            (haystack_text, needle_char_position)
        """
        # Determine target token count based on length (or explicit override)
        length_tokens = {
            "short": 500,
            "medium": 2000,
            "long": 8000,
            "very_long": 16000,
            "extreme": 32000
        }

        target_tokens = target_tokens_override if target_tokens_override else length_tokens.get(target_length, 2000)
        # Roughly 4 chars per token in Turkish
        target_chars = target_tokens * 4
        
        # Get template paragraphs
        template_paras = self.CONTEXT_TEMPLATES.get(template_name, self.CONTEXT_TEMPLATES["financial_services_info"])
        
        # Build haystack
        paragraphs = []
        current_length = 0
        
        # Add template paragraphs first
        for para in template_paras:
            paragraphs.append(para)
            current_length += len(para)
        
        # Add filler paragraphs until we reach target length
        while current_length < target_chars * 0.9:  # Use 90% of target for pre-content
            para = random.choice(self.FILLER_PARAGRAPHS)
            paragraphs.append(para)
            current_length += len(para)
        
        # Shuffle paragraphs
        random.shuffle(paragraphs)
        
        # Determine needle insertion position
        total_paras = len(paragraphs)
        position_map = {
            "beginning": 0,
            "early_middle": total_paras // 4,
            "middle": total_paras // 2,
            "late_middle": (total_paras * 3) // 4,
            "end": total_paras,
            "deep_middle": total_paras // 2,
            "deep_end": max(0, total_paras - max(1, total_paras // 10))
        }
        
        insert_index = position_map.get(position, total_paras // 2)
        
        # Insert needle
        paragraphs.insert(insert_index, needle)
        
        # Join all paragraphs
        haystack = "\n\n".join(paragraphs)
        
        # Calculate needle position in characters
        needle_char_pos = haystack.find(needle)
        
        return haystack, needle_char_pos
    
    def evaluate_needle_finding(
        self,
        adapter: UnifiedLLMAdapter,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate model's ability to find needle in haystack.
        
        Returns:
            {
                "success": bool,
                "needle_found": bool,
                "response": str,
                "expected_answer": str,
                "context_length": int (chars),
                "needle_position": int (chars),
                "needle_position_percent": float,
                "accuracy_score": float
            }
        """
        # Generate haystack
        if "needles" in test_case:
            # Multi-needle case
            return self._evaluate_multi_needle(adapter, test_case)
        
        needle = test_case["needle"]
        position = test_case["needle_position"]
        length = test_case["context_length"]
        template = test_case.get("context_template", "financial_services_info")
        target_tokens_override = test_case.get("target_tokens")
        
        haystack, needle_char_pos = self.generate_haystack(
            needle,
            position,
            length,
            template,
            target_tokens_override=target_tokens_override
        )
        
        # Prepare prompt
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        
        messages = [
            {
                "role": "system",
                "content": "Sen yardımcı bir asistansın. Verilen bilgiler doğrultusunda soruları yanıtla. Sadece verilen bilgileri kullan."
            },
            {
                "role": "user",
                "content": f"Aşağıdaki bilgileri oku:\n\n{haystack}\n\nSoru: {question}"
            }
        ]
        
        # Get model response
        try:
            result = adapter.generate(messages, temperature=0.0, max_tokens=200)
            response = result.get("content", "")
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "needle_found": False,
                "test_id": test_case.get("id", "unknown")
            }
        
        # Check if needle information was found
        needle_found = self._check_answer_contains_expected(response, expected_answer)
        
        # Calculate metrics
        context_length = len(haystack)
        needle_position_percent = (needle_char_pos / context_length * 100) if context_length > 0 else 0
        
        # Calculate accuracy score
        accuracy_score = 1.0 if needle_found else 0.0
        
        # If judge is available, use it for more nuanced scoring
        if self.judge and needle_found:
            judge_score = self._judge_answer_quality(question, response, expected_answer)
            accuracy_score = judge_score
        
        return {
            "success": True,
            "needle_found": needle_found,
            "response": response,
            "expected_answer": expected_answer,
            "context_length": context_length,
            "context_length_tokens": context_length // 4,  # Rough estimate
            "needle_position": needle_char_pos,
            "needle_position_percent": needle_position_percent,
            "accuracy_score": accuracy_score,
            "test_id": test_case.get("id", "unknown"),
            "difficulty": test_case.get("difficulty", "unknown")
        }
    
    def _evaluate_multi_needle(
        self,
        adapter: UnifiedLLMAdapter,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate finding multiple needles in haystack"""
        needles = test_case["needles"]
        position = test_case["needle_position"]
        length = test_case["context_length"]
        template = test_case.get("context_template", "financial_services_info")
        target_tokens_override = test_case.get("target_tokens")
        
        # Build haystack with multiple needles scattered
        template_paras = self.CONTEXT_TEMPLATES.get(template, self.CONTEXT_TEMPLATES["financial_services_info"])
        
        length_tokens = {"short": 500, "medium": 2000, "long": 8000, "very_long": 16000, "extreme": 32000}
        target_tokens = target_tokens_override if target_tokens_override else length_tokens.get(length, 2000)
        target_chars = target_tokens * 4
        
        paragraphs = list(template_paras)
        current_length = sum(len(p) for p in paragraphs)
        
        # Add filler
        while current_length < target_chars * 0.9:
            para = random.choice(self.FILLER_PARAGRAPHS)
            paragraphs.append(para)
            current_length += len(para)
        
        # Insert needles at different positions
        random.shuffle(paragraphs)
        total = len(paragraphs)
        positions = [total // 4, total // 2, (total * 3) // 4]
        
        for i, needle in enumerate(needles):
            if i < len(positions):
                paragraphs.insert(positions[i], needle)
        
        haystack = "\n\n".join(paragraphs)
        
        # Test
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        
        messages = [
            {"role": "system", "content": "Sen yardımcı bir asistansın. Verilen bilgiler doğrultusunda soruları yanıtla."},
            {"role": "user", "content": f"Aşağıdaki bilgileri oku:\n\n{haystack}\n\nSoru: {question}"}
        ]
        
        try:
            result = adapter.generate(messages, temperature=0.0, max_tokens=300)
            response = result.get("content", "")
        except Exception as e:
            return {"success": False, "error": str(e), "needle_found": False, "test_id": test_case.get("id")}
        
        # Check how many needles were found
        needles_found = sum(1 for needle_info in expected_answer.split(",") if needle_info.strip().lower() in response.lower())
        total_needles = len(needles)
        
        needle_found = needles_found == total_needles
        accuracy_score = needles_found / total_needles if total_needles > 0 else 0.0
        
        return {
            "success": True,
            "needle_found": needle_found,
            "needles_found": needles_found,
            "total_needles": total_needles,
            "response": response,
            "expected_answer": expected_answer,
            "context_length": len(haystack),
            "accuracy_score": accuracy_score,
            "test_id": test_case.get("id", "unknown"),
            "difficulty": test_case.get("difficulty", "unknown")
        }
    
    def _check_answer_contains_expected(self, response: str, expected: str) -> bool:
        """Check if response contains expected information"""
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Try exact match
        if expected_lower in response_lower:
            return True
        
        # Try fuzzy match - check if key parts are present
        expected_parts = expected_lower.split()
        matches = sum(1 for part in expected_parts if len(part) > 3 and part in response_lower)
        
        # Consider it found if most key parts are present
        return matches >= len(expected_parts) * 0.7
    
    def _judge_answer_quality(
        self,
        question: str,
        response: str,
        expected: str
    ) -> float:
        """Use LLM judge to score answer quality"""
        judge_prompt = f"""
Verilen soru ve beklenen cevaba göre modelin yanıtını değerlendir.

Soru: {question}
Beklenen Cevap: {expected}
Model Yanıtı: {response}

Yanıt doğru bilgiyi içeriyor mu? 0.0-1.0 arası skor ver.
JSON formatında yanıt ver: {{"score": <0.0-1.0>, "reasoning": "<açıklama>"}}
"""
        
        messages = [
            {"role": "system", "content": "Sen objektif bir değerlendirme uzmanısın."},
            {"role": "user", "content": judge_prompt}
        ]
        
        try:
            result = self.judge.generate(messages, temperature=0.0, max_tokens=200)
            response_text = result.get("content", "")
            
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("score", 0.5)
        except Exception:
            pass
        
        return 0.5


def evaluate_needle_in_haystack(
    adapter: UnifiedLLMAdapter,
    test_cases: List[Dict[str, Any]],
    judge_adapter: Optional[UnifiedLLMAdapter] = None
) -> Dict[str, Any]:
    """
    Run needle in haystack evaluation on multiple test cases.
    
    Returns comprehensive results including success rate by position and length.
    """
    evaluator = NeedleInHaystackEvaluator(judge_adapter)
    
    results = []
    total_score = 0.0
    found_count = 0
    
    # Track by difficulty and position
    by_difficulty = {}
    by_position = {}
    by_length = {}
    
    for i, test_case in enumerate(test_cases):
        print(f"Running needle test {i+1}/{len(test_cases)}: {test_case.get('id', 'unknown')}")
        
        result = evaluator.evaluate_needle_finding(adapter, test_case)
        results.append(result)
        
        if result.get("success"):
            total_score += result["accuracy_score"]
            if result["needle_found"]:
                found_count += 1
            
            # Categorize
            difficulty = result["difficulty"]
            position = test_case.get("needle_position", "unknown")
            length = test_case.get("context_length", "unknown")
            
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = {"total": 0, "found": 0}
            by_difficulty[difficulty]["total"] += 1
            if result["needle_found"]:
                by_difficulty[difficulty]["found"] += 1
            
            if position not in by_position:
                by_position[position] = {"total": 0, "found": 0}
            by_position[position]["total"] += 1
            if result["needle_found"]:
                by_position[position]["found"] += 1
            
            if length not in by_length:
                by_length[length] = {"total": 0, "found": 0}
            by_length[length]["total"] += 1
            if result["needle_found"]:
                by_length[length]["found"] += 1
    
    total_tests = len(results)
    
    return {
        "test_results": results,
        "summary": {
            "total_tests": total_tests,
            "needles_found": found_count,
            "success_rate": found_count / total_tests if total_tests > 0 else 0.0,
            "average_score": total_score / total_tests if total_tests > 0 else 0.0,
            "by_difficulty": by_difficulty,
            "by_position": by_position,
            "by_length": by_length
        }
    }
