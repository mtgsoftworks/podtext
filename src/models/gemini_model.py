"""
Gemini LLM analiz modülü - Google Gemini API entegrasyonu
"""
import google.generativeai as genai
from typing import Dict, List, Optional, Any
import json
import re
from ..utils.logger import LoggerMixin
from ..utils.config import Config
import os
try:
    import openai  # DeepSeek R-1 modeli OpenAI uyumlu istemci kullanır
except ImportError:  # Paket eksikse geçici olarak None ata; çalışma sırasında hata durumunda bildirilir
    openai = None


# ------------------ SmartModel tanımı ------------------

class _SmartModel:
    """Gemini çağrısı başarısız olursa DeepSeek R-1 modeline otomatik geçiş yapan sarmal model."""

    def __init__(self, gemini_model, deepseek_key: str = None, deepseek_model: str = "deepseek-r-1", temperature: float = 0.3, logger=None):
        self._gemini_model = gemini_model
        self._deepseek_key = deepseek_key
        self._deepseek_model = deepseek_model
        self._temperature = temperature
        self._logger = logger if logger else (lambda *a, **k: None)

        # DeepSeek API ayarları (OpenAI uyumlu endpoint)
        if self._deepseek_key:
            if openai:
                openai.api_key = self._deepseek_key
                openai.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            else:
                self._logger("openai paketi yüklü değil, DeepSeek fallback devre dışı.")

    # Gemini ile aynı interface'i sunar
    def generate_content(self, prompt: str):
        try:
            return self._gemini_model.generate_content(prompt)
        except Exception as gemini_err:
            self._logger(f"Gemini hata verdi, DeepSeek R-1 modeline geçiliyor: {gemini_err}")

            if not self._deepseek_key:
                raise gemini_err  # DeepSeek anahtarı yok

            try:
                if not openai:
                    raise RuntimeError("openai paketi bulunamadığı için DeepSeek kullanılamıyor.")

                completion = openai.ChatCompletion.create(
                    model=self._deepseek_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                )

                text = completion.choices[0].message.content if completion.choices else ""

                # Gemini benzeri response yapısı oluştur
                class _Dummy:
                    pass

                resp = _Dummy()
                part_cls = type("Part", (), {"text": text})
                content_cls = type("Content", (), {"parts": [part_cls()]})
                candidate_cls = type("Candidate", (), {"content": content_cls()})
                resp.candidates = [candidate_cls()]
                return resp
            except Exception as deep_err:
                self._logger(f"DeepSeek'de de hata oluştu: {deep_err}")
                raise deep_err


class GeminiAnalyzer(LoggerMixin):
    """Gemini LLM analiz sınıfı"""
    
    def __init__(self, config: Config):
        self.config = config
        self.gemini_config = config.get("gemini", {})
        
        # API anahtarını al ve kur
        api_key = self.config.get_api_key("google_gemini")
        if not api_key:
            raise ValueError("Google Gemini API anahtarı bulunamadı!")
        
        genai.configure(api_key=api_key)
        
        # Model ayarları
        self.model_name = self.gemini_config.get("model", "gemini-1.5-flash")
        
        gemini_primary_model = genai.GenerativeModel(self.model_name)

        # DeepSeek entegrasyonu
        self.deepseek_key = self.config.get_api_key("deepseek")
        deepseek_cfg = self.config.get("deepseek", {})
        self.deepseek_model_name = deepseek_cfg.get("model", "deepseek-r-1")
        self.deepseek_temperature = deepseek_cfg.get("temperature", self.gemini_config.get("temperature", 0.3))

        # SmartModel sarıcı kullan
        self.model = _SmartModel(
            gemini_primary_model,
            deepseek_key=self.deepseek_key,
            deepseek_model=self.deepseek_model_name,
            temperature=self.deepseek_temperature,
            logger=self.log_warning,
        )

        self.log_info(
            f"GeminiAnalyzer başlatıldı. Birincil model: '{self.model_name}', Yedek model: '{self.deepseek_model_name if self.deepseek_key else 'None'}'"
        )
    
    def evaluate_audio_and_transcription_quality(self, transcription_text: str, audio_info: Dict, label_suggestions: List[Dict]) -> Dict:
        """
        Yönergelere göre hem ses hem de transkripsiyon kalitesini analiz eder.

        Args:
            transcription_text: Transkripsiyon metni
            audio_info: Ses dosyası hakkında temel bilgiler
            label_suggestions: AutoLabeler tarafından üretilen [unsure: ] etiket önerileri

        Returns:
            Yönergelere uygun 5 kategoride kalite analizi sonucu
        """
        try:
            self.log_info("Gelişmiş ses ve transkripsiyon kalite analizi başlatılıyor...")
            prompt = self._create_comprehensive_quality_prompt(transcription_text, audio_info, label_suggestions)
            response = self.model.generate_content(prompt)
            
            if getattr(response, 'candidates', None) and response.candidates:
                raw_text = ''.join([p.text for p in response.candidates[0].content.parts])
            else:
                raw_text = ''
            
            analysis = self._parse_json_response(raw_text)
            self.log_info("Gelişmiş kalite analizi tamamlandı.")
            return analysis

        except Exception as e:
            self.log_error(f"Gelişmiş kalite analizi hatası: {e}", exc_info=True)
            return {
                "unclear_audio": {"value": False, "reason": "Analysis failed"},
                "heavy_accent": {"value": False, "reason": "Analysis failed"},
                "incorrect_language": {"value": False, "reason": "Analysis failed"},
                "is_synthesized": {"value": False, "reason": "Analysis failed"},
                "multiple_voices": {"value": False, "reason": "Analysis failed"},
                "error": str(e)
            }

    def correct_proper_nouns(self, text: str) -> str:
        """
        FAQ 9: Özel isimleri POLLY kriterlerine göre düzeltir
        - Celebrities, companies: reliable sources check
        - Regular names: context-based famous person check
        - Uncertain spellings: [unsure: ] tag
        
        Args:
            text: Düzeltilecek transkripsiyon metni

        Returns:
            Özel isimleri düzeltilmiş metin
        """
        try:
            self.log_info("POLLY FAQ 9 uyumlu özel isim düzeltme başlatılıyor...")
            prompt = self._create_polly_proper_noun_prompt(text)
            response = self.model.generate_content(prompt)

            if getattr(response, 'candidates', None) and response.candidates:
                corrected_text = ''.join([p.text for p in response.candidates[0].content.parts])
            else:
                corrected_text = text

            # Model bazen açıklamalar ekleyebilir, sadece metni al
            corrected_text = corrected_text.strip().replace("```", "")

            self.log_info("POLLY uyumlu özel isim düzeltme tamamlandı")
            return corrected_text if corrected_text else text

        except Exception as e:
            self.log_error(f"POLLY özel isim düzeltme hatası: {e}", exc_info=True)
            return text

    def correct_entities_and_abbreviations(self, text: str) -> str:
        """
        POLLY Kurallar 1 ve 2: Kısaltmalar (Abbreviations) ve Baş Harflerden Oluşan Terimler (Acronyms/Initialisms) düzeltir
        
        Kurallar:
        1. Abbreviations:
           - Konuşmacı kısaltma kullanıyorsa, yazımda da kısaltma kullan
           - Sistem doğru kısaltmışsa, olduğu gibi bırak
           - Sistem yanlış yazmışsa düzelt (Dr., Inc., Ave. vb.)
        
        2. Acronyms & Initialisms:
           - BÜYÜK HARFLERLE ve noktasız yazılmalı (NASA, DMV)
           - Çoğul formda "ATM's" değil, "ATMs" yazılmalı
           - Harf harf söylenen ("L O L") birleştirilmeli (LOL)
        
        Args:
            text: Düzeltilecek transkripsiyon metni

        Returns:
            Kısaltmalar ve baş harfler düzeltilmiş metin
        """
        try:
            self.log_info("POLLY kısaltma ve baş harf düzeltme başlatılıyor...")
            prompt = self._create_abbreviation_correction_prompt(text)
            response = self.model.generate_content(prompt)

            if getattr(response, 'candidates', None) and response.candidates:
                corrected_text = ''.join([p.text for p in response.candidates[0].content.parts])
            else:
                corrected_text = text

            # Model bazen açıklamalar ekleyebilir, sadece metni al
            corrected_text = corrected_text.strip().replace("```", "")

            self.log_info("POLLY kısaltma ve baş harf düzeltme tamamlandı")
            return corrected_text if corrected_text else text

        except Exception as e:
            self.log_error(f"POLLY kısaltma düzeltme hatası: {e}", exc_info=True)
            return text
    
    def analyze_polly_quality_flags(self, transcription_text: str, audio_info: Dict) -> Dict:
        """
        POLLY FAQ kriterlerine göre kalite bayraklarını analiz eder
        
        Returns:
            - unclear_audio: FAQ 22 kriterleri
            - heavy_accent: FAQ 15 kriterleri  
            - wrong_language: FAQ 14 kriterleri
            - multiple_voices: FAQ 17 kriterleri
        """
        try:
            self.log_info("POLLY kalite bayrağı analizi başlatılıyor...")
            prompt = self._create_polly_quality_flags_prompt(transcription_text, audio_info)
            response = self.model.generate_content(prompt)
            
            if getattr(response, 'candidates', None) and response.candidates:
                raw_text = ''.join([p.text for p in response.candidates[0].content.parts])
            else:
                raw_text = ''
            
            analysis = self._parse_json_response(raw_text)
            self.log_info("POLLY kalite bayrağı analizi tamamlandı")
            return analysis

        except Exception as e:
            self.log_error(f"POLLY kalite bayrağı analizi hatası: {e}", exc_info=True)
            return {
                "unclear_audio": {"should_flag": False, "reason": "Analysis failed"},
                "heavy_accent": {"should_flag": False, "reason": "Analysis failed"},
                "wrong_language": {"should_flag": False, "reason": "Analysis failed"},
                "multiple_voices": {"should_flag": False, "reason": "Analysis failed"},
                "error": str(e)
            }

    def analyze_transcription_quality(self, transcription_text: str, audio_info: Dict = None) -> Dict:
        """
        Transkripsiyon kalitesini analiz et
        
        Args:
            transcription_text: Transkripsiyon metni
            audio_info: Ses dosyası bilgileri
            
        Returns:
            Kalite analizi sonucu
        """
        try:
            self.log_info("Transkripsiyon kalitesi analiz ediliyor...")
            
            prompt = self._create_quality_analysis_prompt(transcription_text, audio_info)
            
            response = self.model.generate_content(prompt)
            
            # JSON yanıtını parse etmeden önce raw metni al
            if getattr(response, 'candidates', None) and response.candidates:
                parts = response.candidates[0].content.parts
                raw_text = ''.join([p.text for p in parts])
            else:
                raw_text = ''
            analysis = self._parse_json_response(raw_text)
            
            self.log_info("Transkripsiyon kalite analizi tamamlandı")
            return analysis
            
        except Exception as e:
            self.log_error(f"Kalite analizi hatası: {e}")
            return {
                "topics": [],
                "segments": [],
                "summary": "",
                "key_points": [],
                "error": str(e)
            }
    
    def suggest_corrections(self, transcription_text: str) -> Dict:
        """
        Düzeltme önerileri sun
        
        Args:
            transcription_text: Transkripsiyon metni
            
        Returns:
            Düzeltme önerileri
        """
        try:
            self.log_info("Düzeltme önerileri oluşturuluyor...")
            
            prompt = self._create_correction_prompt(transcription_text)
            
            response = self.model.generate_content(prompt)
            
            # raw metni al ve parse et
            if getattr(response, 'candidates', None) and response.candidates:
                parts = response.candidates[0].content.parts
                raw_text = ''.join([p.text for p in parts])
            else:
                raw_text = ''
            corrections = self._parse_json_response(raw_text)
            
            self.log_info(f"{len(corrections.get('corrections', []))} düzeltme önerisi oluşturuldu")
            return corrections
            
        except Exception as e:
            self.log_error(f"Düzeltme önerisi hatası: {e}")
            return {
                "corrections": [],
                "grammar_fixes": [],
                "style_improvements": [],
                "error": str(e)
            }
    
    def detect_speech_issues(self, transcription_text: str) -> Dict:
        """
        Konuşma sorunlarını tespit et
        
        Args:
            transcription_text: Transkripsiyon metni
            
        Returns:
            Tespit edilen sorunlar
        """
        try:
            self.log_info("Konuşma sorunları tespit ediliyor...")
            
            prompt = self._create_speech_issues_prompt(transcription_text)
            
            response = self.model.generate_content(prompt)
            
            # raw metni al ve parse et
            if getattr(response, 'candidates', None) and response.candidates:
                parts = response.candidates[0].content.parts
                raw_text = ''.join([p.text for p in parts])
            else:
                raw_text = ''
            issues = self._parse_json_response(raw_text)
            
            self.log_info("Konuşma sorunları analizi tamamlandı")
            return issues
            
        except Exception as e:
            self.log_error(f"Konuşma sorunları analizi hatası: {e}")
            return {
                "stutters": [],
                "filler_words": [],
                "unclear_speech": [],
                "repetitions": [],
                "error": str(e)
            }
    
    def suggest_labels(self, transcription_text: str, audio_quality: Dict = None) -> Dict:
        """
        Etiket önerileri sun
        
        Args:
            transcription_text: Transkripsiyon metni
            audio_quality: Ses kalitesi bilgileri
            
        Returns:
            Etiket önerileri
        """
        try:
            self.log_info("Etiket önerileri oluşturuluyor...")
            
            prompt = self._create_labeling_prompt(transcription_text, audio_quality)
            
            response = self.model.generate_content(prompt)
            
            # raw metni al ve parse et
            if getattr(response, 'candidates', None) and response.candidates:
                parts = response.candidates[0].content.parts
                raw_text = ''.join([p.text for p in parts])
            else:
                raw_text = ''
            labels = self._parse_json_response(raw_text)
            
            self.log_info(f"{len(labels.get('suggested_labels', []))} etiket önerisi oluşturuldu")
            return labels
            
        except Exception as e:
            self.log_error(f"Etiket önerisi hatası: {e}")
            return {
                "suggested_labels": [],
                "unsure_segments": [],
                "truncated_segments": [],
                "error": str(e)
            }
    
    def analyze_content_structure(self, transcription_text: str) -> Dict:
        """
        İçerik yapısını analiz et
        
        Args:
            transcription_text: Transkripsiyon metni
            
        Returns:
            İçerik yapı analizi
        """
        try:
            self.log_info("İçerik yapısı analiz ediliyor...")
            
            prompt = self._create_structure_analysis_prompt(transcription_text)
            
            response = self.model.generate_content(prompt)
            
            # raw metni al ve parse et
            if getattr(response, 'candidates', None) and response.candidates:
                parts = response.candidates[0].content.parts
                raw_text = ''.join([p.text for p in parts])
            else:
                raw_text = ''
            structure = self._parse_json_response(raw_text)
            
            self.log_info("İçerik yapı analizi tamamlandı")
            return structure
            
        except Exception as e:
            self.log_error(f"İçerik yapı analizi hatası: {e}")
            return {
                "topics": [],
                "segments": [],
                "summary": "",
                "key_points": [],
                "error": str(e)
            }
    
    def _create_quality_analysis_prompt(self, text: str, audio_info: Dict = None) -> str:
        """Transkripsiyon kalitesini analiz et prompt'u oluştur"""
        audio_context = ""
        if audio_info:
            audio_context = f"""
Ses dosyası bilgileri:
- Süre: {audio_info.get('sure_formatli', 'Bilinmiyor')}
- SNR: {audio_info.get('tahmini_snr_db', 'Bilinmiyor')} dB
- Format: {audio_info.get('format', 'Bilinmiyor')}
"""
        
        return f"""
Lütfen **yalnızca** geçerli bir JSON nesnesi olarak yanıt verin. Başka hiçbir metin, kod bloğu veya açıklama eklemeyin.

Aşağıdaki transkripsiyon metninde konuşma sorunlarını tespit edin:

{audio_context}

Transkripsiyon Metni:
{text}

Lütfen aşağıdaki JSON formatında sorunları belirtin:

{{
    "stutters": [
        {{
            "text": "kekeme_kısmı",
            "location": "konum_bilgisi"
        }}
    ],
    "filler_words": [
        {{
            "word": "dolgu_kelime",
            "frequency": "kaç_kez_geçiyor",
            "suggestions": ["alternatif_ifadeler"]
        }}
    ],
    "unclear_speech": [
        {{
            "text": "belirsiz_kısım",
            "reason": "neden_belirsiz"
        }}
    ],
    "repetitions": [
        {{
            "repeated_phrase": "tekrarlanan_ifade",
            "count": "tekrar_sayısı"
        }}
    ]
}}
"""
    
    def _create_correction_prompt(self, text: str) -> str:
        """Düzeltme önerileri prompt'u oluştur"""
        return f"""
Lütfen **yalnızca** geçerli bir JSON nesnesi olarak yanıt verin. Başka hiçbir metin, kod bloğu veya açıklama eklemeyin.

Düzeltme önerileri oluşturmak için aşağıdaki transkripsiyon metnini kullanın:
{text}

Lütfen aşağıdaki JSON formatında düzeltme önerilerini verin:

{{
    "corrections": [
        {{
            "original": "hatalı_metin",
            "corrected": "düzeltilmiş_metin",
            "type": "imla|dilbilgisi|kelime_seçimi|noktalama",
            "explanation": "neden_düzeltilmeli"
        }}
    ],
    "grammar_fixes": [
        {{
            "issue": "dilbilgisi_sorunu",
            "fix": "çözüm_önerisi"
        }}
    ],
    "style_improvements": [
        "stil_geliştirme_önerileri"
    ]
}}
"""
    
    def _create_speech_issues_prompt(self, text: str) -> str:
        """Konuşma sorunları prompt'u oluştur"""
        return f"""
Lütfen **yalnızca** geçerli bir JSON nesnesi olarak yanıt verin. Başka hiçbir metin, kod bloğu veya açıklama eklemeyin.

Aşağıdaki transkripsiyon metninde konuşma sorunlarını tespit edin:
{text}

Lütfen aşağıdaki JSON formatında sorunları belirtin:

{{
    "stutters": [
        {{
            "text": "kekeme_kısmı",
            "location": "konum_bilgisi"
        }}
    ],
    "filler_words": [
        {{
            "word": "dolgu_kelime",
            "frequency": "kaç_kez_geçiyor",
            "suggestions": ["alternatif_ifadeler"]
        }}
    ],
    "unclear_speech": [
        {{
            "text": "belirsiz_kısım",
            "reason": "neden_belirsiz"
        }}
    ],
    "repetitions": [
        {{
            "repeated_phrase": "tekrarlanan_ifade",
            "count": "tekrar_sayısı"
        }}
    ]
}}
"""
    
    def _create_labeling_prompt(self, text: str, audio_quality: Dict = None) -> str:
        """Etiketleme önerisi prompt'u oluştur"""
        quality_context = ""
        if audio_quality:
            quality_context = f"""
Ses kalitesi bilgileri:
{json.dumps(audio_quality, indent=2, ensure_ascii=False)}
"""
        
        return f"""
Lütfen **yalnızca** geçerli bir JSON nesnesi olarak yanıt verin. Başka hiçbir metin, kod bloğu veya açıklama eklemeyin.

{quality_context}
Transkripsiyon Metni:
{text}

Lütfen aşağıdaki JSON formatında etiket önerilerini verin:

{{
    "suggested_labels": [
        {{
            "text": "etiketlenecek_kısım",
            "label": "unsure|truncated|inaudible|overlap",
            "reason": "neden_bu_etiket",
            "confidence": 0.0-1.0_güven_skoru
        }}
    ],
    "unsure_segments": [
        "belirsiz_olan_kısımlar"
    ],
    "truncated_segments": [
        "kesilmiş_olan_kısımlar"
    ]
}}
"""
    
    def _create_structure_analysis_prompt(self, text: str) -> str:
        """İçerik yapısı analiz prompt'u oluştur"""
        return f"""
Lütfen **yalnızca** geçerli bir JSON nesnesi olarak yanıt verin. Başka hiçbir metin, kod bloğu veya açıklama eklemeyin.

Aşağıdaki transkripsiyon metninin içerik yapısını analiz edin:
{text}

Lütfen aşağıdaki JSON formatında analiz sonucunu verin:

{{
    "topics": [
        {{
            "topic": "konu_başlığı",
            "start_position": "yaklaşık_başlangıç",
            "importance": "yüksek|orta|düşük"
        }}
    ],
    "segments": [
        {{
            "type": "giriş|ana_içerik|sonuç|soru_cevap",
            "content_summary": "bu_kısmın_özeti"
        }}
    ],
    "summary": "genel_özet",
    "key_points": [
        "önemli_noktalar"
    ],
    "estimated_structure": "röportaj|sunum|sohbet|ders|diğer"
}}
"""
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """JSON yanıtını parse et"""
        try:
            # Markdown code block'larını temizle
            clean_text = re.sub(r'```json\s*|\s*```', '', response_text)
            clean_text = clean_text.strip()
            
            # JSON parse et
            return json.loads(clean_text)
            
        except json.JSONDecodeError as e:
            # JSON parse hatasını debug seviyesinde logluyoruz, uyarı gösterilmesini engelliyoruz
            self.log_debug(f"JSON parse hatası (gizlendi): {e}")
            
            # Basit fallback - metin olarak döndür (yalnızca raw_response içerir)
            return {
                "raw_response": response_text,
                "parse_error": str(e)
            }
    
    def batch_analyze(self, texts: List[str], analysis_type: str = "quality") -> List[Dict]:
        """
        Toplu analiz
        
        Args:
            texts: Analiz edilecek metin listesi
            analysis_type: Analiz tipi (quality, corrections, issues, labels)
            
        Returns:
            Analiz sonuçları listesi
        """
        results = []
        
        for i, text in enumerate(texts):
            self.log_info(f"Toplu analiz: {i+1}/{len(texts)}")
            
            if analysis_type == "quality":
                result = self.analyze_transcription_quality(text)
            elif analysis_type == "corrections":
                result = self.suggest_corrections(text)
            elif analysis_type == "issues":
                result = self.detect_speech_issues(text)
            elif analysis_type == "labels":
                result = self.suggest_labels(text)
            else:
                result = {"error": f"Bilinmeyen analiz tipi: {analysis_type}"}
            
            results.append(result)
        
        return results
    
    def get_analysis_summary(self, analysis_results: List[Dict]) -> Dict:
        """
        Analiz sonuçlarının özetini çıkar
        
        Args:
            analysis_results: Analiz sonuçları listesi
            
        Returns:
            Özet bilgiler
        """
        if not analysis_results:
            return {"error": "Analiz sonucu bulunamadı"}
        
        # Genel istatistikler
        total_analyses = len(analysis_results)
        successful_analyses = len([r for r in analysis_results if "error" not in r])
        
        # Kalite dağılımı (eğer kalite analizi varsa)
        quality_distribution = {}
        for result in analysis_results:
            quality = result.get("overall_quality")
            if quality:
                quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        return {
            "toplam_analiz": total_analyses,
            "basarili_analiz": successful_analyses,
            "basari_orani": (successful_analyses / total_analyses) * 100,
            "kalite_dagilimi": quality_distribution
        }
    
    def detect_accent(self, transcription_text: str) -> Dict:
        """
        Aksan tespiti yapar: transkripsiyon metninden konuşmacının ağır aksanı olup olmadığını belirler
        Returns:
            {"heavy_accent": bool}
        """
        try:
            self.log_info("Aksan tespiti yapılıyor...")
            prompt = f"""
Lütfen **yalnızca** geçerli bir JSON nesnesi olarak yanıt verin. Başka metin, kod bloğu veya açıklama eklemeyin.

Aşağıdaki transkripsiyon metnindeki konuşmacının ağır aksanı olup olmadığını true/false olarak belirtin:
"""
            prompt += f"""
Transkripsiyon Metni:
{transcription_text}

Yanıt formatı:
{{
    "heavy_accent": true/false
}}
"""
            response = self.model.generate_content(prompt)
            if getattr(response, 'candidates', None) and response.candidates:
                parts = response.candidates[0].content.parts
                raw_text = ''.join(p.text for p in parts)
            else:
                raw_text = ''
            result = self._parse_json_response(raw_text)
            self.log_info("Aksan tespiti tamamlandı")
            return result
        except Exception as e:
            self.log_error(f"Aksan tespiti hatası: {e}")
            return {"heavy_accent": False}

    def _create_comprehensive_quality_prompt(self, transcription_text: str, audio_info: Dict, label_suggestions: List[Dict]) -> str:
        """Kapsamlı kalite değerlendirmesi için Gemini'ye gönderilecek prompt'u oluşturur."""
        audio_context = f"""
Ses dosyası teknik bilgileri:
- Süre: {audio_info.get('sure_formatli', 'Bilinmiyor')} saniye
- Tahmini SNR: {audio_info.get('tahmini_snr_db', 'Bilinmiyor'):.1f} dB
- Clipping Oranı: {audio_info.get('clipping_ratio', 0.0) * 100:.1f}%
"""
        
        unsure_tags_context = "Transkripsiyonda [unsure: ] etiketi gerektiren hiçbir sorun bulunmadı."
        if label_suggestions:
            reasons = [f"- '{s.get('text')}' (sebep: {s.get('reason')})" for s in label_suggestions if s.get('suggested_label', '').startswith('[unsure')]
            if reasons:
                unsure_tags_context = "Transkripsiyonda [unsure: ] etiketi gerektiren şu sorunlar tespit edildi:\n" + "\n".join(reasons)

        return f"""
Lütfen **yalnızca** aşağıda belirtilen yapıya sahip tek bir geçerli JSON nesnesi döndür. Başka hiçbir metin, markdown (```json) veya açıklama ekleme.

GÖREV: Sağlanan transkripsiyon metnini, sesin teknik bilgilerini ve tespit edilen `[unsure: ]` etiketlerini analiz ederek, 5 kalite kategorisi için bir değerlendirme yap. Her kategori için 'value' (true/false) ve kısa bir 'reason' (sebep) belirt.

METİN:
"{transcription_text}"

{audio_context}

TESPİT EDİLEN [unsure: ] ETİKETLERİ VE NEDENLERİ:
{unsure_tags_context}


KATEGORİLER VE KURALLAR:

1.  "unclear_audio": Ana konuşmacının konuşmasının bir kısmını veya tamamını, aşağıdaki nedenlerle güvenle yazıya dökemiyorsan 'true' yap:
    - Yüksek arkaplan gürültüsü (TV, radyo, makine sesi).
    - Arka planda konuşan başka birinin sesinin ana konuşmacının sesini bastırması.
    - Anlaşılmayan fısıltı veya şarkı söyleme.
    - Sesin kesik (truncated) olması ve aynı zamanda gürültüyle maskelenmesi.

2.  "heavy_accent": **Bu bayrağı SADECE VE SADECE, yukarıda listelenen `[unsure: ]` etiketlerinin bir veya daha fazlasının nedeni ağır bir aksan veya bölgesel diyalekt ise 'true' yap.** Eğer hiç `[unsure: ]` etiketi yoksa veya etiketlerin nedeni gürültü gibi başka bir şeyse, aksan ne kadar belirgin olursa olsun 'false' yap. Çocuk gevelemesi bu kategoriye girmez.

3.  "incorrect_language": Ana konuşmacının TÜM konuşması Türkçe dışında bir dildeyse 'true' yap. Konuşmanın sadece bir kısmı yabancı dildeyse 'false' yap.

4.  "is_synthesized": Ana konuşmacının sesi sentezlenmiş (robotik, akıllı asistan vb.) veya kaydedilmiş (örneğin bir anons) ise 'true' yap.

5.  "multiple_voices": Arka planda başka bir konuşmacı varsa VE bu konuşmacının sesi ana konuşmacınınkiyle neredeyse aynı ses seviyesindeyse VE konuşması anlaşılabilecek kadar netse 'true' yap. Belirsiz mırıltılar veya TV/radyo konuşmaları için 'false' yap.

BEKLENEN JSON FORMATI:
{{
  "unclear_audio": {{"value": boolean, "reason": "string"}},
  "heavy_accent": {{"value": boolean, "reason": "string"}},
  "incorrect_language": {{"value": boolean, "reason": "string"}},
  "is_synthesized": {{"value": boolean, "reason": "string"}},
  "multiple_voices": {{"value": boolean, "reason": "string"}}
}}
"""

    def _create_polly_proper_noun_prompt(self, text: str) -> str:
        """Özel isimleri düzeltmek için prompt oluşturur."""
        return f"""
GÖREV: Aşağıdaki transkripsiyon metnini analiz et ve içindeki özel isimleri (kişi, yer, coğrafi konum, şirket, marka, müzik eseri, sanatçı adı vb.) düzelt. Yalnızca düzeltilmiş metni döndür. Hiçbir açıklama veya markdown (```) ekleme.

KURALLAR:
1.  Kişi, yer ve coğrafi adresler için Wikipedia'yı referans al.
2.  Müzik eserleri (şarkı, albüm) ve sanatçılar için Apple Music verilerini referans al.
3.  Genel şirket ve marka adları için standart büyük harf kurallarını uygula.
4.  Eğer bir isim uydurma gibi duruyorsa veya kaynaklarda bulunamıyorsa, küçük harfle bırak.
5.  Metnin geri kalanını, dilbilgisini veya kelime sırasını DEĞİŞTİRME. Sadece özel isimlerin yazımını ve büyük/küçük harf durumunu düzelt.

ORİJİNAL METİN:
"{text}"

DÜZELTİLMİŞ METİN:
"""

    def _create_polly_quality_flags_prompt(self, transcription_text: str, audio_info: Dict) -> str:
        """POLLY kalite bayrağı analizi için prompt oluşturur."""
        audio_context = f"""
Ses dosyası teknik bilgileri:
- Süre: {audio_info.get('sure_formatli', 'Bilinmiyor')} saniye
- Tahmini SNR: {audio_info.get('tahmini_snr_db', 'Bilinmiyor'):.1f} dB
- Clipping Oranı: {audio_info.get('clipping_ratio', 0.0) * 100:.1f}%
"""
        
        return f"""
Lütfen **yalnızca** geçerli bir JSON nesnesi olarak yanıt verin. Başka hiçbir metin, markdown (```json) veya açıklama ekleme.

GÖREV: Aşağıdaki transkripsiyon metnini ve ses bilgilerini analiz et ve POLLY FAQ kriterlerine göre kalite bayraklarını belirt.

METİN:
"{transcription_text}"

{audio_context}

BEKLENEN JSON FORMATI:
{{
  "unclear_audio": {{"should_flag": boolean, "reason": "string"}},
  "heavy_accent": {{"should_flag": boolean, "reason": "string"}},
  "wrong_language": {{"should_flag": boolean, "reason": "string"}},
  "multiple_voices": {{"should_flag": boolean, "reason": "string"}}
}}
"""

    def _create_abbreviation_correction_prompt(self, text: str) -> str:
        """Kısaltmalar ve baş harfler için düzeltme prompt'u oluşturur."""
        return f"""
GÖREV: Aşağıdaki transkripsiyon metnini POLLY STEP 2 kurallarına (1-2, 15-23) göre düzelt. Yalnızca düzeltilmiş metni döndür. Hiçbir açıklama veya markdown (```) ekleme.

TÜRKÇE ÖZEL KURALLAR:

1. ABBREVIATIONS (Kısaltmalar):
   - Konuşmacı kısaltma kullanıyorsa → kısaltma kullan
   - Sistem doğru kısaltmışsa → olduğu gibi bırak
   - Türkçe kısaltmalar: Dr., Doç., Prof., Ltd., Şti., vb., vs.
   - Yabancı markalar: Inc., LLC., Corp., Ave., St.

2. TÜRKÇE ACRONYMS & INITIALISMS:
   - BÜYÜK HARFLERLE, NOKTASIZ yazılır:
     * TRT, THY, PTT, TSK, MİT, TBMM
     * NATO, UNESCO, FIFA, UEFA
   - Çoğul eki: THY'ler değil, THY şirketleri
   - Harf harf söylenen birleştir: "T R T" → TRT

15. TÜRKÇE SAYILAR:
   - Yanlış sayıları yazıyla düzelt: "on bir" (11 değil)
   - "Sıfır" yerine "oh" kullanılırsa bırak: "dört üç oh dört"
   - Türkçe marka adları: 7UP → "yedi up" değil
   - Roma rakamları: "Üçüncü Mehmet" → III. Mehmet
   - Sokak isimleri: "Elli Üçüncü Sokak" → 53. Sokak

16. TÜRKÇE ÖZEL İSİMLER:
   - Türk isimleri: Mehmet, Ayşe, Fatma (büyük harfle)
   - Şehirler: İstanbul, Ankara, İzmir
   - Markalar: Türk Telekom, Garanti Bankası, Arçelik
   - Yabancı isimler: McDonald's, iPhone, Facebook

17. TÜRKÇE BOŞLUK KURALLARI:
   - Bileşik kelimeler: "göz yaşı" değil "gözyaşı"
   - Apostrof: "Türkiye 'nin" → "Türkiye'nin"
   - Tire: "İstanbul - Ankara" → "İstanbul-Ankara"

18. TÜRKÇE HARF HARF YAZILANLAR:
   - "H A B E R" → HABER
   - "İ K İ" → iki (sayı ise)
   - Model numaraları: "X-846274" (olduğu gibi)
   - "İki tane L E N" → "A L L E N" ("iki tane" silinir)

19. TÜRKÇE KEKEMELİK:
   - "bir bir bir mesaj gönder" → "bir mesaj gönder"
   - "ne ne ne yapıyorsun" → "ne yapıyorsun"
   - Yarım kelimeler: "gön... göndermek istiyorum" → "göndermek istiyorum"

20. TÜRKÇE NOKTALAMA:
   - Konuşmacı "nokta" diyorsa → .
   - "virgül" diyorsa → ,
   - "soru işareti" diyorsa → ?
   - "ünlem işareti" diyorsa → !
   - Sembolleri kelime olarak YAZMA

21. TÜRKÇE BÜYÜK/KÜÇÜK HARF:
   - Cümle başları büyük
   - Özel isimler büyük: "türkiye" → "Türkiye"
   - Günler: "pazartesi" (küçük)
   - Aylar: "ocak" (küçük)
   - Milliyetler: "türk" → "Türk"

22. TÜRKÇE WEB ADRESLERİ:
   - "google nokta com" → "google.com"
   - "w w w nokta hurriyet nokta com nokta t r" → "www.hurriyet.com.tr"
   - E-posta: "ali at gmail nokta com" → "ali@gmail.com"

23. TÜRKÇE DİL BİLGİSİ:
   - Ünlü uyumu: "kitaplar" (kitablar değil)
   - Ünsüz yumuşaması: "kitabı" (kitapı değil)
   - Büyük ünlü uyumu: "şöförler" (şoförler değil)
   - Ağız özelliklerini standart Türkçe'ye çevir

TRANSKRİPSİYON METNİ:
{text}

DÜZELTİLMİŞ METİN:
""" 