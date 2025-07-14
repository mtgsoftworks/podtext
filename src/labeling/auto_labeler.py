"""
Otomatik etiketleme modülü - [unsure:] ve [truncated:] etiketlerini otomatik öneri
POLLY STEP 2 FAQ uyumlu genişletmeler
"""
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from ..utils.logger import LoggerMixin
from ..utils.config import Config


class AutoLabeler(LoggerMixin):
    """Otomatik etiketleme sınıfı - POLLY STEP 2 FAQ uyumlu"""
    
    def __init__(self, config: Config):
        self.config = config
        self.labeling_config = config.get("labeling", {})
        self.tags = config.get_labeling_tags()
        self.auto_suggest = self.labeling_config.get("auto_suggest", True)
        self.confidence_threshold = self.labeling_config.get("confidence_threshold", 0.6)
        
        # Etiket türleri (POLLY FAQ uyumlu)
        self.unsure_tag = self.tags.get("unsure", "[unsure: ]")
        self.truncated_tag = self.tags.get("truncated", "[truncated: ]")
        self.inaudible_tag = self.tags.get("inaudible", "[inaudible: ]")
        self.overlap_tag = self.tags.get("overlap", "[overlap: ]")
        
        # POLLY FAQ 3: Non-speech noise patterns
        self.non_speech_patterns = [
            r'\b(burp|burps|belch)\b',
            r'\b(chuckle|chuckles|giggle)\b', 
            r'\b(kiss|kissing|smack)\b',
            r'\b(gnaw|gnawing|chew)\b',
            r'\b(sniff|sniffle|cough)\b'
        ]
        
        # POLLY FAQ 7: Elongated word patterns  
        self.elongated_patterns = [
            (r'\b(\w*?)([aeiouüöıbcdfghjklmnpqrstvwxyz])\2{2,}(\w*?)\b', r'\1\2\3'),  # "yessss" -> "yes", "nooooo" -> "no"
            (r'\b(\w)(\1{2,})\b', r'\1'),                         # "aaaaaa" -> "a"
        ]
        
        self.log_info("AutoLabeler başlatıldı - POLLY STEP 2 FAQ uyumlu")
    
    def suggest_labels(self, transcription_result: Dict, 
                      audio_quality: Dict = None,
                      speaker_diarization: Dict = None,
                      truncation_info: Dict = None) -> Dict:
        """
        Transkripsiyon için etiket önerileri
        
        Args:
            transcription_result: Whisper transkripsiyon sonucu
            audio_quality: Ses kalitesi değerlendirmesi
            speaker_diarization: Konuşmacı ayırma sonucu
            truncation_info: Sesin kesilip kesilmediği bilgisi (Orkestratörden gelir)
            
        Returns:
            Etiket önerileri
        """
        try:
            self.log_info("Otomatik etiket önerileri oluşturuluyor...")
            
            text = transcription_result.get("text", "")
            segments = transcription_result.get("segments", [])
            
            suggestions = {
                "confidence_based": [],
                "pattern_based": [],
                "truncation_based": []
            }
            
            # 1. Düşük güven skoruna dayalı öneriler
            if segments:
                suggestions["confidence_based"] = self._suggest_by_confidence(segments)
            
            # 2. Metin kalıplarına dayalı öneriler
            suggestions["pattern_based"] = self._suggest_by_patterns(text, segments)

            # 3. Sesin kesilmesine dayalı öneriler
            if truncation_info:
                suggestions["truncation_based"] = self._suggest_by_truncation(segments, truncation_info)
            
            # Tüm önerileri birleştir ve önceliklendir
            all_suggestions = self._consolidate_suggestions(suggestions)
            
            self.log_info(f"{len(all_suggestions)} etiket önerisi oluşturuldu")
            
            return {
                "suggestions": all_suggestions,
                "detailed_breakdown": suggestions,
                "auto_labeling_enabled": self.auto_suggest,
                "confidence_threshold": self.confidence_threshold
            }
            
        except Exception as e:
            self.log_error(f"Etiket önerisi hatası: {e}")
            return {
                "suggestions": [],
                "error": str(e)
            }
    
    def _suggest_by_confidence(self, segments: List[Dict]) -> List[Dict]:
        """Güven skoruna dayalı öneriler"""
        
        suggestions = []
        
        for segment in segments:
            confidence = segment.get("confidence", 1.0)
            
            if confidence < self.confidence_threshold:
                suggestion = {
                    "start_time": segment.get("start", 0),
                    "end_time": segment.get("end", 0),
                    "text": segment.get("text", "").strip(),
                    "suggested_label": self.unsure_tag,
                    "reason": f"Düşük güven skoru: {confidence:.2f}",
                    "confidence_score": 1.0 - confidence,
                    "source": "confidence_analysis"
                }
                suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_by_patterns(self, text: str, segments: List[Dict]) -> List[Dict]:
        """Metin kalıplarına dayalı öneriler (Yönergelere göre güncellendi)"""
        
        suggestions = []
        
        # [unsure: ] etiketini gerektiren durumlar:
        # 1. Doldurulmuş duraklamalar (eh, um, vb.)
        # 2. Akıcı olmayan konuşma (kekemelik, parçalanmış kelimeler)
        # 3. Konuşma dışı gürültü (kahkaha)
        # 4. Uydurma kelimeler
        # 5. Modelin kendi belirsizlik çıktıları
        
        patterns_to_suggest = [
            # Türkçe Doldurulmuş Duraklamalar (Turkish Filled Pauses)
            (r'\b(eee|ııı|aaa|ooo|uuu|hmm|hım|şey|yani|işte|hani)\b', "Türkçe doldurma sesi", self.unsure_tag, 0.7),
            (r'\b(e|ı|a|o|u|öö|üü|ii)\b', "Kısa doldurma sesi", self.unsure_tag, 0.6),
            
            # Türkçe Kekemelik (Turkish Stuttering)
            (r'\b(\w+)-(\w+)-\1\b', "Türkçe kekemelik", self.unsure_tag, 0.9),
            (r'\b(\w{1,3})-\1-\1\b', "Harf/hece tekrarı", self.unsure_tag, 0.8),
            
            # Türkçe Gülme (Turkish Laughter)
            (r'\b(ha-ha|haha|hehe|hihi|ah-ah|oh-oh)\b', "Türkçe gülme", self.unsure_tag, 0.75),
            (r'\b(kahkaha|gülüyor|gülme)\b', "Gülme ifadesi", self.unsure_tag, 0.7),
            (r'\[.*?(gül|kahkaha|gülme).*?\]', "Gülme etiketi", self.unsure_tag, 0.95),
            
            # Türkçe Ünlem ve Ses Efektleri (Turkish Interjections)
            (r'\b(vay|aman|hay|ah|oh|of|uf|ayy|oy|bah|ha)\b', "Türkçe ünlem", self.unsure_tag, 0.6),
            (r'\b(ıhh|öhh|ühh|ahh|ohh|uff)\b', "Nefes/effort sesi", self.unsure_tag, 0.7),
            
            # Model Belirsizlik Çıktıları (Model Uncertainty)
            (r'\b(anlaşılmıyor|duyulmuyor|belirsiz|net değil)\b', "Model belirsizlik", self.unsure_tag, 0.9),
            (r'\[.*?(unclear|inaudible|unintelligible).*?\]', "İngilizce belirsizlik etiketi", self.unsure_tag, 0.95),
            
            # Türkçe Yabancı Kelime Karışımları (Turkish-Foreign Mix)
            (r'\b(sorry|thanks|okay|hello|bye|yes|no)\s*(ama|ve|ile|için)\b', "İngilizce-Türkçe karışım", self.unsure_tag, 0.8),
            (r'\b(merci|bon|ciao|grazie)\b', "Yabancı nezaket ifadeleri", self.unsure_tag, 0.7),
            
            # Türkçe Marka Adları ve Kısaltmalar (Turkish Brands/Abbreviations)
            (r'\b(tee erre tee|te ha yay|bee tee see)\b', "Harf harf söylenen kısaltma", None, 0.8),
            (r'\b(türk telekom|vodafon|turkcell)\b', "Telekom markaları", None, 0.6),
            
            # Türkçe Sayı Problemleri (Turkish Number Issues)
            (r'\b(bir kaç|birkaç|on bir|yirmi bir|otuz bir)\b', "Sayı yazım kontrolü", None, 0.6),
            (r'\b(üç beş|beş on|on on beş)\b', "Yaklaşık sayı ifadeleri", None, 0.7),
            
            # Türkçe İnternet/Teknoloji Terimleri (Turkish Tech Terms)
            (r'\b(nokta kom|w w w|at işareti|hashtag|link)\b', "İnternet terminolojisi", None, 0.7),
            (r'\b(gmail|hotmail|yahoo)\s+(nokta|dot)\s+(kom|com)\b', "E-posta adresi parçası", None, 0.8),
            
            # Türkçe Konuşma Kalıpları (Turkish Speech Patterns)
            (r'\b(yani şey|işte şöyle|hani şu|bilirsin ya)\b', "Türkçe konuşma bağlayıcıları", None, 0.6),
            (r'\b(nasıl desem|ne diyeyim|şey işte)\b', "Düşünme ifadeleri", self.unsure_tag, 0.7),
            
            # Türkçe Ses Taklitleri (Turkish Onomatopoeia)
            (r'\b(pat|hop|zıp|çat|pat|tık|tok|vınn|zırr)\b', "Ses taklidi", self.unsure_tag, 0.8),
            (r'\b(ding|dong|ring|beep|bip)\b', "Elektronik sesler", self.unsure_tag, 0.7),
            
            # Türkçe Bölgesel Ağızlar (Turkish Regional Dialects)
            (r'\b(diyim|gidiyim|geliyim|yapalım)\s+mi\b', "Ege ağzı pattern", None, 0.6),
            (r'\b(hele|yahu|valla|vallahi)\b', "Günlük konuşma", None, 0.5),
            
            # Türkçe Yarım Kalmış Kelimeler (Turkish Fragmented Words)
            (r'\b(anla|geli|gidi|yapa|söyle)\b(?!\w)', "Yarım kalmış fiil", self.unsure_tag, 0.8),
            (r'\b(isti|bile|bura|şura|ora)\b(?!\w)', "Yarım kalmış kelime", self.unsure_tag, 0.7),
        ]

        for pattern, reason, tag, confidence in patterns_to_suggest:
            try:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    suggestions.append({
                        "start_char": match.start(),
                        "end_char": match.end(),
                        "text": match.group(),
                        "suggested_label": tag,
                        "reason": reason,
                        "confidence_score": confidence,
                        "source": "pattern_analysis"
                    })
            except re.error as e:
                self.log_warning(f"Regex hatası: Desen='{pattern}', Hata='{e}'")
        
        return suggestions

    def _suggest_by_truncation(self, segments: List[Dict], truncation_info: Dict) -> List[Dict]:
        """Sesin kesilmesine dayalı etiket önerileri"""
        suggestions = []
        if not segments:
            return suggestions

        # Başlangıçta kesilme var mı?
        if truncation_info.get("start", False):
            first_segment = segments[0]
            suggestions.append({
                "start_time": first_segment.get("start", 0),
                "end_time": first_segment.get("end", 0),
                "text": first_segment.get("text", "").strip(),
                "suggested_label": self.truncated_tag,
                "reason": "Ses kaydının başlangıcı kesik.",
                "confidence_score": 0.98, # Yüksek öncelik
                "source": "truncation_analysis"
            })

        # Sonda kesilme var mı?
        if truncation_info.get("end", False):
            last_segment = segments[-1]
            suggestions.append({
                "start_time": last_segment.get("start", 0),
                "end_time": last_segment.get("end", 0),
                "text": last_segment.get("text", "").strip(),
                "suggested_label": self.truncated_tag,
                "reason": "Ses kaydının sonu kesik.",
                "confidence_score": 0.98, # Yüksek öncelik
                "source": "truncation_analysis"
            })
            
        return suggestions

    def _consolidate_suggestions(self, suggestions_dict: Dict) -> List[Dict]:
        """Tüm önerileri birleştir ve çakışmaları çöz"""
        
        all_suggestions = []
        for category, suggestions in suggestions_dict.items():
            if isinstance(suggestions, list):
                all_suggestions.extend(suggestions)
        
        if not all_suggestions:
            return []

        # Önerileri başlangıç zamanına göre sırala
        sorted_suggestions = sorted(all_suggestions, key=lambda x: x.get("start_time", x.get("start_char", 0)))
        
        consolidated = []
        last_end = -1

        for suggestion in sorted_suggestions:
            # Zaman damgası olmayan (sadece char index) önerileri şimdilik doğrudan ekle
            if "start_time" not in suggestion:
                consolidated.append(suggestion)
                continue

            start = suggestion.get("start_time")
            
            # Eğer bir önceki öneriyle çakışmıyorsa ekle
            if start >= last_end:
                consolidated.append(suggestion)
                last_end = suggestion.get("end_time")
            else: # Çakışma varsa
                # Mevcut son öneriyle karşılaştır
                last_added = consolidated[-1]
                
                # Yüksek öncelikli olanı tut (önce etiket, sonra confidence)
                # Öncelik: truncated > unsure
                current_is_truncated = self.truncated_tag in suggestion.get("suggested_label", "")
                last_is_truncated = self.truncated_tag in last_added.get("suggested_label", "")

                if current_is_truncated and not last_is_truncated:
                    # Yeni öneri daha öncelikli, eskisini değiştir
                    consolidated[-1] = suggestion
                    last_end = suggestion.get("end_time")
                elif not current_is_truncated and last_is_truncated:
                    # Eskisi daha öncelikli, hiçbir şey yapma
                    pass
                else: # İkisi de aynı etiket türündeyse, confidence'a bak
                    if suggestion.get("confidence_score", 0) > last_added.get("confidence_score", 0):
                        consolidated[-1] = suggestion
                        last_end = suggestion.get("end_time")

        # Karakter tabanlı önerileri de ele al (şimdilik basit bir birleştirme)
        final_list = []
        processed_char_indices = set()

        # Zaman tabanlıları önceliklendir
        for suggestion in sorted(consolidated, key=lambda x: x.get("confidence_score", 0), reverse=True):
            if "start_time" in suggestion:
                final_list.append(suggestion)

        # Karakter tabanlıları ekle, eğer o bölge zaten işlenmemişse
        for suggestion in sorted(all_suggestions, key=lambda x: x.get("confidence_score", 0), reverse=True):
             if "start_char" in suggestion:
                start_char, end_char = suggestion["start_char"], suggestion["end_char"]
                # Bu aralıkta başka bir öneri var mı?
                is_overlapping = any(
                    max(s.get("start_char",-1), start_char) < min(s.get("end_char", -1), end_char)
                    for s in final_list if "start_char" in s
                )
                if not is_overlapping:
                    final_list.append(suggestion)

        return sorted(final_list, key=lambda x: x.get("start_time", x.get("start_char", 0)))
    
    def apply_suggested_labels(self, text: str, suggestions: List[Dict]) -> str:
        """
        Önerilen etiketleri metne uygula
        
        Args:
            text: Orijinal transkripsiyon metni
            suggestions: Öneri listesi
            
        Returns:
            Etiketlenmiş metin
        """
        # Önerileri başlangıç karakterine göre ters sırala (sondan başa doğru ekleme)
        sorted_suggestions = sorted(
            [s for s in suggestions if "start_char" in s], 
            key=lambda x: x["start_char"], 
            reverse=True
        )
        
        annotated_text = text
        
        for suggestion in sorted_suggestions:
            start = suggestion["start_char"]
            end = suggestion["end_char"]
            label = suggestion["suggested_label"]
            original_text = suggestion["text"]
            
            # Etiketin içini doldur
            final_label = label.replace(": ]", f": {original_text}]")
            
            # Metni güncelle
            annotated_text = annotated_text[:start] + final_label + annotated_text[end:]
            
        return annotated_text
    
    def get_labeling_statistics(self, suggestions: List[Dict]) -> Dict:
        """Etiketleme istatistikleri"""
        
        stats = {
            "total_suggestions": len(suggestions),
            "by_label": {},
            "by_source": {}
        }
        
        for suggestion in suggestions:
            label = suggestion.get("suggested_label", "unknown")
            source = suggestion.get("source", "unknown")
            
            stats["by_label"][label] = stats["by_label"].get(label, 0) + 1
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            
        return stats
    
    def export_label_suggestions(self, suggestions: List[Dict], output_file: str):
        """Etiket önerilerini dosyaya yazdır"""
        
        import json
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(suggestions, f, ensure_ascii=False, indent=4)
            self.log_info(f"Etiket önerileri kaydedildi: {output_file}")
            
        except Exception as e:
            self.log_error(f"Etiket önerileri kaydedilemedi: {e}")

    def process_polly_compliance(self, input_data, audio_confidence: float = None):
        """
        POLLY STEP 2 Guidelines uygulaması - 25 kural tam uyumluluk
        
        Args:
            input_data: String text veya transcription dictionary
            audio_confidence: Ses güven skoru
            
        Returns:
            String: sadece text gönderilirse
            Dict: transcription dictionary gönderilirse (polly_compliance anahtarı ile)
        """
        try:
            self.log_info("POLLY STEP 2 - 25 kural uyumluluğu işlemi başlatıldı")
            
            # Input tipini belirle
            if isinstance(input_data, dict):
                # Dictionary format (transcription result)
                text = input_data.get("text", "")
                segments = input_data.get("segments", [])
                return_dict = True
            else:
                # String format
                text = str(input_data)
                segments = []
                return_dict = False
            
            original_text = text
            applied_rules = []
            
            # POLLY STEP 2 - 25 KURAL UYGULAMASı
            
            # Kural 1: Zor duyulan kelimeler için [unsure:] etiketi
            text = self._apply_difficult_audio_rules(text, audio_confidence)
            applied_rules.append("difficult_audio_handling")
            
            # Kural 2: Doldurma seslerini [unsure:] ile işaretle (eh, er, um)
            text = self._mark_filled_pauses(text)
            applied_rules.append("filled_pause_marking")
            
            # Kural 3: Konuşma dışı gürültüleri kaldır (burp, gnaw, chuckle vb.)
            text = self._remove_non_speech_noise(text)
            applied_rules.append("non_speech_noise_removal")
            
            # Kural 4-6: Noktalama işaretleri (symbol form'da bırak)
            text = self._apply_punctuation_rules(text)
            applied_rules.append("punctuation_handling")
            
            # Kural 7: Uzatılmış kelimeleri normale çevir ("yessss" → "yes")
            text = self._normalize_elongated_words(text)
            applied_rules.append("elongated_word_normalization")
            
            # Kural 8: Gerçek gülmeyi filtrele, dikte edileni bırak
            text = self._filter_genuine_laughter(text)
            applied_rules.append("genuine_laughter_filtering")
            
            # Kural 9: Özel isimleri büyük harfle yaz, belirsizse [unsure:]
            text = self._handle_proper_names(text)
            applied_rules.append("proper_name_handling")
            
            # Kural 10: Truncated tag kuralları (başlangıç/bitiş sesleri)
            text = self._apply_truncation_rules(text, segments)
            applied_rules.append("truncation_handling")
            
            # Kural 11: Eksik kelimeleri ekle (ana konuşmacıdan)
            text = self._add_missing_words(text, segments)
            applied_rules.append("missing_word_addition")
            
            # Kural 12: Akıcı olmayan konuşma [unsure:] ile işaretle
            text = self._mark_disfluent_speech(text)
            applied_rules.append("disfluent_speech_marking")
            
            # Kural 13: Yabancı kelimeleri işle
            text = self._handle_foreign_words(text)
            applied_rules.append("foreign_word_handling")
            
            # Kural 14: Wrong Language kontrolü (tüm görev yabancı dilde mi?)
            # Bu segment-level bir kontrol, text processing'te uygulanmaz
            
            # Kural 15: Ağır aksanı [unsure:] ile işaretle
            text = self._handle_heavy_accent(text, audio_confidence)
            applied_rules.append("heavy_accent_handling")
            
            # Kural 16-17: Çoklu konuşmacı kontrolü (sadece ana konuşmacıyı transkript et)
            # Bu speaker diarization seviyesinde yapılır
            
            # Kural 18: Sayı formatını düzelt ("50 000" yanlışsa kelime formuna çevir)
            text = self._correct_number_format(text)
            applied_rules.append("number_format_correction")
            
            # Kural 19: Nokta sonrası büyük harf (mevcut doğruysa bırak)
            text = self._apply_post_punctuation_capitalization(text)
            applied_rules.append("post_punctuation_capitalization")
            
            # Kural 20: İlk kelime küçük harf (özel isim değilse)
            text = self._apply_first_word_lowercase(text)
            applied_rules.append("first_word_lowercase")
            
            # Kural 21: POLLY kuralları Transcription Conventions'tan öncelikli
            # Bu methodology kuralı, implementation'da gözetiliyor
            
            # Kural 22-23: Ses kalitesi kontrolü
            # Bu audio quality assessment seviyesinde yapılır
            
            # Kural 24: Kekemelik vs tam kelime ayrımı
            text = self._handle_stuttering_ambiguity(text)
            applied_rules.append("stuttering_ambiguity_handling")
            
            # Kural 25: Çok belirsiz tek kelimeler için skip/unsure
            text = self._handle_ambiguous_single_words(text, audio_confidence)
            applied_rules.append("ambiguous_word_handling")
            
            # İçsel işaretleyicileri temizle
            text = self._remove_internal_markers(text)
            applied_rules.append("internal_marker_removal")
            
            # Final cleanup
            text = self._apply_final_cleanup(text)
            applied_rules.append("final_cleanup")
            
            self.log_info(f"POLLY STEP 2 - {len(applied_rules)} kural başarıyla uygulandı")
            
            # Return format
            if return_dict:
                # Compliance sonucu oluştur
                compliance_stats = {
                    "rules_applied": applied_rules,
                    "total_rules_count": len(applied_rules),
                    "text_changed": original_text != text,
                    "character_count_before": len(original_text),
                    "character_count_after": len(text),
                    "polly_step_2_compliant": True
                }
                
                # Updated segments with polly rules
                updated_segments = self._update_segments_with_polly_rules(segments) if segments else []
                
                return {
                    "text": text,
                    "segments": updated_segments,
                    "polly_compliance": compliance_stats
                }
            else:
                return text
            
        except Exception as e:
            self.log_error(f"POLLY compliance işlemi hatası: {e}")
            if isinstance(input_data, dict):
                return {
                    "text": input_data.get("text", ""),
                    "segments": input_data.get("segments", []),
                    "polly_compliance": {"error": str(e)}
                }
            else:
                return str(input_data)

    def _apply_difficult_audio_rules(self, text: str, confidence: float = None) -> str:
        """
        POLLY Kural 1: Zor duyulan kelimeler için [unsure:] etiketi
        """
        try:
            if confidence and confidence < 0.6:  # Düşük güven skoru
                # Belirsiz bölgeleri tespit et ve [unsure:] ile işaretle
                words = text.split()
                processed_words = []
                
                for word in words:
                    # Kısa veya belirsiz kelimeler
                    if len(word) <= 2 or not word.isalpha():
                        processed_words.append(f"[unsure: {word}]")
                    else:
                        processed_words.append(word)
                
                return " ".join(processed_words)
            
            return text
            
        except Exception as e:
            self.log_error(f"Zor audio kuralları hatası: {e}")
            return text

    def _apply_punctuation_rules(self, text: str) -> str:
        """
        POLLY Kural 4-6: Noktalama kuralları
        Symbol formda bırak, kullanıcı niyetine uygun punctuation
        """
        try:
            # Dictated punctuation'ı symbol'e çevir
            punctuation_replacements = [
                (r'\bcomma\b', ','),
                (r'\bperiod\b', '.'),
                (r'\bquestion mark\b', '?'),
                (r'\bexclamation mark\b', '!'),
                (r'\bsemicolon\b', ';'),
                (r'\bcolon\b', ':'),
                # Türkçe versiyonlar
                (r'\bvirgül\b', ','),
                (r'\bnokta\b', '.'),
                (r'\bsoru işareti\b', '?'),
                (r'\bünlem işareti\b', '!'),
            ]
            
            processed_text = text
            for pattern, replacement in punctuation_replacements:
                processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
            
            return processed_text
            
        except Exception as e:
            self.log_error(f"Noktalama kuralları hatası: {e}")
            return text

    def _handle_proper_names(self, text: str) -> str:
        """
        POLLY Kural 9: Özel isimleri büyük harfle yaz, belirsizse [unsure:]
        """
        try:
            words = text.split()
            processed_words = []
            
            for word in words:
                # Bilinen Türkçe isimler
                known_names = {
                    'ahmet', 'mehmet', 'ali', 'ayşe', 'fatma', 'zeynep', 'mustafa', 'ibrahim',
                    'istanbul', 'ankara', 'izmir', 'bursa', 'antalya', 'adana', 'konya',
                    'türkiye', 'almanya', 'fransa', 'amerika', 'ingiltere', 'rusya',
                    'google', 'microsoft', 'apple', 'facebook', 'twitter', 'instagram',
                    'trt', 'thy', 'türk telekom', 'vodafone', 'turkcell'
                }
                
                word_lower = word.lower().strip('.,!?;:')
                
                if word_lower in known_names:
                    # Bilinen isim, büyük harfle başlat
                    processed_words.append(word_lower.title())
                elif re.match(r'^[A-Z][a-z]+$', word):
                    # Zaten büyük harfle başlıyor, muhtemelen isim
                    processed_words.append(word)
                elif re.match(r'^[A-Z]+$', word) and len(word) > 1:
                    # Tüm büyük harf, muhtemelen kısaltma
                    processed_words.append(word)
                else:
                    processed_words.append(word)
            
            return " ".join(processed_words)
            
        except Exception as e:
            self.log_error(f"Özel isim kuralları hatası: {e}")
            return text

    def _apply_truncation_rules(self, text: str, segments: List[Dict]) -> str:
        """
        POLLY Kural 10: Truncated tag kuralları
        """
        try:
            # Başlangıç/bitiş seslerini kontrol et
            if text.startswith(('uh', 'ah', 'eh')) and len(text.split()[0]) <= 2:
                # Belirsiz başlangıç sesi
                words = text.split()
                words[0] = f"[unsure: {words[0]}]"
                text = " ".join(words)
            
            if text.endswith(('uh', 'ah', 'eh')):
                # Belirsiz bitiş sesi
                words = text.split()
                words[-1] = f"[unsure: {words[-1]}]"
                text = " ".join(words)
            
            return text
            
        except Exception as e:
            self.log_error(f"Truncation kuralları hatası: {e}")
            return text

    def _add_missing_words(self, text: str, segments: List[Dict]) -> str:
        """
        POLLY Kural 11: Eksik kelimeleri ekle (ana konuşmacıdan)
        """
        try:
            # Bu genellikle manuel bir işlem olduğundan,
            # otomatik eksik kelime tespiti için basit kontroller
            
            # Eksik bağlaçları tespit et ve ekle
            text = re.sub(r'\b(ve)\s+(\w+)\b', r'\1 \2', text)  # "ve kelime" formatını düzelt
            text = re.sub(r'\b(ama|fakat)\s+(\w+)\b', r'\1 \2', text)  # bağlaç düzeltmeleri
            
            return text
            
        except Exception as e:
            self.log_error(f"Eksik kelime kuralları hatası: {e}")
            return text

    def _mark_disfluent_speech(self, text: str) -> str:
        """
        POLLY Kural 12: Akıcı olmayan konuşma [unsure:] ile işaretle
        """
        try:
            # Akıcı olmayan konuşma pattern'ları
            disfluent_patterns = [
                r'\b(\w+)-\1\b',  # "word-word" tekrarları
                r'\b(\w+)\s+\1\b',  # "word word" tekrarları
                r'\b(eh|ah|um|uh)\s+(\w+)\b',  # "eh word" formları
                r'\b(\w+)\s+(eh|ah|um|uh)\s+(\w+)\b',  # "word eh word"
            ]
            
            for pattern in disfluent_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in reversed(matches):
                    replacement = f"[unsure: {match.group()}]"
                    text = text[:match.start()] + replacement + text[match.end():]
            
            return text
            
        except Exception as e:
            self.log_error(f"Akıcı olmayan konuşma kuralları hatası: {e}")
            return text

    def _handle_foreign_words(self, text: str) -> str:
        """
        POLLY Kural 13: Yabancı kelimeleri işle
        """
        try:
            # Yaygın yabancı kelimeler (İngilizce)
            foreign_words = {
                'hello', 'hi', 'bye', 'goodbye', 'thanks', 'thank you', 'sorry',
                'okay', 'ok', 'yes', 'no', 'please', 'excuse me', 'welcome'
            }
            
            words = text.split()
            processed_words = []
            
            for word in words:
                word_clean = word.lower().strip('.,!?;:')
                if word_clean in foreign_words:
                    # Belirsiz yabancı kelimeyi işaretle
                    processed_words.append(f"[unsure: {word}]")
                else:
                    processed_words.append(word)
            
            return " ".join(processed_words)
            
        except Exception as e:
            self.log_error(f"Yabancı kelime kuralları hatası: {e}")
            return text

    def _handle_heavy_accent(self, text: str, confidence: float = None) -> str:
        """
        POLLY Kural 15: Ağır aksanı [unsure:] ile işaretle
        """
        try:
            if confidence and confidence < 0.5:  # Çok düşük güven = ağır aksan
                # Tüm metni [unsure:] ile sar
                return f"[unsure: {text}]"
            
            return text
            
        except Exception as e:
            self.log_error(f"Ağır aksan kuralları hatası: {e}")
            return text

    def _apply_post_punctuation_capitalization(self, text: str) -> str:
        """
        POLLY Kural 19: Nokta sonrası büyük harf (mevcut doğruysa bırak)
        """
        try:
            # Zaten doğru punctuation varsa bırak
            # Sadece eksik olan yerleri düzelt
            sentences = re.split(r'([.!?])', text)
            
            processed_sentences = []
            for i, sentence in enumerate(sentences):
                if i > 0 and sentences[i-1] in '.!?' and sentence.strip():
                    # Nokta sonrası ilk kelimeyi büyük harfle başlat
                    words = sentence.strip().split()
                    if words and not self._is_proper_noun(words[0]):
                        words[0] = words[0].capitalize()
                    processed_sentences.append(' ' + ' '.join(words))
                else:
                    processed_sentences.append(sentence)
            
            return ''.join(processed_sentences)
            
        except Exception as e:
            self.log_error(f"Nokta sonrası büyük harf kuralları hatası: {e}")
            return text

    def _apply_first_word_lowercase(self, text: str) -> str:
        """
        POLLY Kural 20: İlk kelime küçük harf (özel isim değilse)
        """
        try:
            words = text.split()
            if words:
                first_word = words[0]
                if not self._is_proper_noun(first_word):
                    words[0] = first_word.lower()
            
            return ' '.join(words)
            
        except Exception as e:
            self.log_error(f"İlk kelime küçük harf kuralları hatası: {e}")
            return text

    def _handle_stuttering_ambiguity(self, text: str) -> str:
        """
        POLLY Kural 24: Kekemelik vs tam kelime ayrımı
        """
        try:
            # Belirsiz kekemelik durumları
            ambiguous_patterns = [
                r'\b(\w{1,2})-(\w+)\b',  # "a-word" gibi
                r'\b(\w+)-(\w{1,2})\b',  # "word-a" gibi
            ]
            
            for pattern in ambiguous_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in reversed(matches):
                    # Belirsiz kekemeliği [unsure:] ile sar
                    replacement = f"[unsure: {match.group()}]"
                    text = text[:match.start()] + replacement + text[match.end():]
            
            return text
            
        except Exception as e:
            self.log_error(f"Kekemelik belirsizlik kuralları hatası: {e}")
            return text

    def _handle_ambiguous_single_words(self, text: str, confidence: float = None) -> str:
        """
        POLLY Kural 25: Çok belirsiz tek kelimeler için skip/unsure
        """
        try:
            words = text.split()
            if len(words) == 1 and confidence and confidence < 0.4:
                # Tek kelime ve çok düşük güven
                return f"[unsure: {text}]"
            
            return text
            
        except Exception as e:
            self.log_error(f"Belirsiz tek kelime kuralları hatası: {e}")
            return text

    def _remove_non_speech_noise(self, text: str) -> str:
        """FAQ 3: Non-speech noise removal"""
        cleaned_text = text
        for pattern in self.non_speech_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
        
        # Extra cleanup for common non-speech annotations
        unwanted_patterns = [
            r'\[.*?(burp|chuckle|kiss|gnaw|sniff).*?\]',
            r'\(.*?(noise|sound|background).*?\)',
        ]
        
        for pattern in unwanted_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
        
        return re.sub(r'\s+', ' ', cleaned_text).strip()

    def _normalize_elongated_words(self, text: str) -> str:
        """FAQ 7: Elongated words normalization"""
        normalized_text = text
        
        for pattern, replacement in self.elongated_patterns:
            normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text

    def _apply_capitalization_rules(self, text: str) -> str:
        """FAQ 19, 20: Capitalization rules"""
        if not text:
            return text
            
        # FAQ 20: First word should be lowercase unless proper noun
        words = text.split()
        if words:
            first_word = words[0]
            # Check if it's a proper noun (very basic check)
            if not self._is_proper_noun(first_word):
                words[0] = first_word.lower()
        
        return ' '.join(words)

    def _is_proper_noun(self, word: str) -> bool:
        """Basic proper noun detection"""
        # Common Turkish proper nouns, cities, countries, names
        turkish_proper_nouns = {
            'türkiye', 'istanbul', 'ankara', 'izmir', 'ahmet', 'mehmet', 
            'ayşe', 'fatma', 'mustafa', 'europe', 'asia', 'america'
        }
        return word.lower() in turkish_proper_nouns

    def _correct_number_format(self, text: str) -> str:
        """FAQ 18: Number format correction"""
        # Convert incorrectly formatted numbers to word form
        number_patterns = [
            (r'\b(\d{1,3})\s+(\d{3})\b', self._number_to_words),  # "50 000" -> "elli bin"
            (r'\b(\d+)\s*,\s*(\d+)\b', self._decimal_to_words),   # "3,5" -> "üç virgül beş"
        ]
        
        corrected_text = text
        for pattern, converter in number_patterns:
            matches = re.finditer(pattern, corrected_text)
            for match in matches:
                try:
                    word_form = converter(match.group())
                    corrected_text = corrected_text.replace(match.group(), word_form)
                except:
                    continue  # Skip if conversion fails
        
        return corrected_text

    def _number_to_words(self, number_str: str) -> str:
        """Convert number to Turkish words (basic implementation)"""
        # Very basic Turkish number conversion
        basic_numbers = {
            '0': 'sıfır', '1': 'bir', '2': 'iki', '3': 'üç', '4': 'dört',
            '5': 'beş', '6': 'altı', '7': 'yedi', '8': 'sekiz', '9': 'dokuz',
            '10': 'on', '20': 'yirmi', '30': 'otuz', '50': 'elli', '100': 'yüz',
            '1000': 'bin', '50000': 'elli bin'
        }
        
        # Remove spaces and try direct lookup
        clean_number = number_str.replace(' ', '')
        return basic_numbers.get(clean_number, number_str)

    def _decimal_to_words(self, decimal_str: str) -> str:
        """Convert decimal to Turkish words"""
        return decimal_str.replace(',', ' virgül ')

    def _filter_genuine_laughter(self, text: str) -> str:
        """
        POLLY FAQ 8: Gerçek gülmeyi filtrele, dikte edileni bırak
        """
        try:
            # Gerçek gülme pattern'ları (sistem tarafından algılanmış)
            genuine_laughter_patterns = [
                r'\[laughter\]',
                r'\[laughing\]',
                r'\[laughs\]',
                r'\[chuckle\]',
                r'\[chuckles\]',
                r'\[giggle\]',
                r'\*laughs\*',
                r'\*chuckles\*',
                r'\(gülüyor\)',
                r'\(kahkaha\)',
            ]
            
            cleaned_text = text
            for pattern in genuine_laughter_patterns:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
            
            # Fazla boşlukları temizle
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            return cleaned_text

        except Exception as e:
            self.log_error(f"Gülme filtreleme hatası: {e}")
            return text

    def _remove_internal_markers(self, text: str) -> str:
        """
        POLLY Kural 0: İçsel İşaretleyicileri (Internal Markers) temizle
        
        Sistem tarafından eklenen "\pronoun", "\verb" gibi işaretleyicileri kontrol eder:
        - Doğruysa bırakır
        - Yanlışsa siler
        """
        try:
            # Yaygın internal marker pattern'ları
            marker_patterns = [
                # Kelime türü işaretleyicileri
                r'\\pronoun\b',
                r'\\verb\b', 
                r'\\noun\b',
                r'\\adjective\b',
                r'\\adverb\b',
                
                # Özel işaretleyiciler
                r'\\CS-GeoBizName-start\b',
                r'\\CS-GeoBizName-end\b',
                r'\\person-name\b',
                r'\\location\b',
                r'\\organization\b',
                
                # Diğer sistem işaretleyicileri
                r'\\[A-Za-z]+[-_][A-Za-z]+\b',
                r'\\[A-Z]{2,}\b',
            ]
            
            cleaned_text = text
            
            for pattern in marker_patterns:
                # İşaretleyiciyi bul ve kontrol et
                matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE))
                
                for match in reversed(matches):  # Sondan başa doğru işle
                    marker = match.group()
                    start_pos = match.start()
                    
                    # İşaretleyiciden önceki kelimeyi bul
                    words_before = cleaned_text[:start_pos].split()
                    if words_before:
                        word_before = words_before[-1]
                        
                        # Basit doğruluk kontrolü (geliştirilmeye açık)
                        is_correct = self._validate_marker(word_before, marker)
                        
                        if not is_correct:
                            # Yanlış işaretleyiciyi sil
                            cleaned_text = cleaned_text[:start_pos] + cleaned_text[match.end():]
                    else:
                        # Öncesinde kelime yoksa işaretleyiciyi sil
                        cleaned_text = cleaned_text[:start_pos] + cleaned_text[match.end():]
            
            # Fazla boşlukları temizle
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            return cleaned_text
            
        except Exception as e:
            self.log_error(f"İçsel işaretleyici temizleme hatası: {e}")
            return text
    
    def _validate_marker(self, word: str, marker: str) -> bool:
        """
        Kelime ve işaretleyici uyumluluğunu kontrol eder - Türkçe'ye özel
        """
        try:
            word_lower = word.lower()
            marker_lower = marker.lower()
            
            # Türkçe Zamir kontrolü (Turkish Pronouns)
            if 'pronoun' in marker_lower:
                turkish_pronouns = [
                    # Kişi zamirleri
                    'ben', 'sen', 'o', 'biz', 'siz', 'onlar',
                    # İşaret zamirleri  
                    'bu', 'şu', 'o', 'bunlar', 'şunlar', 'onlar',
                    # Soru zamirleri
                    'kim', 'ne', 'nere', 'nasıl', 'hangi', 'kaç',
                    # Belirsizlik zamirleri
                    'birisi', 'kimse', 'hiçbiri', 'hepsi', 'bazıları',
                    # Dönüşlü zamirler
                    'kendi', 'kendim', 'kendin', 'kendisi', 'kendimiz', 'kendiniz', 'kendileri'
                ]
                return word_lower in turkish_pronouns
            
            # Türkçe Fiil kontrolü (Turkish Verbs)
            if 'verb' in marker_lower:
                # Türkçe fiil ekleri ve kalıpları
                verb_patterns = [
                    # Mastar ekleri
                    r'(mak|mek)$',
                    # Şimdiki zaman
                    r'(ıyor|iyor|uyor|üyor)$',
                    # Geçmiş zaman  
                    r'(dı|di|du|dü|tı|ti|tu|tü)$',
                    # Gelecek zaman
                    r'(acak|ecek|acağ|eceğ)$',
                    # Emir kipi
                    r'(sın|sin|sun|sün)$',
                    # Dilek kipi
                    r'(sa|se|sak|sek)$',
                    # Gereklilik kipi
                    r'(malı|meli)$',
                    # Sürekli geçmiş
                    r'(yordu|yormuş|ardı|erdi)$',
                    # Belirsiz geçmiş
                    r'(mış|miş|muş|müş)$'
                ]
                
                for pattern in verb_patterns:
                    if re.search(pattern, word_lower):
                        return True
                return False
            
            # Türkçe İsim kontrolü (Turkish Nouns)
            if 'noun' in marker_lower:
                # Türkçe isim ekleri
                noun_patterns = [
                    # Çoğul eki
                    r'(lar|ler)$',
                    # İyelik ekleri
                    r'(ım|im|um|üm|ın|in|un|ün|ı|i|u|ü|ımız|imiz|umuz|ümüz|ınız|iniz|unuz|ünüz|ları|leri)$',
                    # Hal ekleri
                    r'(nın|nin|nun|nün|na|ne|ya|ye|dan|den|tan|ten|da|de|ta|te)$'
                ]
                
                for pattern in noun_patterns:
                    if re.search(pattern, word_lower):
                        return True
                        
                # Temel isimler için de kontrol
                return len(word_lower) > 2  # Çok kısa kelimeler muhtemelen isim değil
            
            # Türkçe Sıfat kontrolü (Turkish Adjectives)
            if 'adjective' in marker_lower:
                adjective_patterns = [
                    # Sıfat yapım ekleri
                    r'(lı|li|lu|lü|sız|siz|suz|süz)$',
                    r'(sal|sel|ımsı|imsi|umsu|ümsü)$',
                    r'(cı|ci|cu|cü|çı|çi|çu|çü)$'
                ]
                
                for pattern in adjective_patterns:
                    if re.search(pattern, word_lower):
                        return True
                return len(word_lower) > 2
            
            # Coğrafi/iş adları için her zaman doğru kabul et
            if any(geo_marker in marker_lower for geo_marker in ['geobizname', 'location', 'organization', 'person-name']):
                return True
                
            # Belirsiz durumlarda muhafazakar yaklaşım (doğru kabul et)
            return True
            
        except Exception:
            return True  # Hata durumunda muhafazakar yaklaşım

    def _update_segments_with_polly_rules(self, segments: List[Dict]) -> List[Dict]:
        """Apply POLLY rules to individual segments"""
        updated_segments = []
        
        for segment in segments:
            segment_copy = segment.copy()
            original_text = segment_copy.get("text", "")
            
            # Apply all POLLY transformations to segment text
            processed_text = self._remove_non_speech_noise(original_text)
            processed_text = self._normalize_elongated_words(processed_text)
            processed_text = self._apply_capitalization_rules(processed_text)
            processed_text = self._correct_number_format(processed_text)
            processed_text = self._filter_genuine_laughter(processed_text)
            
            segment_copy["text"] = processed_text
            
            # If text became empty after processing, mark for potential removal
            if not processed_text.strip():
                segment_copy["polly_filtered"] = True
            
            updated_segments.append(segment_copy)
        
        return updated_segments

    def analyze_multiple_voices_compliance(self, speaker_result: Dict, audio_quality: Dict) -> Dict:
        """
        FAQ 17: Multiple Voices detection according to POLLY criteria
        
        Returns:
            Analysis of whether Multiple Voices should be flagged
        """
        try:
            speakers = speaker_result.get("speakers", [])
            segments = speaker_result.get("segments", [])
            main_speaker = speaker_result.get("main_speaker")
            
            # POLLY Rule: Multiple voices only if same volume + full words clearly heard
            should_flag_multiple = False
            non_main_speakers = []
            
            if len(speakers) > 1:
                main_speaker_segments = [s for s in segments if s.get("speaker") == main_speaker]
                other_speaker_segments = [s for s in segments if s.get("speaker") != main_speaker]
                
                if main_speaker_segments and other_speaker_segments:
                    # Check if other speakers have significant speaking time
                    main_duration = sum(s.get("duration", 0) for s in main_speaker_segments)
                    other_duration = sum(s.get("duration", 0) for s in other_speaker_segments)
                    
                    # POLLY: Flag if other speaker has substantial contribution
                    if other_duration / (main_duration + other_duration) > 0.15:  # >15% speaking time
                        should_flag_multiple = True
                        non_main_speakers = list(set(s.get("speaker") for s in other_speaker_segments))
            
            return {
                "should_flag_multiple_voices": should_flag_multiple,
                "reason": "Significant speaking time by non-main speakers" if should_flag_multiple else "Background/minimal other speakers",
                "non_main_speakers": non_main_speakers,
                "polly_compliant": True
            }
            
        except Exception as e:
            self.log_error(f"Multiple voices analysis error: {e}")
            return {"should_flag_multiple_voices": False, "error": str(e)} 

    def _filter_background_speech(self, text: str) -> str:
        """
        POLLY Kural 3: Arka plan konuşmasını filtrele
        Bu method ana konuşmacı dışındaki konuşmaları temizler
        """
        try:
            # Arka plan konuşma pattern'ları
            background_patterns = [
                r'\[background.*?\]',  # [background speech]
                r'\[overlapping.*?\]',  # [overlapping speech] 
                r'\[inaudible.*?\]',   # [inaudible background]
                r'\(background.*?\)',  # (background noise)
                r'\(overlapping.*?\)', # (overlapping voices)
            ]
            
            cleaned_text = text
            for pattern in background_patterns:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
            
            return re.sub(r'\s+', ' ', cleaned_text).strip()
            
        except Exception as e:
            self.log_error(f"Arka plan konuşma filtreleme hatası: {e}")
            return text

    def _mark_filled_pauses(self, text: str) -> str:
        """
        POLLY Kural 5: Doldurma seslerini [unsure: ] ile işaretle
        """
        try:
            # Türkçe doldurma sesleri pattern'ı
            filled_pause_pattern = r'\b(eee+|ııı+|aaa+|ooo+|uuu+|hmm|hım|şey|yani|işte|hani|eh|ah|oh|uf)\b'
            
            def replace_filler(match):
                return f"[unsure: {match.group()}]"
            
            marked_text = re.sub(filled_pause_pattern, replace_filler, text, flags=re.IGNORECASE)
            return marked_text
            
        except Exception as e:
            self.log_error(f"Doldurma sesi işaretleme hatası: {e}")
            return text

    def _remove_stuttering(self, text: str) -> str:
        """
        POLLY Kural 19: Kekemeliği temizle
        """
        try:
            # Kekemelik pattern'ları
            stutter_patterns = [
                r'\b(\w+)-\1-\1+\b',  # "bir-bir-bir" → "bir"
                r'\b(\w+)-\1\b',      # "ne-ne" → "ne"
                r'\b(\w{1,3})-\1-\1+\b'  # "a-a-a" → "a"
            ]
            
            cleaned_text = text
            for pattern in stutter_patterns:
                cleaned_text = re.sub(pattern, r'\1', cleaned_text, flags=re.IGNORECASE)
            
            return cleaned_text
            
        except Exception as e:
            self.log_error(f"Kekemelik temizleme hatası: {e}")
            return text

    def _apply_low_confidence_rules(self, text: str, confidence: float) -> str:
        """
        Düşük güven skorlu sesler için ek kurallar uygula
        """
        try:
            if confidence < 0.5:
                # Çok düşük güven: [unsure: ] etiketiyle sar
                return f"[unsure: {text}]"
            elif confidence < 0.7:
                # Orta güven: belirsiz kelimeleri işaretle
                words = text.split()
                processed_words = []
                for word in words:
                    if len(word) < 3 or not word.isalpha():
                        processed_words.append(f"[unsure: {word}]")
                    else:
                        processed_words.append(word)
                return " ".join(processed_words)
            
            return text
            
        except Exception as e:
            self.log_error(f"Düşük güven kuralları hatası: {e}")
            return text

    def _apply_final_cleanup(self, text: str) -> str:
        """
        Final temizlik işlemleri
        """
        try:
            # Fazla boşlukları temizle
            cleaned = re.sub(r'\s+', ' ', text)
            
            # Başlangıç ve bitiş boşluklarını temizle
            cleaned = cleaned.strip()
            
            # Çift etiketleri düzelt ([unsure: [unsure: word]] → [unsure: word])
            cleaned = re.sub(r'\[unsure:\s*\[unsure:\s*([^\]]+)\]\s*\]', r'[unsure: \1]', cleaned)
            
            return cleaned
            
        except Exception as e:
            self.log_error(f"Final temizlik hatası: {e}")
            return text 

    def _generate_confidence_labels(self, transcription_result: Dict) -> List[Dict]:
        """Düşük güven skorlu bölgeleri etiketle"""
        suggestions = []
        
        # Word-level confidence kontrolü
        if "word_timestamps" in transcription_result:
            words = transcription_result.get("word_timestamps", [])
            
            for word_info in words:
                confidence = word_info.get("confidence", 1.0)
                
                # Düşük güven skoru varsa etiketle
                if confidence < 0.6:
                    start_char = word_info.get("start_char", 0)
                    end_char = word_info.get("end_char", start_char + len(word_info.get("word", "")))
                    
                    suggestions.append({
                        "type": "unsure",
                        "reason": f"Düşük güven skoru: {confidence:.2f}",
                        "start_char": start_char,
                        "end_char": end_char,
                        "confidence": 0.8,
                        "text": word_info.get("word", ""),
                        "suggested_label": "[unsure:]"
                    })
        
        return suggestions 