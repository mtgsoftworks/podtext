"""
Türkçe NLP Analizi - SpaCy tr_core_news_lg modeli kullanarak
"""
import spacy
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..utils.logger import LoggerMixin
from ..utils.config import Config


class TurkishNLPAnalyzer(LoggerMixin):
    """Türkçe metin analizi sınıfı - SpaCy tr_core_news_lg modeli"""
    
    def __init__(self, config: Config):
        self.config = config
        self.nlp_config = self.config.get("nlp", {})
        self.tasks = self.nlp_config.get("tasks", [])
        
        self.log_info("SpaCy Türkçe tr_core_news_lg modeli yükleniyor...")
        
        try:
            # SpaCy Turkish large model
            model_name = self.nlp_config.get("model_name", "tr_core_news_lg")
            self.nlp = spacy.load(model_name)
            
            self.log_info(f"SpaCy Türkçe modeli başarıyla yüklendi: {model_name}")
            self.log_info(f"Model pipeline: {self.nlp.pipe_names}")
            
        except OSError as e:
            self.log_error(f"SpaCy Türkçe modeli '{model_name}' bulunamadı: {e}")
            self.log_info("Modelin kurulu olduğundan emin olun. Kurulum için proje dizinindeyken:")
            self.log_info(r"pip install C:\Users\MTG\Desktop\podtext\tr_core_news_lg-1.0-py3-none-any.whl")
            raise
        except Exception as e:
            self.log_error(f"SpaCy Türkçe modeli yüklenirken hata oluştu: {e}")
            raise

        self.log_info(f"TurkishNLPAnalyzer başlatıldı - SpaCy model: {model_name}")
    
    def analyze_text(self, text: str) -> Dict:
        """
        Kapsamlı metin analizi
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            Analiz sonuçları
        """
        if not text or not self.nlp:
            return {"error": "Model yüklü değil veya metin boş"}
        
        try:
            self.log_info(f"Metin analizi başlatılıyor: {len(text)} karakter")
            
            # SpaCy analizi
            doc = self.nlp(text)
            
            results = {
                "text_stats": self._calculate_text_stats(text, doc),
                "entities": self._extract_entities(doc),
                "sentiment": self._analyze_sentiment(text, doc),
                "tokens": self._analyze_tokens(doc),
                "keywords": self._extract_keywords(text),
                "summary": self._generate_summary(text)
            }
            
            self.log_info("Metin analizi tamamlandı")
            return results
            
        except Exception as e:
            self.log_error(f"Analiz hatası: {e}")
            return {"error": str(e)}
    
    def _calculate_text_stats(self, text: str, doc) -> Dict:
        """Temel metin istatistikleri - SpaCy doc ile zenginleştirilmiş"""
        sentences = list(doc.sents)
        tokens = [token for token in doc if not token.is_space]
        words = [token for token in doc if not token.is_punct and not token.is_space]
        
        return {
            "character_count": len(text),
            "token_count": len(tokens),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(token.text) for token in words]) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "punct_count": len([token for token in doc if token.is_punct]),
            "stop_word_count": len([token for token in doc if token.is_stop])
        }
    
    def _extract_entities(self, doc) -> List[Dict]:
        """Varlık tanıma (NER) - SpaCy NER kullanarak"""
        try:
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "label_desc": spacy.explain(ent.label_) or ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0  # SpaCy confidence score genelde yok, 1.0 veriyoruz
                })
            
            self.log_info(f"NER: {len(entities)} varlık bulundu")
            return entities
            
        except Exception as e:
            self.log_warning(f"NER hatası: {e}")
            return []
    
    def _analyze_sentiment(self, text: str, doc) -> Dict:
        """Duygu analizi - Basit rule-based Türkçe sentiment"""
        try:
            # Türkçe pozitif/negatif kelimeler
            positive_words = {
                'güzel', 'harika', 'mükemmel', 'başarılı', 'iyi', 'sevindirici', 
                'mutlu', 'keyifli', 'hoş', 'faydalı', 'yararlı', 'etkili',
                'beğendim', 'teşekkür', 'şükür', 'memnun', 'tatmin'
            }
            
            negative_words = {
                'kötü', 'berbat', 'korkunç', 'başarısız', 'kızgın', 'üzgün',
                'sinirli', 'rahatsız', 'problemli', 'hatalı', 'eksik', 'yetersiz',
                'beğenmedim', 'şikayet', 'sorun', 'dert', 'sıkıntı'
            }
            
            tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
            
            positive_count = sum(1 for token in tokens if token in positive_words)
            negative_count = sum(1 for token in tokens if token in negative_words)
            
            total_words = len(tokens)
            
            if positive_count > negative_count:
                label = "POSITIVE"
                confidence = min(0.9, 0.5 + (positive_count / total_words) * 2)
            elif negative_count > positive_count:
                label = "NEGATIVE" 
                confidence = min(0.9, 0.5 + (negative_count / total_words) * 2)
            else:
                label = "NEUTRAL"
                confidence = 0.5
            
            return {
                "label": label,
                "confidence": round(confidence, 3),
                "positive_words": positive_count,
                "negative_words": negative_count,
                "method": "rule_based_turkish"
            }
            
        except Exception as e:
            self.log_warning(f"Sentiment analizi hatası: {e}")
            return {"label": "UNKNOWN", "confidence": 0.0, "method": "error"}
    
    def _analyze_tokens(self, doc) -> List[Dict]:
        """Token analizi - POS tagging, morphology, dependency"""
        try:
            tokens = []
            for token in doc:
                if not token.is_space:  # Boşlukları atla
                    tokens.append({
                        "text": token.text,
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "tag": token.tag_,
                        "dep": token.dep_,
                        "head": token.head.text,
                        "is_alpha": token.is_alpha,
                        "is_stop": token.is_stop,
                        "is_punct": token.is_punct,
                        "morphology": str(token.morph) if token.morph else ""
                    })
            
            return tokens[:50]  # İlk 50 token'ı döndür (performans için)
            
        except Exception as e:
            self.log_warning(f"Token analizi hatası: {e}")
            return []
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict]:
        """Anahtar kelime çıkarma - Mevcut basit TF-IDF benzeri yaklaşım"""
        try:
            # Metni işle
            doc = self.nlp(text)
            
            # Anlamlı kelimeleri al (stop word, punctuation, space olmayan)
            meaningful_tokens = [
                token.lemma_.lower() for token in doc 
                if not token.is_stop and not token.is_punct and not token.is_space 
                and token.is_alpha and len(token.text) > 2
            ]
            
            # Kelime frekansları
            word_freq = {}
            for word in meaningful_tokens:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # En sık kullanılan kelimeleri al
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            keywords = []
            for word, freq in sorted_words[:top_k]:
                keywords.append({
                    "word": word,
                    "frequency": freq,
                    "score": freq / len(meaningful_tokens) if meaningful_tokens else 0
                })
            
            return keywords
            
        except Exception as e:
            self.log_warning(f"Anahtar kelime çıkarma hatası: {e}")
            return []
    
    def _generate_summary(self, text: str) -> Dict:
        """Basit metin özeti - Cümle önem skoruna göre"""
        try:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            if len(sentences) <= 3:
                return {
                    "summary": text,
                    "compression_ratio": 1.0,
                    "method": "full_text"
                }
            
            # Cümle skorları (keyword yoğunluğuna göre)
            sentence_scores = []
            for sent in sentences:
                # Her cümledeki anlamlı kelime oranı
                meaningful_tokens = [
                    token for token in sent 
                    if not token.is_stop and not token.is_punct and not token.is_space
                ]
                score = len(meaningful_tokens) / len(sent) if len(sent) > 0 else 0
                sentence_scores.append((sent.text, score))
            
            # En yüksek skorlu cümleleri al
            sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
            summary_count = max(2, len(sentences) // 3)  # En az 2, maksimum 1/3'ü
            
            selected_sentences = sorted_sentences[:summary_count]
            summary_text = '. '.join([sent[0].strip() for sent in selected_sentences])
            
            return {
                "summary": summary_text,
                "compression_ratio": len(summary_text) / len(text),
                "method": "sentence_scoring",
                "original_sentences": len(sentences),
                "summary_sentences": len(selected_sentences)
            }
            
        except Exception as e:
            self.log_warning(f"Özet oluşturma hatası: {e}")
            return {"summary": text[:200] + "...", "method": "truncate"}
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """İki metin arasında benzerlik analizi"""
        try:
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # SpaCy ile vector benzerliği
            similarity = doc1.similarity(doc2) if doc1.vector_norm and doc2.vector_norm else 0.0
            
            return {
                "similarity": round(similarity, 3),
                "method": "spacy_vectors",
                "comparable": similarity > 0.1  # Eşik değer
            }
            
        except Exception as e:
            self.log_warning(f"Metin karşılaştırma hatası: {e}")
            return {"similarity": 0.0, "method": "error", "comparable": False}

    def detect_turkish_patterns(self, text: str) -> Dict:
        """
        Türkçe'ye özel konuşma kalıplarını tespit eder
        POLLY kuralları için Türkçe pattern tespiti
        """
        try:
            doc = self.nlp(text)
            patterns = {
                "hesitation_markers": [],
                "filler_words": [],
                "regional_dialect": [],
                "code_switching": [],
                "repeated_words": [],
                "incomplete_words": [],
                "informal_speech": []
            }
            
            # Tereddüt işaretleri (Hesitation markers)
            hesitation_patterns = [
                r'\b(eee+|ııı+|aaa+|ooo+|uuu+)\b',
                r'\b(şey|yani|işte|hani|bilirsin)\b',
                r'\b(nasıl desem|ne diyeyim|şöyle)\b'
            ]
            
            # Doldurma kelimeleri (Filler words)
            filler_patterns = [
                r'\b(hmm|hım|eh|ah|oh|of|uf)\b',
                r'\b(ıhh|öhh|ühh|ahh|ohh)\b'
            ]
            
            # Bölgesel ağız (Regional dialect)
            dialect_patterns = [
                r'\b(hele|yahu|valla|vallahi|ya bi)\b',
                r'\b(diyim|gidiyim|geliyim|oliyim)\s+mi\b',  # Ege ağzı
                r'\b(gitcem|gelcem|yapcem|olcak)\b',  # İstanbul ağzı
                r'\b(laan|yaa|bee)\b'  # Günlük konuşma
            ]
            
            # Kod değiştirme (Code switching)
            code_switch_patterns = [
                r'\b(sorry|thanks|okay|hello|bye|yes|no)\b',
                r'\b(actually|basically|like|you know)\b',
                r'\b(merci|bon|ciao|bitte)\b'
            ]
            
            # Tekrarlanan kelimeler (Repeated words)
            repeated_patterns = [
                r'\b(\w+)\s+\1\b',  # "bir bir", "ne ne"
                r'\b(\w+)\s+\1\s+\1\b'  # "çok çok çok"
            ]
            
            # Yarım kalmış kelimeler (Incomplete words)
            incomplete_patterns = [
                r'\b(gel|git|yap|ol|bil|gör|al)\b(?=\s)',  # Yarım fiiller
                r'\b\w+\-$',  # Tire ile biten
                r'\b\w{1,3}\.{2,}',  # Nokta nokta ile devam eden
            ]
            
            # Gayri resmi konuşma (Informal speech)
            informal_patterns = [
                r'\b(neyse|neyse işte|her neyse)\b',
                r'\b(falan|filan|felan|böyle|şöyle)\b',
                r'\b(yav|ya|abi|kardeş|kanka)\b'
            ]
            
            text_lower = text.lower()
            
            # Pattern'ları uygula
            pattern_groups = [
                ("hesitation_markers", hesitation_patterns),
                ("filler_words", filler_patterns), 
                ("regional_dialect", dialect_patterns),
                ("code_switching", code_switch_patterns),
                ("repeated_words", repeated_patterns),
                ("incomplete_words", incomplete_patterns),
                ("informal_speech", informal_patterns)
            ]
            
            import re
            for pattern_name, pattern_list in pattern_groups:
                for pattern in pattern_list:
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        patterns[pattern_name].append({
                            "match": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                            "context": text[max(0, match.start()-20):match.end()+20]
                        })
            
            # İstatistikler
            total_patterns = sum(len(v) for v in patterns.values())
            pattern_density = total_patterns / len(text.split()) if text.split() else 0
            
            return {
                "patterns": patterns,
                "statistics": {
                    "total_patterns": total_patterns,
                    "pattern_density": round(pattern_density, 3),
                    "dominant_pattern": max(patterns.keys(), key=lambda k: len(patterns[k])) if total_patterns > 0 else None
                },
                "quality_indicators": {
                    "is_formal_speech": pattern_density < 0.1,
                    "has_hesitation": len(patterns["hesitation_markers"]) > 0,
                    "has_dialect": len(patterns["regional_dialect"]) > 0,
                    "has_code_switching": len(patterns["code_switching"]) > 0,
                    "speech_completeness": 1.0 - min(1.0, len(patterns["incomplete_words"]) / 10)
                }
            }
            
        except Exception as e:
            self.log_error(f"Türkçe pattern tespiti hatası: {e}")
            return {"patterns": {}, "statistics": {}, "quality_indicators": {}}
    
    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """Metin gömme vektörleri - SpaCy word vectors"""
        try:
            doc = self.nlp(text)
            
            if doc.vector.size == 0:
                self.log_warning("Model word vectors içermiyor")
                return None
            
            # Document-level vector (ortalama word vectors)
            return doc.vector.tolist()
            
        except Exception as e:
            self.log_warning(f"Embedding hatası: {e}")
            return None
    
    def is_available(self) -> bool:
        """Model kullanılabilir mi?"""
        return self.nlp is not None
    
    def get_model_info(self) -> Dict:
        """Model bilgileri"""
        if not self.nlp:
            return {"error": "Model yüklü değil"}
        
        return {
            "model_name": self.nlp.meta.get("name", "Bilinmiyor"),
            "version": self.nlp.meta.get("version", "Bilinmiyor"),
            "language": self.nlp.meta.get("lang", "tr"),
            "pipeline": self.nlp.pipe_names,
            "vector_size": self.nlp.vocab.vectors.shape[1] if self.nlp.vocab.vectors.size > 0 else 0,
            "vocab_size": len(self.nlp.vocab)
        } 