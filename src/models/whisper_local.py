"""
Yerel Whisper modeli - OpenAI orijinal whisper entegrasyonu (pywhispercpp ile GGML desteği)
"""
import os
import torch
import whisper
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from pywhispercpp.model import Model as WhisperCppModel
import tempfile
import subprocess
from ..utils.logger import LoggerMixin
from ..utils.config import Config
import psutil
import time
import threading

try:
    from pywhispercpp.model import Model as WhisperCppModel  # type: ignore
    PYWHISPERCPP_AVAILABLE = True
except ImportError:
    WhisperCppModel = None
    PYWHISPERCPP_AVAILABLE = False

try:
    import whisper  # type: ignore
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False


class LocalWhisperModel(LoggerMixin):
    """Yerel Whisper modeli sınıfı (OpenAI orijinal whisper + pywhispercpp ile GGML desteği)"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.whisper_config = config.get("whisper", {})
        self.performance_config = config.get("performance", {})
        
        # CPU kontrolü parametreleri
        self.cpu_limit = self.performance_config.get("cpu_usage_limit", 0.9)
        self.auto_adjust_threads = self.performance_config.get("auto_adjust_threads", True)
        self.cpu_monitor_active = False
        
        # Model ayarları
        self.model_size = self.whisper_config.get("model", "large-v3")
        self.language = self.whisper_config.get("language", "tr")
        self.task = self.whisper_config.get("task", "transcribe")
        self.temperature = self.whisper_config.get("temperature", 0.0)
        self.beam_size = self.whisper_config.get("beam_size", 5)
        self.best_of = self.whisper_config.get("best_of", 5)
        
        # GGML model path
        self.models_dir = Path("models")
        self.ggml_model_path = self.models_dir / "ggml-large-v3-turbo.bin"
        
        # GPU ayarları
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # GPU optimizasyonları
        if torch.cuda.is_available():
            # Mixed precision kullan
            if config.get("performance.mixed_precision", True):
                torch.set_default_dtype(torch.float16)
            
            # GPU memory optimizasyonu
            gpu_memory_fraction = config.get("performance.gpu_memory_fraction", 0.9)
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        self.model = None
        self.log_info(f"LocalWhisperModel başlatıldı - Device: {self.device}")
        self.log_info(f"GGML Model Path: {self.ggml_model_path}")
        
        # Model seçim mantığı
        self.use_ggml_model = False
        
        if self._check_ggml_model_exists():
            self.log_info(f"GGML Model Path: {os.path.join('models', f'ggml-{self.model_size}.bin')}")
            self.use_ggml_model = True
            self.log_info("Yerel Whisper modeli (GPU optimizeli) kullanılıyor")
            self._load_model()
        else:
            existing_model = self._check_existing_whisper_models()
            if existing_model:
                self.log_info(f"Mevcut OpenAI Whisper modeli bulundu: {existing_model}")
                self.model_size = existing_model
                self._load_model()
            else:
                self.log_info("Kullanılabilir model bulunamadı, gerektiğinde yüklenecek")
    
    def _check_ggml_model_exists(self) -> bool:
        """GGML modelinin varlığını kontrol et"""
        exists = self.ggml_model_path.exists()
        if exists:
            size_mb = self.ggml_model_path.stat().st_size / (1024 * 1024)
            self.log_info(f"GGML modeli bulundu: {self.ggml_model_path} ({size_mb:.1f} MB)")
        else:
            self.log_warning(f"GGML modeli bulunamadı: {self.ggml_model_path}")
        return exists

    def _check_existing_whisper_models(self) -> Optional[str]:
        """Models klasöründeki mevcut Whisper modellerini kontrol et"""
        model_patterns = {
            "large-v3": ["large-v3.pt"],
            "large-v2": ["large-v2.pt"],
            "large": ["large.pt"],
            "medium": ["medium.pt"],
            "small": ["small.pt"],
            "base": ["base.pt"],
            "tiny": ["tiny.pt"]
        }
        
        # İstenen model boyutuna göre kontrol et
        size_key = self.model_size.replace("-turbo", "")  # large-v3-turbo -> large-v3
        
        if size_key in model_patterns:
            for pattern in model_patterns[size_key]:
                model_path = self.models_dir / pattern
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    self.log_info(f"Mevcut Whisper Pytorch modeli bulundu: {model_path} ({size_mb:.1f} MB)")
                    return str(model_path)
        
        return None

    def _load_model(self):
        """
        Whisper modelini yükle - SADECE yerel GGML modelini kullanmaya zorla.
        Fallback veya indirme yok.
        """
        if self.model is not None:
            return

        self.log_info("Model yükleme mantığı: Sadece yerel GGML modeli kullanılacak.")

        # 1. GGML modelinin varlığını kontrol et
        if not self._check_ggml_model_exists():
            raise RuntimeError(
                f"Zorunlu model dosyası bulunamadı: {self.ggml_model_path.resolve()}\n"
                "Lütfen 'ggml-large-v3-turbo.bin' dosyasının 'models' klasöründe olduğundan emin olun. "
                "Program başka bir model denemeyecek veya indirme yapmayacaktır."
            )

        # 2. GGML modelini yüklemeyi dene
        self.log_info("GGML modeli bulundu, pywhispercpp ile yükleniyor...")
        try:
            self.model = self._load_ggml_model()
            self.log_info(f"GGML modeli '{self.ggml_model_path.name}' başarıyla yüklendi.")
            self.log_info("Sistem sadece bu modeli kullanacak şekilde yapılandırıldı.")
        except Exception as e:
            self.log_error(f"GGML modeli yüklenirken kritik bir hata oluştu: {e}", exc_info=True)
            raise RuntimeError(
                f"GGML modeli '{self.ggml_model_path.name}' bulundu ancak yüklenemedi. "
                "Dosyanın bozuk olmadığını veya pywhispercpp ile uyumlu olduğunu kontrol edin. "
                "Program devam edemiyor."
            )

    def _load_ggml_model(self) -> Optional[Any]:
        """GGML modelini yükle"""
        if not PYWHISPERCPP_AVAILABLE or WhisperCppModel is None:
            self.log_error("pywhispercpp kütüphanesi bulunamadı!")
            return None
            
        default_params = self.whisper_config.get("pywhispercpp_params", {})
        self.log_info(f"pywhispercpp için kullanılan model başlatma parametreleri: {default_params}")
        # Model'i başlat (dile dair parametreler transcribe sırasında verilecek)
        model = WhisperCppModel(str(self.ggml_model_path.resolve()), **default_params)
        return model

    def transcribe_audio(self, audio_file: str) -> Dict:
        """
        Ses dosyasını transkript et
        
        Args:
            audio_file: Transkript edilecek ses dosyası yolu
            
        Returns:
            Transkripsiyon sonucu
        """
        try:
            # CPU monitörünü başlat
            self._start_cpu_monitor()
            
            # Model tipini daha güvenilir şekilde belirle
            is_ggml_model = self.use_ggml_model and isinstance(self.model, WhisperCppModel)
            
            if is_ggml_model:
                self.log_info("GGML (pywhispercpp) modeli ile transkripsiyon yapılıyor.")
                
                # pywhispercpp parametrelerini hazırla
                params = self.whisper_config.get("pywhispercpp_transcribe_params", {}).copy()
                
                # pywhispercpp desteklemeyen parametreleri kaldır
                unsupported_params = ["word_timestamps", "response_format"]
                for param in unsupported_params:
                    if param in params:
                        self.log_info(f"pywhispercpp desteklemiyor, parametre kaldırıldı: {param}")
                        params.pop(param)
                
                # Türkçe dilini zorla ve çeviriyi kapat
                params["language"] = "tr"
                params["translate"] = False
                
                # CPU kullanımına göre optimal thread ve processor sayısını ayarla
                if self.auto_adjust_threads:
                    optimal_threads = self._get_optimal_thread_count()
                    optimal_processors = max(2, optimal_threads // 2)
                    params["n_processors"] = optimal_processors
                    self.log_info(f"🎯 CPU kontrolü: n_processors={optimal_processors} (limit: %{self.cpu_limit*100:.0f})")
                
                # Çok işlem (paralel process) sayısını parametrelerden çek, yoksa CPU çekirdek sayısını kullan
                n_processors = params.pop("n_processors", os.cpu_count())
                self.log_info(f"pywhispercpp transkripsiyon parametreleri: {params}, n_processors={n_processors}")
                
                result = self.model.transcribe(audio_file, n_processors=n_processors, **params)
            else:
                self.log_info("OpenAI Whisper (Pytorch) modeli ile transkripsiyon yapılıyor.")
                if self.model is None:
                    self.log_error("Model yüklenmemiş!")
                    return {"text": "", "segments": [], "error": "Model not loaded"}
                    
                result = self.model.transcribe(
                    audio_file,
                    language=self.language,
                    task=self.task,
                    temperature=self.temperature,
                    fp16=False  # CPU kullanımı için
                )
            
            self.log_info("Transkripsiyon tamamlandı.")
            model_name = "GGML" if is_ggml_model else "OpenAI Whisper"
            return self._process_transcription_result(result, model_name)
            
        except Exception as e:
            self.log_error(f"Transkripsiyon sırasında hata: {e}", exc_info=True)
            return {"text": "", "segments": [], "error": str(e)}
        finally:
            # CPU monitörünü durdur
            self._stop_cpu_monitor()

    def _process_transcription_result(self, result, model_name):
        """Transkripsiyon sonucunu işle ve standardize et"""
        processed_result = {
            'text': '',
            'segments': [],
            'language': 'tr',
            'model_used': model_name,
            'word_confidence': [],
            'hallucination_risk': 0.0,
            'confidence_stats': {}
        }
        
        # Model tipini güvenilir biçimde tespit et
        is_ggml_model = isinstance(self.model, WhisperCppModel)
        
        if is_ggml_model:
            def _seg_to_dict(seg):
                return {
                    'start': getattr(seg, 'start', 0.0),
                    'end': getattr(seg, 'end', 0.0),
                    'text': getattr(seg, 'text', str(seg))
                }

            if isinstance(result, list):
                # Segment listesi -> dict'e dönüştür
                processed_result['segments'] = [_seg_to_dict(s) for s in result]
                processed_result['text'] = " ".join(seg['text'] for seg in processed_result['segments'])
            elif hasattr(result, 'text'):
                processed_result['text'] = result.text
                if hasattr(result, 'segments'):
                    processed_result['segments'] = [_seg_to_dict(s) for s in result.segments]
                else:
                    processed_result['segments'] = [_seg_to_dict(result)]
        else:
            # OpenAI Whisper formatı
            processed_result['text'] = result.get('text', '')
            processed_result['segments'] = result.get('segments', [])
            processed_result['language'] = result.get('language', 'tr')
            
            # Word-level confidence extraction (sadece OpenAI Whisper için)
            self._extract_word_confidence(result, processed_result)
        
        # Hallucination detection (her iki model için)
        hallucination_analysis = self._detect_hallucination(processed_result['segments'])
        processed_result['hallucination_risk'] = hallucination_analysis.get('risk_score', 0.0)
        processed_result['hallucination_analysis'] = hallucination_analysis
        
        return processed_result
    
    def _get_optimal_thread_count(self) -> int:
        """CPU kullanımına göre optimal thread sayısı hesapla"""
        cpu_count = os.cpu_count() or 4
        cpu_percent = psutil.cpu_percent(interval=1.0)
        
        if cpu_percent > self.cpu_limit * 100:
            # CPU kullanımı yüksekse thread sayısını azalt
            optimal_threads = max(2, int(cpu_count * 0.5))
            self.log_info(f"CPU kullanımı yüksek (%{cpu_percent:.1f}), thread sayısı azaltıldı: {optimal_threads}")
        else:
            # Normal durumda CPU limitine göre ayarla
            optimal_threads = max(2, int(cpu_count * self.cpu_limit))
            
        return optimal_threads
    
    def _start_cpu_monitor(self):
        """CPU kullanım monitörünü başlat"""
        if not self.auto_adjust_threads or self.cpu_monitor_active:
            return
            
        self.cpu_monitor_active = True
        
        def monitor_cpu():
            while self.cpu_monitor_active:
                try:
                    cpu_percent = psutil.cpu_percent(interval=5.0)
                    if cpu_percent > self.cpu_limit * 100:
                        self.log_info(f"⚠️ CPU kullanımı limit aşıldı: %{cpu_percent:.1f} (limit: %{self.cpu_limit*100:.0f})")
                    time.sleep(10)
                except Exception as e:
                    self.log_error(f"CPU monitör hatası: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
        monitor_thread.start()
        self.log_info(f"CPU monitörü başlatıldı (limit: %{self.cpu_limit*100:.0f})")
    
    def _stop_cpu_monitor(self):
        """CPU monitörünü durdur"""
        self.cpu_monitor_active = False
    
    def _extract_word_confidence(self, result, processed_result):
        """OpenAI Whisper sonuçlarından word-level confidence çıkar"""
        word_confidence = []
        
        for segment in result.get('segments', []):
            if 'words' in segment:
                for word_info in segment['words']:
                    word_confidence.append({
                        'word': word_info.get('word', '').strip(),
                        'start': word_info.get('start', 0.0),
                        'end': word_info.get('end', 0.0),
                        'probability': word_info.get('probability', 0.5)
                    })
        
        processed_result['word_confidence'] = word_confidence
        
        # Confidence istatistikleri
        if word_confidence:
            confidences = [w['probability'] for w in word_confidence]
            processed_result['confidence_stats'] = {
                'mean': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'low_confidence_count': len([c for c in confidences if c < 0.3])
            }
    
    def transcribe_with_speaker_diarization(self, audio_file: str, diarization_result: Dict) -> Dict:
        """
        Konuşmacı bilgileri ile birlikte transkript et
        
        Args:
            audio_file: Ses dosyası yolu
            diarization_result: Speaker diarization sonucu
            
        Returns:
            Konuşmacı etiketli transkripsiyon sonucu
        """
        # Normal transkripsiyon yap
        transcription_result = self.transcribe_audio(audio_file)
        
        if transcription_result.get("error"):
            return transcription_result
        
        # Konuşmacı etiketlerini ekle
        segments_with_speakers = []
        
        for segment in transcription_result.get("segments", []):
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            
            # Bu segment için konuşmacıyı bul
            speaker = self._find_speaker_for_segment(start_time, end_time, diarization_result)
            
            segment_with_speaker = segment.copy()
            segment_with_speaker["speaker"] = speaker
            segments_with_speakers.append(segment_with_speaker)
        
        # Sonuçları güncelle
        transcription_result["segments"] = segments_with_speakers
        
        # Konuşmacılı metni oluştur
        speaker_labeled_text = ""
        current_speaker = None
        
        for segment in segments_with_speakers:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            
            if speaker != current_speaker:
                if speaker_labeled_text:
                    speaker_labeled_text += "\n\n"
                speaker_labeled_text += f"[{speaker}]: "
                current_speaker = speaker
            
            speaker_labeled_text += text + " "
        
        transcription_result["speaker_labeled_text"] = speaker_labeled_text.strip()
        
        # Yalnızca ana konuşmacının transkripsiyonunu al
        main_speaker = diarization_result.get("main_speaker")
        main_segments = [seg for seg in segments_with_speakers if seg.get("speaker") == main_speaker]
        transcription_result["segments"] = main_segments
        # Ana konuşmacının birleştirilmiş metni
        transcription_result["text"] = " ".join(seg.get("text", "") for seg in main_segments).strip()
        
        return transcription_result
    
    def _find_speaker_for_segment(self, start_time: float, end_time: float, diarization_result: Dict) -> str:
        """
        Segment için en uygun konuşmacıyı bul
        
        Args:
            start_time: Segment başlangıç zamanı
            end_time: Segment bitiş zamanı
            diarization_result: Konuşmacı ayırma sonucu
            
        Returns:
            Konuşmacı adı
        """
        segment_duration = end_time - start_time
        speaker_durations = {}
        
        # Her konuşmacı için overlap süresini hesapla
        for diar_segment in diarization_result.get("segments", []):
            diar_start = diar_segment["start_time"]
            diar_end = diar_segment["end_time"]
            speaker = diar_segment["speaker"]
            
            # Overlap hesapla
            overlap_start = max(start_time, diar_start)
            overlap_end = min(end_time, diar_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0
            speaker_durations[speaker] += overlap_duration
        
        # En çok overlap olan konuşmacıyı döndür
        if speaker_durations:
            return max(speaker_durations.items(), key=lambda x: x[1])[0]
        
        return "UNKNOWN"
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini al"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": "GGML" if self.model is None else "OpenAI Whisper",
            "language": self.language,
            "task": self.task,
            "is_loaded": self.model is not None
        }
    
    def format_for_editing(self, transcription_result: Dict) -> str:
        """
        Düzenleme için formatlanmış metni döndür.
        Konuşmacı etiketli metin varsa onu, yoksa ham metni kullan.
        """
        # Eğer konuşmacı etiketli metin varsa kullan
        if transcription_result.get("speaker_labeled_text"):
            return transcription_result.get("speaker_labeled_text")
        # Aksi halde düz metni döndür
        return transcription_result.get("text", "")

    def _detect_hallucination(self, segments: List[Dict]) -> Dict:
        """Şimdilik her zaman 'halüsinasyon yok' sonucu döndüren sade fonksiyon."""
        return {"risk_score": 0.0, "details": []}

    def detect_hallucinations(self, segments: List[Dict], language: str) -> List[Dict]:
        """Transkripsiyon segmentlerinde olası halüsinasyonları tespit eder."""
        try:
            from whisper_normalizer.indic_normalizer import IndicNormalizer
            from whisper_normalizer.english_normalizer import EnglishNormalizer
            
            if language == "tr":
                # Türkçe için özel normalleştirici gerekebilir, şimdilik temel karakterler
                normalizer = lambda x: x
            elif language in ["en", "english"]:
                normalizer = EnglishNormalizer()
            else:
                 # Diğer diller için genel bir normalleştirici
                normalizer = IndicNormalizer()

            hallucinations = []
            for segment in segments:
                # Segment nesnesinden metni doğru al
                text = segment.text if hasattr(segment, 'text') else segment.get('text', '')
                normalized_text = normalizer(text)
                
                # Basit halüsinasyon tespiti kuralları (örnek)
                # ... (buraya daha karmaşık kurallar eklenebilir)
                
            return hallucinations
            
        except ImportError:
            self.log_warning("Halüsinasyon tespiti için 'whisper_normalizer' kurulu değil, atlanıyor.")
            return []
        except Exception as e:
            self.log_error(f"Hallucination tespiti hatası: {e}")
            return []

    def _apply_language_rules(self, transcription_result: Dict) -> Dict:
        """Türkçe metin üzerinde basit dil temizliği yapabilir.
        Şimdilik doğrudan gelen nesneyi döndürür."""
        return transcription_result 