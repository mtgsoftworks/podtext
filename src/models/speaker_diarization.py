"""
Konuşmacı tanıma ve ayırma modülü - pyannote.audio entegrasyonu
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import tempfile
import os
from ..utils.logger import LoggerMixin
from ..utils.config import Config


class SpeakerDiarizer(LoggerMixin):
    """Konuşmacı tanıma ve ayırma sınıfı"""
    
    def __init__(self, config: Config):
        self.config = config
        self.speaker_config = config.get_speaker_config()
        self.model_name = self.speaker_config.get("model", "pyannote/speaker-diarization-3.1")
        self.min_speakers = self.speaker_config.get("min_speakers", 1)
        self.max_speakers = self.speaker_config.get("max_speakers", 10)
        self.clustering_threshold = self.speaker_config.get("clustering_threshold", 0.7)
        
        self.pipeline = None
        
        # GPU zorla kullan
        force_gpu = config.get("performance.force_gpu", True)
        if force_gpu and not torch.cuda.is_available():
            self.log_warning("GPU zorla kullanım istendi ama CUDA mevcut değil!")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # GPU optimizasyonları
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True  # GPU performansı için
            torch.backends.cudnn.deterministic = False  # Hız için
            
            # Memory fraksiyonu ayarla
            gpu_memory_fraction = config.get("performance.gpu_memory_fraction", 0.9)
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            
            gpu_props = torch.cuda.get_device_properties(0)
            self.log_info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            self.log_info(f"GPU Memory fraction: {gpu_memory_fraction}")
        
        self.log_info(f"SpeakerDiarizer başlatıldı - Device: {self.device}")
    
    def _load_pipeline(self):
        """Pyannote pipeline'ını yükle"""
        if self.pipeline is None:
            try:
                # HuggingFace token'ı al
                hf_token = self.config.get_api_key("huggingface")
                if not hf_token:
                    raise ValueError("HuggingFace token bulunamadı!")
                
                self.log_info(f"Konuşmacı tanıma modeli yükleniyor: {self.model_name}")
                
                # GPU optimizasyonları ile pipeline yükle
                device_config = torch.device(self.device)
                
                # Unicode encoding sorunları için environment ayarları
                original_lang = os.environ.get('LANG')
                original_lc_all = os.environ.get('LC_ALL')
                
                try:
                    # Güvenli encoding ayarları
                    os.environ['LANG'] = 'en_US.UTF-8'
                    os.environ['LC_ALL'] = 'en_US.UTF-8'
                    
                    self.pipeline = Pipeline.from_pretrained(
                        self.model_name,
                        use_auth_token=hf_token,
                        cache_dir=self.config.get("performance.cache_dir", "models/")
                    ).to(device_config)
                    
                finally:
                    # Environment'ı restore et
                    if original_lang:
                        os.environ['LANG'] = original_lang
                    elif 'LANG' in os.environ:
                        del os.environ['LANG']
                        
                    if original_lc_all:
                        os.environ['LC_ALL'] = original_lc_all
                    elif 'LC_ALL' in os.environ:
                        del os.environ['LC_ALL']
                
                # GPU'ya gönder ve optimize et
                if self.device == "cuda":
                    self.pipeline.to(device_config)
                    
                    # Mixed precision kullan
                    if self.config.get("performance.mixed_precision", True):
                        # Half precision için model'i convert et
                        for module in self.pipeline._models.values():
                            if hasattr(module, 'half'):
                                module.half()
                    
                    # Batch size'ı optimize et
                    batch_size = self.config.get("performance.batch_size", 32)
                    if hasattr(self.pipeline, '_batch_size'):
                        self.pipeline._batch_size = batch_size
                    
                    # GPU memory kontrolü
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    used_memory = torch.cuda.memory_allocated() / 1024**3
                    self.log_info(f"Model GPU'ya yüklendi. Memory: {used_memory:.1f}/{gpu_memory:.1f} GB")
                    self.log_info(f"Mixed precision: {self.config.get('performance.mixed_precision', True)}")
                    self.log_info(f"Batch size: {batch_size}")
                
                self.log_info("Konuşmacı tanıma modeli başarıyla yüklendi")
                
            except UnicodeEncodeError as e:
                self.log_error(f"Unicode encoding hatası: {e}")
                self.log_warning("Konuşmacı tanıma atlanıyor - fallback moduna geçiliyor")
                self.pipeline = None
                raise
            except Exception as e:
                self.log_error(f"Konuşmacı tanıma modeli yüklenemedi: {e}")
                # HuggingFace bağlantı sorunu için özel handling
                if "latin-1" in str(e) or "codec" in str(e).lower():
                    self.log_warning("Encoding sorunu tespit edildi. Sistem ayarlarını kontrol edin.")
                    self.log_info("Geçici çözüm: LANG=en_US.UTF-8 export edin")
                raise
    
    def diarize_audio(self, audio_file: str) -> Dict:
        """
        Ses dosyasında konuşmacı ayırma işlemi yap
        
        Args:
            audio_file: Ses dosyası yolu
            
        Returns:
            Konuşmacı segmentleri bilgisi
        """
        self._load_pipeline()
        
        try:
            self.log_info(f"Konuşmacı ayırma başlatılıyor: {audio_file}")
            
            # Konuşmacı ayırma işlemi
            diarization = self.pipeline(
                audio_file,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            # Sonuçları işle
            result = self._process_diarization_result(diarization)
            
            self.log_info(f"Konuşmacı ayırma tamamlandı. {result['speaker_count']} konuşmacı bulundu")
            
            return result
            
        except Exception as e:
            self.log_error(f"Konuşmacı ayırma hatası: {e}")
            return {
                "segments": [],
                "speakers": [],
                "speaker_count": 0,
                "main_speaker": None,
                "error": str(e)
            }
    
    def _process_diarization_result(self, diarization: Annotation) -> Dict:
        """
        Konuşmacı ayırma sonuçlarını işle
        
        Args:
            diarization: Pyannote diarization sonucu
            
        Returns:
            İşlenmiş sonuçlar
        """
        segments = []
        speakers = set()
        speaker_durations = {}
        
        # Her segment için bilgileri topla
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_info = {
                "start_time": float(turn.start),
                "end_time": float(turn.end),
                "duration": float(turn.end - turn.start),
                "speaker": speaker
            }
            segments.append(segment_info)
            speakers.add(speaker)
            
            # Konuşmacı sürelerini hesapla
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0.0
            speaker_durations[speaker] += segment_info["duration"]
        
        # Ana konuşmacıyı belirle (en uzun süre konuşan)
        main_speaker = None
        if speaker_durations:
            main_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
        
        # Konuşmacı istatistikleri
        speaker_stats = []
        for speaker in sorted(speakers):
            duration = speaker_durations.get(speaker, 0.0)
            total_duration = sum(speaker_durations.values())
            percentage = (duration / total_duration * 100) if total_duration > 0 else 0
            
            speaker_stats.append({
                "speaker": speaker,
                "total_duration": duration,
                "percentage": percentage,
                "is_main": speaker == main_speaker
            })
        
        # Prepare initial result
        result = {
            "segments": segments,
            "speakers": sorted(list(speakers)),
            "speaker_count": len(speakers),
            "main_speaker": main_speaker,
            "speaker_stats": speaker_stats,
            "total_segments": len(segments)
        }
        # Filter minor speakers: if main speaker covers most of the time, treat as single speaker
        total_time = sum(speaker_durations.values()) or 0.0
        main_time = speaker_durations.get(main_speaker, 0.0)
        # threshold (ratio) for filtering minor speakers
        filter_threshold = self.config.get("speaker_diarization.filter_threshold", 0.95)
        if result["speaker_count"] > 1 and total_time > 0 and (main_time / total_time) >= filter_threshold:
            self.log_info(f"Minor speaker filter applied: main speaker covers {(main_time/total_time):.2f}, threshold {filter_threshold}")
            # override to single speaker
            result.update({
                "segments": [{
                    "start_time": 0.0,
                    "end_time": total_time,
                    "duration": total_time,
                    "speaker": main_speaker
                }],
                "speakers": [main_speaker],
                "speaker_count": 1,
                "speaker_stats": [{
                    "speaker": main_speaker,
                    "total_duration": total_time,
                    "percentage": 100.0,
                    "is_main": True
                }],
                "total_segments": 1
            })
        return result
    
    def get_main_speaker_segments(self, diarization_result: Dict) -> List[Dict]:
        """
        Ana konuşmacının segmentlerini al
        
        Args:
            diarization_result: Konuşmacı ayırma sonucu
            
        Returns:
            Ana konuşmacı segmentleri
        """
        main_speaker = diarization_result.get("main_speaker")
        if not main_speaker:
            return []
        
        main_segments = [
            segment for segment in diarization_result.get("segments", [])
            if segment["speaker"] == main_speaker
        ]
        
        return main_segments
    
    def filter_by_speaker(self, diarization_result: Dict, speaker: str) -> List[Dict]:
        """
        Belirli bir konuşmacının segmentlerini filtrele
        
        Args:
            diarization_result: Konuşmacı ayırma sonucu
            speaker: Hedef konuşmacı
            
        Returns:
            Filtrelenmiş segmentler
        """
        return [
            segment for segment in diarization_result.get("segments", [])
            if segment["speaker"] == speaker
        ]
    
    def get_speaker_timeline(self, diarization_result: Dict) -> str:
        """
        Konuşmacı zaman çizelgesi oluştur
        
        Args:
            diarization_result: Konuşmacı ayırma sonucu
            
        Returns:
            Formatlanmış zaman çizelgesi
        """
        segments = diarization_result.get("segments", [])
        timeline = []
        
        for segment in segments:
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            speaker = segment["speaker"]
            
            timeline.append(
                f"{start_time:06.1f}s - {end_time:06.1f}s: {speaker}"
            )
        
        return "\n".join(timeline)
    
    def merge_short_segments(self, diarization_result: Dict, min_duration: float = 1.0) -> Dict:
        """
        Kısa segmentleri birleştir
        
        Args:
            diarization_result: Konuşmacı ayırma sonucu
            min_duration: Minimum segment süresi (saniye)
            
        Returns:
            Birleştirilmiş sonuçlar
        """
        segments = diarization_result.get("segments", [])
        if not segments:
            return diarization_result
        
        merged_segments = []
        current_segment = segments[0].copy()
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            
            # Aynı konuşmacı ve kısa süre varsa birleştir
            if (next_segment["speaker"] == current_segment["speaker"] and
                current_segment["duration"] < min_duration):
                
                current_segment["end_time"] = next_segment["end_time"]
                current_segment["duration"] = (
                    current_segment["end_time"] - current_segment["start_time"]
                )
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        # Son segmenti ekle
        merged_segments.append(current_segment)
        
        # Sonuçları güncelle
        result = diarization_result.copy()
        result["segments"] = merged_segments
        result["total_segments"] = len(merged_segments)
        
        return result
    
    def get_overlap_segments(self, diarization_result: Dict, threshold: float = 0.1) -> List[Dict]:
        """
        Çakışan konuşma segmentlerini bul
        
        Args:
            diarization_result: Konuşmacı ayırma sonucu
            threshold: Çakışma eşik değeri (saniye)
            
        Returns:
            Çakışan segmentler
        """
        segments = diarization_result.get("segments", [])
        overlaps = []
        
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments[i+1:], i+1):
                # Çakışma kontrolü
                overlap_start = max(seg1["start_time"], seg2["start_time"])
                overlap_end = min(seg1["end_time"], seg2["end_time"])
                
                if overlap_end - overlap_start > threshold:
                    overlaps.append({
                        "start_time": overlap_start,
                        "end_time": overlap_end,
                        "duration": overlap_end - overlap_start,
                        "speakers": [seg1["speaker"], seg2["speaker"]]
                    })
        
        return overlaps
    
    def export_to_rttm(self, diarization_result: Dict, output_file: str, audio_file: str):
        """
        RTTM formatında dışa aktar
        
        Args:
            diarization_result: Konuşmacı ayırma sonucu
            output_file: Çıktı dosyası yolu
            audio_file: Ses dosyası adı
        """
        segments = diarization_result.get("segments", [])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                # RTTM format: SPEAKER <file> <channel> <start> <duration> <ortho> <stype> <speaker> <conf> <slat>
                line = f"SPEAKER {audio_file} 1 {segment['start_time']:.3f} {segment['duration']:.3f} <NA> <NA> {segment['speaker']} <NA> <NA>\n"
                f.write(line)
        
        self.log_info(f"RTTM dosyası oluşturuldu: {output_file}") 