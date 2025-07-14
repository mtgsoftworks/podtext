"""
Ana orkestratör modülü - tüm bileşenleri koordine eden merkezi sınıf
"""
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TimeElapsedColumn
from langdetect import detect, LangDetectException  # language detection for incorrect locale
import numpy as np
import time
import json
from datetime import datetime
import asyncio
import concurrent.futures
from functools import partial
import gc
import psutil

from ..utils.logger import LoggerMixin
from ..utils.config import Config
from ..audio.processor import AudioProcessor
from ..models.whisper_local import LocalWhisperModel
from ..models.speaker_diarization import SpeakerDiarizer
from ..models.gemini_model import GeminiAnalyzer
from ..quality.evaluator import AudioQualityEvaluator
from ..labeling.auto_labeler import AutoLabeler
from ..nlp.turkish_analyzer import TurkishNLPAnalyzer


class PodcastTranscriptionOrchestrator(LoggerMixin):
    """
    Podcast transkripsiyon sürecinin tamamını yönetir.
    Ses işleme, konuşmacı ayırma, transkripsiyon, kalite analizi ve çıktı
    oluşturma adımlarını koordine eder.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Orchestrator'ı başlatır.

        Args:
            config_path: Konfigürasyon dosyasının yolu.
        """
        super().__init__()
        self.config = Config(config_path)
        self.console = Console()
        
        # Bileşenleri başlat
        try:
            self.log_info("AudioProcessor başlatıldı")
            self.audio_processor = AudioProcessor(self.config)
            
            self.log_info("LocalWhisperModel başlatıldı")
            self.whisper_model = LocalWhisperModel(self.config)
            
            self.log_info("SpeakerDiarizer başlatıldı")
            self.speaker_diarizer = SpeakerDiarizer(self.config)
            
            self.log_info("GeminiAnalyzer başlatıldı")
            self.gemini_analyzer = GeminiAnalyzer(self.config)
            
            self.log_info("AudioQualityEvaluator başlatıldı")
            self.quality_evaluator = AudioQualityEvaluator(self.config)
            
            self.log_info("AutoLabeler başlatıldı")
            self.auto_labeler = AutoLabeler(self.config)
            
            self.log_info("TurkishNLPAnalyzer başlatıldı")
            self.turkish_analyzer = TurkishNLPAnalyzer(self.config.get("nlp", {}))
            
            # Sistem performans bilgileri
            performance_config = self.config.get("performance", {})
            cpu_limit = performance_config.get("cpu_usage_limit", 0.9) * 100
            self.log_info(f"🎯 CPU kullanım limiti: %{cpu_limit:.0f}")

            self.log_info("Tüm bileşenler başarıyla başlatıldı.")
            
        except Exception as e:
            self.log_error(f"Bileşen başlatma hatası: {e}")
            raise
    
    def process_podcast(self, audio_file: str, interactive: bool = True, save_intermediate: bool = False) -> Dict:
        """
        Ana podcast işleme fonksiyonu.

        Args:
            audio_file: İşlenecek ses dosyasının yolu.
            interactive: İnteraktif mod (kullanıcıya soru sorar).
            save_intermediate: Ara işlem sonuçlarını kaydet.

        Returns:
            İşlem sonuçlarını içeren bir sözlük.
        """
        results = {}
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("İşlem adımları", total=8)
            
            try:
                # 1. Ses dosyasını işle ve hazırla
                progress.update(task, description="Ses dosyası hazırlanıyor...", advance=1)
                
                # Paralel ses işleme ve bilgi toplama
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    # Audio info ve processing'i paralel çalıştır
                    audio_info_future = executor.submit(self.audio_processor.get_audio_info, audio_file)
                    audio_load_future = executor.submit(self.audio_processor.load_audio, audio_file)
                    
                    # Sonuçları al
                    audio_info = audio_info_future.result()
                    audio_data, sr = audio_load_future.result()
                    
                    # Audio processing pipeline'ını paralel çalıştır
                    processing_futures = []
                    if self.config.get("audio_processing.noise_reduction.enabled", False):
                        processing_futures.append(
                            executor.submit(self.audio_processor.reduce_noise, audio_data, sr)
                        )
                    
                    # Processing sonuçlarını al
                    if processing_futures:
                        audio_data = processing_futures[0].result()
                    
                    # Normalizasyon
                    if self.config.get("audio_processing.normalization.enabled", False):
                        target_lufs = self.config.get("audio_processing.normalization.target_lufs", -23.0)
                        audio_data = self.audio_processor.normalize_audio(audio_data, target_lufs)
                    
                    # İşlenmiş sesi geçici dosyaya kaydet
                    processed_audio_path = self.audio_processor.save_temp_audio(audio_data, sr)
                    audio_info["processed_path"] = processed_audio_path

                    if interactive:
                        self._display_audio_info(audio_info)

                # 2. Ses Kalitesi Değerlendirmesi
                progress.update(task, description="Ses kalitesi ölçülüyor...", advance=1)
                quality_result = self.quality_evaluator.evaluate_audio_quality(processed_audio_path)
                results["audio_quality"] = quality_result
                if interactive:
                    self._display_quality_assessment(quality_result)

                # 3. Konuşmacı ayırma (çok kısa dosyalarda optimize et)
                progress.update(task, description="Konuşmacılar tespit ediliyor...", advance=1)
                if audio_info.get('sure_saniye', 0) < 5:
                    self.log_info("Ses dosyası 5 saniyeden kısa, performans için konuşmacı ayırma atlanıyor.")
                    speaker_result = {
                        "segments": [{
                            "start_time": 0,
                            "end_time": audio_info.get('sure_saniye', 0),
                            "duration": audio_info.get('sure_saniye', 0),
                            "speaker": "SPEAKER_00"
                        }],
                        "speakers": ["SPEAKER_00"],
                        "speaker_count": 1,
                        "main_speaker": "SPEAKER_00",
                        "skipped": True
                    }
                else:
                    try:
                        speaker_result = self.speaker_diarizer.diarize_audio(processed_audio_path)
                    except (UnicodeEncodeError, ConnectionError, ValueError) as e:
                        self.log_error(f"Konuşmacı ayırma başarısız: {e}")
                        self.log_warning("Single speaker moduna geçiliyor...")
                        speaker_result = {
                            "segments": [{
                                "start_time": 0,
                                "end_time": audio_info.get('sure_saniye', 0),
                                "duration": audio_info.get('sure_saniye', 0),
                                "speaker": "SPEAKER_00"
                            }],
                            "speakers": ["SPEAKER_00"],
                            "speaker_count": 1,
                            "main_speaker": "SPEAKER_00",
                            "fallback": True,
                            "fallback_reason": str(e)
                        }
                results["speaker_diarization"] = speaker_result

                if interactive:
                    # Konuşmacı bilgilerini göster
                    temp_main_speaker = speaker_result.get("main_speaker", speaker_result.get("speakers", ["SPEAKER_00"])[0])
                    self._display_speaker_info(speaker_result, temp_main_speaker)
                    
                    # Kullanıcıdan ana konuşmacı seçimini al
                    main_speaker = self._interactive_main_speaker_selection(speaker_result)
                    results["selected_main_speaker"] = main_speaker
                else:
                    # Non-interactive modda varsayılan ana konuşmacıyı kullan
                    main_speaker = speaker_result.get("main_speaker", speaker_result.get("speakers", ["SPEAKER_00"])[0])
                    results["selected_main_speaker"] = main_speaker

                # 4. Transkripsiyon
                progress.update(task, description="Transkripsiyon başlıyor...", advance=1)
                
                # Paralel transkripsiyon seçenekleri
                enable_comparison = self.config.get("transcription.enable_model_comparison", False)
                
                if enable_comparison:
                    # Çoklu model karşılaştırması
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        transcription_futures = {}
                        
                        # Tüm mevcut modelleri paralel çalıştır
                        if hasattr(self, 'whisper_openai'):
                            transcription_futures["whisper_openai"] = executor.submit(
                                self.whisper_openai.transcribe_audio, processed_audio_path
                            )
                        if hasattr(self, 'whisper_local'):
                            transcription_futures["whisper_local"] = executor.submit(
                                self.whisper_local.transcribe_audio, processed_audio_path
                            )
                        # Ana whisper_transcriber de dahil et
                        transcription_futures["whisper_transcriber"] = executor.submit(
                            self.whisper_model.transcribe_with_speaker_diarization,
                            processed_audio_path, speaker_result
                        )
                        
                        # Sonuçları topla
                        transcription_results = {}
                        for model_name, future in transcription_futures.items():
                            try:
                                transcription_results[model_name] = future.result(timeout=300)
                                self.log_info(f"{model_name} transkripsiyon tamamlandı")
                            except Exception as e:
                                self.log_error(f"{model_name} transkripsiyon hatası: {e}")
                                transcription_results[model_name] = None
                        
                        # Ana sonucu kullan (whisper_transcriber)
                        transcription_result = transcription_results.get("whisper_transcriber")
                        results["model_comparison"] = transcription_results
                        
                else:
                    # Standart tek model kullanımı
                    progress.update(task, description="Transkripsiyon yapılıyor...", advance=1)
                    transcription_result = self.whisper_model.transcribe_with_speaker_diarization(
                        processed_audio_path, speaker_result
                    )
                
                # Sadece ana konuşmacıyı transkribe etme kuralı
                if main_speaker and speaker_result.get("speaker_count", 0) > 1:
                    self.log_info(f"Yalnızca ana konuşmacı ({main_speaker}) için transkripsiyon filtreleniyor...")
                    original_segments = transcription_result.get("segments", [])
                    filtered_segments = [
                        seg for seg in original_segments if seg.get("speaker") == main_speaker
                    ]
                    
                    # Filtrelenmiş segmentlerden metni yeniden oluştur
                    filtered_text = " ".join([seg.get("text", "").strip() for seg in filtered_segments])
                    
                    transcription_result["segments"] = filtered_segments
                    transcription_result["text"] = filtered_text
                    transcription_result["is_filtered_by_main_speaker"] = True
                    self.log_info(f"Filtrelenmiş metin: {filtered_text[:100]}...")

                results["transcription"] = transcription_result
                
                # Yanlış dil kuralı uygulaması
                self._apply_foreign_language_rule(transcription_result)

                # Kesilme tespiti
                truncation_info = self._detect_truncation(audio_data, sr)
                if truncation_info["is_truncated"]:
                    self.log_info(truncation_info["reason"])
                results["truncation_info"] = truncation_info

                # 5. Gelişmiş Kalite Analizi (Gemini)
                progress.update(task, description="AI kalite analizi yapılıyor...", advance=1)
                
                # 5-8. Paralel kalite analizi ve düzeltme işlemleri
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    # Eş zamanlı işlemler
                    labeling_future = executor.submit(
                        self.auto_labeler.suggest_labels,
                        transcription_result, quality_result, speaker_result, truncation_info
                    )
                    
                    polly_future = executor.submit(
                        self.auto_labeler.process_polly_compliance, transcription_result
                    )
                    
                    multiple_voices_future = executor.submit(
                        self.auto_labeler.analyze_multiple_voices_compliance,
                        speaker_result, quality_result
                    )
                    
                    wrong_lang_future = None
                    if transcription_result.get("text"):
                        wrong_lang_future = executor.submit(
                            self._check_wrong_language_polly_rule, transcription_result["text"]
                        )
                    
                    # İlk aşama sonuçlarını al
                    label_suggestions_result = labeling_future.result()
                    transcription_result = polly_future.result()
                    multiple_voices_analysis = multiple_voices_future.result()
                    
                    if wrong_lang_future:
                        wrong_language_check = wrong_lang_future.result()
                        results["wrong_language_analysis"] = wrong_language_check
                    
                    results["multiple_voices_analysis"] = multiple_voices_analysis
                    results["label_suggestions"] = label_suggestions_result
                    
                    # Label önerilerini al
                    label_suggestions_list = label_suggestions_result.get("suggestions", [])
                    
                    # İkinci aşama: Gemini analizleri paralel
                    gemini_quality_future = executor.submit(
                        self.gemini_analyzer.evaluate_audio_and_transcription_quality,
                        transcription_result["text"], audio_info, label_suggestions_list
                    )
                    
                    correction_future = executor.submit(
                        self.gemini_analyzer.correct_entities_and_abbreviations,
                        transcription_result["text"]
                    )
                    
                    accent_future = executor.submit(
                        self.gemini_analyzer.detect_accent,
                        transcription_result["text"]
                    )
                    
                    # İkinci aşama sonuçlarını al
                    gemini_analysis = gemini_quality_future.result()
                    corrected_text = correction_future.result()
                    heavy_accent_flag = accent_future.result()
                    
                    # Sonuçları ata
                    results["gemini_analysis"] = gemini_analysis
                    transcription_result["text"] = corrected_text
                    results["corrected_transcription"] = corrected_text
                    results["heavy_accent_analysis"] = heavy_accent_flag
                    
                if heavy_accent_flag.get("heavy_accent"):
                    self.log_info("Ağır aksan bayrağı aktif edildi.")
                    
                    self.log_info("Paralel kalite analizi ve düzeltme işlemleri tamamlandı.")
                
                progress.update(task, description="Kalite analizi tamamlandı", advance=1)
            
                # Son transkripti oluştur
                final_transcription_text = self._generate_final_transcription_text(results)
                results["final_transcription"] = final_transcription_text
                
                # NLP analizi
                progress.update(task, description="NLP analizi yapılıyor...", advance=1)
                if final_transcription_text:
                    # Paralel NLP analizi
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        # Ana NLP analizi
                        nlp_future = executor.submit(
                            self.turkish_analyzer.analyze_text, final_transcription_text
                        )
                        
                        # Türkçe pattern analizi
                        turkish_future = None
                        if hasattr(self.turkish_analyzer, 'detect_turkish_patterns'):
                            turkish_future = executor.submit(
                                self.turkish_analyzer.detect_turkish_patterns, final_transcription_text
                            )
                        
                        # Sonuçları al
                        nlp_results = nlp_future.result()
                        
                        if turkish_future:
                            turkish_patterns = turkish_future.result()
                        nlp_results["turkish_patterns"] = turkish_patterns
                        
                        # Türkçe pattern'lara göre kalite skorları güncelle
                        quality_indicators = turkish_patterns.get("quality_indicators", {})
                        if quality_indicators:
                            self.log_info(f"Türkçe konuşma kalitesi: Formal={quality_indicators.get('is_formal_speech', False)}, "
                                        f"Tereddüt={quality_indicators.get('has_hesitation', False)}, "
                                        f"Ağız={quality_indicators.get('has_dialect', False)}")
                            
                            # Türkçe pattern'lara göre ek etiket önerileri
                            turkish_label_suggestions = self._generate_turkish_label_suggestions(turkish_patterns)
                            if turkish_label_suggestions:
                                existing_suggestions = results.get("label_suggestions", {}).get("suggestions", [])
                                existing_suggestions.extend(turkish_label_suggestions)
                                self.log_info(f"Türkçe pattern'larına göre {len(turkish_label_suggestions)} ek etiket önerisi eklendi")
                    
                    results["nlp_analysis"] = nlp_results
                else:
                    results["nlp_analysis"] = {}

                progress.update(task, description="İşlem tamamlandı", advance=1)

                # Bellek ve dosya temizliği
                self._cleanup_temp_files(processed_audio_path)
                self._optimize_memory_usage()

            except Exception as e:
                self.log_error(f"İşlem sırasında hata: {e}", exc_info=True)
                results["error"] = str(e)
            
        return results

    def _display_audio_info(self, audio_info: Dict):
        """Ses dosyası bilgilerini konsola yazdır"""
        try:
            table = Table(title="🎵 Ses Dosyası Bilgileri", style="cyan")
            table.add_column("Özellik", style="bold blue")
            table.add_column("Değer", style="green")
            
            # Temel bilgiler
            table.add_row("📁 Dosya", audio_info.get("dosya_adi", "Bilinmiyor"))
            table.add_row("⏱️ Süre", f"{audio_info.get('sure_saniye', 0):.1f} saniye")
            table.add_row("🔊 Örnekleme Hızı", f"{audio_info.get('ornek_hizi', 0)} Hz")
            table.add_row("📻 Kanal Sayısı", str(audio_info.get("kanal_sayisi", 0)))
            table.add_row("💾 Dosya Boyutu", f"{audio_info.get('dosya_boyutu_mb', 0):.1f} MB")
            
            # Ses kalitesi göstergeleri
            if "ortalama_dB" in audio_info:
                table.add_row("📈 Ortalama dB", f"{audio_info['ortalama_dB']:.1f} dB")
            if "maksimum_dB" in audio_info:
                table.add_row("📊 Maksimum dB", f"{audio_info['maksimum_dB']:.1f} dB")
                
            self.console.print(table)
            
        except Exception as e:
            self.log_error(f"Audio info display hatası: {e}")
    
    def _optimize_memory_usage(self):
        """Bellek kullanımını optimize et"""
        try:
            # Garbage collection yap
            collected = gc.collect()
            
            # Bellek kullanım bilgisi
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.log_info(f"Bellek optimizasyonu: {collected} nesne temizlendi, mevcut kullanım: {memory_mb:.1f} MB")
            
            # 1GB'dan fazla kullanımda uyarı ver
            if memory_mb > 1024:
                self.log_warning(f"Yüksek bellek kullanımı tespit edildi: {memory_mb:.1f} MB")
                
        except Exception as e:
            self.log_error(f"Bellek optimizasyon hatası: {e}")
    
    def _cleanup_temp_files(self, processed_audio_path: str = None):
        """Geçici dosyaları temizle"""
        try:
            if processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
                self.log_info(f"Geçici ses dosyası temizlendi: {processed_audio_path}")
                
            # temp/ klasöründeki eski dosyaları temizle
            temp_dir = Path("temp")
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*.wav"):
                    file_age = time.time() - temp_file.stat().st_mtime
                    if file_age > 3600:  # 1 saatten eski
                        temp_file.unlink()
                        self.log_info(f"Eski geçici dosya temizlendi: {temp_file}")
                        
        except Exception as e:
            self.log_error(f"Dosya temizleme hatası: {e}")
    
    def _display_quality_assessment(self, quality_result: Dict):
        """Kalite değerlendirmesi sonuçlarını konsola yazdır"""
        try:
            table = Table(title="🔍 Ses Kalitesi Analizi", style="yellow")
            table.add_column("Metrik", style="bold blue")
            table.add_column("Değer", style="green")
            table.add_column("Durum", style="bold")
            
            # Temel kalite metrikleri
            snr = quality_result.get("snr_db", 0)
            snr_status = "✅ İyi" if snr > 10 else "⚠️ Orta" if snr > 5 else "❌ Düşük"
            table.add_row("📈 Sinyal/Gürültü Oranı", f"{snr:.1f} dB", snr_status)
            
            clarity = quality_result.get("clarity_score", 0)
            clarity_status = "✅ Net" if clarity > 0.7 else "⚠️ Orta" if clarity > 0.4 else "❌ Belirsiz"
            table.add_row("🔊 Netlik Skoru", f"{clarity:.2f}", clarity_status)
            
            # Frekans analizi
            freq_balance = quality_result.get("frequency_balance", {})
            if freq_balance:
                for freq_name, score in freq_balance.items():
                    freq_status = "✅" if score > 0.6 else "⚠️" if score > 0.3 else "❌"
                    table.add_row(f"📊 {freq_name.title()}", f"{score:.2f}", freq_status)
            
            # Categorical assessment sonuçları
            categorical = quality_result.get("categorical_assessment", {})
            if categorical:
                table.add_row("", "", "")  # Boş satır
                table.add_row("🏷️ [bold]Kategori Değerlendirmesi[/bold]", "", "")
                
                categories = {
                    "belirsiz_ses": "🎯 Ses Netliği",
                    "agir_aksan": "🗣️ Aksan Durumu", 
                    "yanlis_dil": "🌐 Dil Uyumu",
                    "sentezlenmis": "🤖 Yapay Ses",
                    "coklu_ses": "👥 Çoklu Konuşmacı"
                }
                
                for key, display_name in categories.items():
                    value = categorical.get(key, False)
                    status = "❌ Sorunlu" if value else "✅ Normal"
                    table.add_row(display_name, "Evet" if value else "Hayır", status)
            
            self.console.print(table)
            
            # Önemli uyarılar
            warnings = []
            if snr < 5:
                warnings.append("⚠️ Düşük sinyal/gürültü oranı - transkripsiyon kalitesi etkilenebilir")
            if clarity < 0.4:
                warnings.append("⚠️ Düşük ses netliği - manuel düzeltme gerekebilir")
            if categorical.get("coklu_ses", False):
                warnings.append("👥 Çoklu konuşmacı tespit edildi - konuşmacı ayırma aktif")
                
            if warnings:
                self.console.print("\n[bold yellow]🚨 Önemli Uyarılar:[/bold yellow]")
                for warning in warnings:
                    self.console.print(f"  {warning}")
                
        except Exception as e:
            self.log_error(f"Quality assessment display hatası: {e}")

    def _display_speaker_info(self, speaker_result: Dict, main_speaker: str):
        """Konuşmacı bilgilerini konsola yazdır"""
        try:
            table = Table(title="👥 Konuşmacı Analizi", style="magenta")
            table.add_column("Konuşmacı", style="bold blue")
            table.add_column("Konuşma Süresi", style="green")
            table.add_column("Yüzde", style="yellow")
            table.add_column("Segment Sayısı", style="cyan")
            
            speakers = speaker_result.get("speakers", [])
            segments = speaker_result.get("segments", [])
            
            # Her konuşmacı için istatistikler
            speaker_stats = {}
            total_duration = 0
            
            for segment in segments:
                speaker = segment.get("speaker", "UNKNOWN")
                duration = segment.get("duration", 0)
                
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {"duration": 0, "segments": 0}
                
                speaker_stats[speaker]["duration"] += duration
                speaker_stats[speaker]["segments"] += 1
                total_duration += duration
            
            # Tabloyu doldur
            for speaker in sorted(speaker_stats.keys()):
                stats = speaker_stats[speaker]
                duration = stats["duration"]
                segments_count = stats["segments"]
                percentage = (duration / total_duration * 100) if total_duration > 0 else 0
                
                # Ana konuşmacıyı vurgula
                speaker_name = f"👑 {speaker}" if speaker == main_speaker else f"  {speaker}"
                
                table.add_row(
                    speaker_name,
                    f"{duration:.1f}s",
                    f"{percentage:.1f}%",
                    str(segments_count)
                )
            
            self.console.print(table)
            
            # Özet bilgiler
            summary_text = f"""
📊 [bold]Özet:[/bold]
• Toplam konuşmacı sayısı: {len(speakers)}
• Toplam segment sayısı: {len(segments)}
• Toplam süre: {total_duration:.1f} saniye
• Ana konuşmacı: {main_speaker}
            """
            
            self.console.print(Panel.fit(summary_text.strip(), style="blue"))
        except Exception as e:
            self.log_error(f"Speaker info display hatası: {e}")

    # ------------------------------------------------------------------
    # Ek Yardımcı Fonksiyonlar (basit/özet implementasyonlar)
    # ------------------------------------------------------------------

    def _apply_foreign_language_rule(self, transcription_result: Dict):
        """Yabancı dil kontrolü – basit stub (detaylı analiz kaldırıldı)"""
        try:
            text = transcription_result.get("text", "")
            if not text:
                return
            from langdetect import detect, LangDetectException
            try:
                detected_language = detect(text)
                transcription_result["detected_language"] = detected_language
                transcription_result["language_warning"] = detected_language != "tr"
            except LangDetectException:
                transcription_result["detected_language"] = "unknown"
                transcription_result["language_warning"] = True
        except Exception as e:
            self.log_error(f"Foreign language rule hatası: {e}")

    def _generate_final_transcription_text(self, results: Dict) -> str:
        """Son transkripti (düz metin) döndürür."""
        return self._get_final_transcription(results)

    def _get_final_transcription(self, results: Dict) -> str:
        transcription_result = results.get("transcription", {})
        return transcription_result.get("text", "")

    def _get_user_choice(self, question: str, choices: List[str]) -> str:
        """Kullanıcı etkileşimi olmayan ortamlar için ilk seçeneği döndürür."""
        return choices[0] if choices else ""

    def save_results(self, results: Dict[str, Any], output_dir: str = "output"):
        """Yalnızca .txt ve .json çıktı oluşturur (SRT/VTT kaldırıldı)."""
        import json
        os.makedirs(output_dir, exist_ok=True)
        final_text = results.get("final_transcription", "")
        timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

        # ---- Dinamik Anket Oluştur ----
        def checkbox(checked: bool, text: str) -> str:
            return f"[{'x' if checked else ' '}] {text}"

        aq = results.get("audio_quality", {})
        clarity = aq.get("clarity_score", 1.0)
        snr = aq.get("snr_db", 20)
        unclear_audio = clarity < 0.4 or snr < 5

        wrong_lang = False
        wl = results.get("wrong_language_analysis", {})
        if isinstance(wl, dict):
            wrong_lang = wl.get("wrong_language", False)
        else:
            # Ayrıca foreign language rule
            tr = results.get("transcription", {})
            wrong_lang = tr.get("language_warning", False)

        multiple_voices = False
        spk = results.get("speaker_diarization", {})
        if isinstance(spk, dict):
            multiple_voices = spk.get("speaker_count", 1) > 1

        questionnaire_lines = [
            "\n\nPROBLEM WITH AUDIO QUESTIONS",
            "Select ALL that apply.\n",
            checkbox(unclear_audio, "User was hard to hear because of unclear audio"),
            "",
            checkbox(False, "User has heavy accent"),
            "",
            checkbox(wrong_lang, "Request is in the incorrect language for this locale"),
            "",
            checkbox(False, "Request is synthesized or recorded speech"),
            "",
            checkbox(multiple_voices, "Multiple voices in the audio"),
            ""
        ]

        questionnaire = "\n".join(questionnaire_lines)

        # Metni ve anketi birleştir
        txt_content = final_text.rstrip() + questionnaire

        # TXT
        txt_path = Path(output_dir) / f"{timestamp}.txt"
        with open(txt_path, "w", encoding="utf-8") as fp:
            fp.write(txt_content)

        # JSON (tüm sonuçlar)
        json_path = Path(output_dir) / f"{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=4, default=str)

        self.log_info(f"Sonuçlar kaydedildi (.txt & .json): {txt_path}, {json_path}")
        return str(json_path)

    # Aşağıdaki fonksiyonların detaylı implementasyonları kaldırıldı; temel işlevsel stub'lar eklendi

    def _interactive_main_speaker_selection(self, speaker_result: Dict) -> str:
        return speaker_result.get("main_speaker", speaker_result.get("speakers", ["SPEAKER_00"])[0])

    def format_for_editing(self, transcription_result: Dict) -> str:
        return transcription_result.get("text", "")

    def interactive_edit(self, initial_text: str) -> str:
        return initial_text

    def _detect_truncation(self, audio_data: np.ndarray, sample_rate: int, threshold: float = 0.1, duration_ms: int = 50) -> Dict:
        return {"start": False, "end": False, "is_truncated": False, "reason": "stub"}

    def _generate_turkish_label_suggestions(self, turkish_patterns: Dict) -> List[Dict]:
        return []

    def _display_correction_suggestions(self, llm_analysis: Dict, label_suggestions: Dict):
        pass

    def _generate_editing_instructions(self, llm_analysis: Dict, label_suggestions: Dict) -> str:
        return ""

    def _display_final_results(self, results: Dict):
        pass

    def _ask_user(self, question: str) -> bool:
        return True

    def _check_wrong_language_polly_rule(self, text: str) -> Dict:
        return {"wrong_language": False}

    def _save_intermediate_results(self, results: Dict, output_dir: str = "output"):
        pass

    # ---- SRT / VTT Fonksiyonları Artık Kullanılmıyor ----
    def _export_srt(self, results: Dict, srt_file: Path):
        pass

    def _export_vtt(self, results: Dict, vtt_file: Path):
        pass

    def _export_speaker_labeled(self, results: Dict, labeled_file: Path):
        pass

    def _seconds_to_srt_time(self, seconds: float) -> str:
        millis = int((seconds % 1) * 1000)
        secs = int(seconds) % 60
        minutes = int(seconds // 60) % 60
        hours = int(seconds // 3600)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _seconds_to_vtt_time(self, seconds: float) -> str:
        millis = int((seconds % 1) * 1000)
        secs = int(seconds) % 60
        minutes = int(seconds // 60) % 60
        hours = int(seconds // 3600)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _seconds_to_timestamp(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"

    def _safe_speaker_diarization(self, processed_audio_path: str) -> Dict:
        return {"segments": [], "speakers": [], "speaker_count": 0, "main_speaker": "SPEAKER_00"}