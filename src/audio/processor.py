"""
Ses işleme modülü - pydub, librosa ve ffmpeg entegrasyonu
"""
import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from pydub import AudioSegment
from pydub.playback import play
import librosa
import soundfile as sf
from ..utils.logger import LoggerMixin
from ..utils.config import Config


class AudioProcessor(LoggerMixin):
    """Ses dosyalarını işleme sınıfı"""
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.get("audio_processing.sample_rate", 16000)
        self.chunk_length = config.get("audio_processing.chunk_length", 30)
        self.overlap = config.get("audio_processing.overlap", 5)
        self.supported_formats = config.get(
            "audio_processing.supported_formats", 
            ["wav", "mp3", "flac", "m4a", "ogg"]
        )
        
        self.log_info("AudioProcessor başlatıldı")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Ses dosyasını yükle
        
        Args:
            file_path: Ses dosyası yolu
            
        Returns:
            (audio_data, sample_rate) tuple
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Ses dosyası bulunamadı: {file_path}")
        
        if file_path.suffix.lower()[1:] not in self.supported_formats:
            raise ValueError(f"Desteklenmeyen ses formatı: {file_path.suffix}")
        
        self.log_info(f"Ses dosyası yükleniyor: {file_path}")
        
        try:
            # librosa ile yükle
            audio_data, sr = librosa.load(
                str(file_path), 
                sr=self.sample_rate,
                mono=True
            )
            
            self.log_info(f"Ses dosyası başarıyla yüklendi. Süre: {len(audio_data)/sr:.2f} saniye")
            return audio_data, sr
            
        except Exception as e:
            self.log_error(f"Ses dosyası yüklenirken hata: {e}")
            
            # pydub ile alternatif yükleme dene
            try:
                audio_segment = AudioSegment.from_file(str(file_path))
                audio_data = np.array(audio_segment.get_array_of_samples())
                
                if audio_segment.channels == 2:
                    audio_data = audio_data.reshape((-1, 2)).mean(axis=1)
                
                # Normalize et
                audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
                
                # Sample rate'i dönüştür
                if audio_segment.frame_rate != self.sample_rate:
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=audio_segment.frame_rate,
                        target_sr=self.sample_rate
                    )
                
                self.log_info("pydub ile ses dosyası yüklendi")
                return audio_data, self.sample_rate
                
            except Exception as e2:
                self.log_error(f"pydub ile de yüklenemedi: {e2}")
                raise
    
    def chunk_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[np.ndarray, float, float]]:
        """
        Ses dosyasını parçalara böl
        
        Args:
            audio_data: Ses verisi
            sample_rate: Örnekleme hızı
            
        Returns:
            [(chunk_data, start_time, end_time), ...] listesi
        """
        total_duration = len(audio_data) / sample_rate
        chunk_samples = int(self.chunk_length * sample_rate)
        overlap_samples = int(self.overlap * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        start_sample = 0
        
        while start_sample < len(audio_data):
            end_sample = min(start_sample + chunk_samples, len(audio_data))
            
            chunk_data = audio_data[start_sample:end_sample]
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            # Çok kısa parçaları atla
            if len(chunk_data) > sample_rate * 0.5:  # En az 0.5 saniye
                chunks.append((chunk_data, start_time, end_time))
            
            start_sample += step_samples
            
            if end_sample >= len(audio_data):
                break
        
        self.log_info(f"Ses {len(chunks)} parçaya bölündü")
        return chunks
    
    def save_temp_audio(self, audio_data: np.ndarray, sample_rate: int, format: str = "wav") -> str:
        """
        Geçici ses dosyası kaydet
        
        Args:
            audio_data: Ses verisi
            sample_rate: Örnekleme hızı
            format: Dosya formatı
            
        Returns:
            Geçici dosya yolu
        """
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            self.log_debug(f"Geçici ses dosyası oluşturuldu: {tmp_file.name}")
            return tmp_file.name
    
    def play_audio(self, file_path: str):
        """
        Terminal üzerinden ses dosyasını oynat
        
        Args:
            file_path: Ses dosyası yolu
        """
        try:
            # ffplay ile oynat
            subprocess.run([
                "ffplay", 
                "-nodisp", 
                "-autoexit", 
                "-loglevel", "quiet",
                str(file_path)
            ], check=True)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # pydub ile alternatif oynatma
                audio = AudioSegment.from_file(str(file_path))
                play(audio)
                
            except Exception as e:
                self.log_warning(f"Ses oynatılamadı: {e}")
                print(f"Ses dosyasını manuel olarak dinleyebilirsiniz: {file_path}")
    
    def normalize_audio(self, audio_data: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
        """
        Ses seviyesini normalize et
        
        Args:
            audio_data: Ses verisi
            target_lufs: Hedef LUFS değeri
            
        Returns:
            Normalize edilmiş ses verisi
        """
        # RMS hesapla
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms > 0:
            # Basit normalizasyon (gerçek LUFS hesaplama daha karmaşık)
            target_rms = 10**(target_lufs/20)
            scaling_factor = target_rms / rms
            audio_data = audio_data * scaling_factor
            
            # Clipping'i önle
            audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data
    
    def get_audio_info(self, file_path: str) -> dict:
        """
        Ses dosyası bilgilerini al
        
        Args:
            file_path: Ses dosyası yolu
            
        Returns:
            Ses dosyası bilgileri
        """
        try:
            audio_segment = AudioSegment.from_file(str(file_path))
            audio_data, sr = self.load_audio(file_path)
            
            # Temel istatistikler
            duration = len(audio_segment) / 1000.0  # saniye
            channels = audio_segment.channels
            sample_rate = audio_segment.frame_rate
            bit_depth = audio_segment.sample_width * 8
            
            # Ses analizi
            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.max(np.abs(audio_data))
            
            # SNR tahmini (basit)
            noise_estimate = np.std(audio_data[:int(0.1 * sr)])  # İlk 100ms'den noise tahmini
            signal_estimate = rms
            snr_db = 20 * np.log10(signal_estimate / (noise_estimate + 1e-10))
            
            info = {
                "dosya_yolu": str(file_path),
                "dosya_boyutu_mb": Path(file_path).stat().st_size / (1024*1024),
                "sure_saniye": duration,
                "sure_formatli": f"{int(duration//60):02d}:{int(duration%60):02d}",
                "kanal_sayisi": channels,
                "ornekleme_hizi": sample_rate,
                "bit_derinligi": bit_depth,
                "rms_seviye": float(rms),
                "peak_seviye": float(peak),
                "tahmini_snr_db": float(snr_db),
                "format": Path(file_path).suffix.lower()[1:]
            }
            
            return info
            
        except Exception as e:
            self.log_error(f"Ses dosyası bilgileri alınamadı: {e}")
            return {}
    
    def convert_format(self, input_path: str, output_path: str, format: str = "wav") -> bool:
        """
        Ses dosyasını farklı formata dönüştür
        
        Args:
            input_path: Girdi dosyası yolu
            output_path: Çıktı dosyası yolu
            format: Hedef format
            
        Returns:
            Başarı durumu
        """
        try:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format=format)
            self.log_info(f"Ses dosyası dönüştürüldü: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.log_error(f"Format dönüştürme hatası: {e}")
            return False
    
    def cleanup_temp_files(self, temp_files: List[str]):
        """
        Geçici dosyaları temizle
        
        Args:
            temp_files: Silinecek geçici dosya yolları
        """
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    self.log_debug(f"Geçici dosya silindi: {temp_file}")
            except Exception as e:
                self.log_warning(f"Geçici dosya silinemedi {temp_file}: {e}")
    
    def reduce_noise(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Gürültü azaltma adımını uygula (stub)."""
        method = self.config.get("audio_processing.noise_reduction.method", "spectral_subtraction")
        if method == "spectral_subtraction":
            self.log_info("Spektral çıkarma gürültü azaltma (stub) uygulandı.")
        return audio_data 