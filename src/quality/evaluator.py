"""
Ses kalitesi değerlendirme modülü - PESQ, STOI ve kategorik kalite analizi
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torchaudio
from scipy import signal
import librosa
# Ses kalitesi metrikleri (Windows'ta PESQ sorunlu olabilir)
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    pesq = None

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    stoi = None
from ..utils.logger import LoggerMixin
from ..utils.config import Config


class AudioQualityEvaluator(LoggerMixin):
    """Ses kalitesi değerlendirme sınıfı"""
    
    def __init__(self, config: Config):
        self.config = config
        self.quality_config = config.get("audio_quality", {})
        self.categories = config.get_quality_categories()
        self.thresholds = self.quality_config.get("thresholds", {})
        
        # Kalite eşikleri
        self.pesq_min = self.thresholds.get("pesq_min", 1.0)
        self.stoi_min = self.thresholds.get("stoi_min", 0.3)
        self.snr_min = self.thresholds.get("snr_min", 5.0)
        
        self.log_info("AudioQualityEvaluator başlatıldı")
    
    def evaluate_audio_quality(self, audio_file: str, reference_file: str = None) -> Dict:
        """
        Kapsamlı ses kalitesi değerlendirmesi
        
        Args:
            audio_file: Değerlendirilecek ses dosyası
            reference_file: Referans ses dosyası (PESQ için gerekli)
            
        Returns:
            Kalite değerlendirme sonuçları
        """
        try:
            self.log_info(f"Ses kalitesi değerlendirmesi başlatılıyor: {audio_file}")
            
            # Ses dosyasını yükle
            audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)
            
            # Temel kalite metrikleri
            basic_metrics = self._calculate_basic_metrics(audio_data, sample_rate)
            
            # Objektif kalite metrikleri (PESQ, STOI)
            objective_metrics = {}
            if reference_file:
                objective_metrics = self._calculate_objective_metrics(
                    audio_file, reference_file, audio_data, sample_rate
                )
            
            # Kategorik kalite değerlendirmesi
            categorical_assessment = self._assess_categorical_quality(
                audio_data, sample_rate
            )
            
            # Genel kalite skoru
            overall_score = self._calculate_overall_quality_score(
                basic_metrics, objective_metrics, categorical_assessment
            )
            
            result = {
                "basic_metrics": basic_metrics,
                "objective_metrics": objective_metrics,
                "categorical_assessment": categorical_assessment,
                "overall_score": overall_score,
                "recommendations": self._generate_recommendations(
                    basic_metrics, categorical_assessment
                )
            }
            
            self.log_info("Ses kalitesi değerlendirmesi tamamlandı")
            return result
            
        except Exception as e:
            self.log_error(f"Ses kalitesi değerlendirme hatası: {e}")
            return {
                "error": str(e),
                "basic_metrics": {},
                "objective_metrics": {},
                "categorical_assessment": {},
                "overall_score": 0.0
            }
    
    def _calculate_basic_metrics(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Temel ses kalitesi metriklerini hesapla"""
        
        # SNR tahmini
        snr_db = self._estimate_snr(audio_data, sample_rate)
        
        # RMS seviyesi
        rms_level = np.sqrt(np.mean(audio_data**2))
        rms_db = 20 * np.log10(rms_level + 1e-10)
        
        # Peak seviye
        peak_level = np.max(np.abs(audio_data))
        peak_db = 20 * np.log10(peak_level + 1e-10)
        
        # Crest factor
        crest_factor = peak_level / (rms_level + 1e-10)
        crest_factor_db = 20 * np.log10(crest_factor)
        
        # Sessizlik oranı
        silence_threshold = 0.01
        silence_ratio = np.sum(np.abs(audio_data) < silence_threshold) / len(audio_data)
        
        # Clipping tespiti
        clipping_ratio = np.sum(np.abs(audio_data) > 0.99) / len(audio_data)
        
        # Spektral özellikler
        spectral_features = self._calculate_spectral_features(audio_data, sample_rate)
        
        return {
            "snr_db": float(snr_db),
            "rms_level_db": float(rms_db),
            "peak_level_db": float(peak_db),
            "crest_factor_db": float(crest_factor_db),
            "silence_ratio": float(silence_ratio),
            "clipping_ratio": float(clipping_ratio),
            "dynamic_range_db": float(peak_db - rms_db),
            **spectral_features
        }
    
    def _estimate_snr(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """SNR tahmini (basit yöntem)"""
        
        # İlk ve son %10'luk kısımlardan noise tahmini
        noise_start = audio_data[:int(0.1 * len(audio_data))]
        noise_end = audio_data[-int(0.1 * len(audio_data)):]
        noise_samples = np.concatenate([noise_start, noise_end])
        
        noise_power = np.mean(noise_samples**2)
        signal_power = np.mean(audio_data**2)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
        else:
            snr_db = 60.0  # Varsayılan yüksek SNR
        
        return snr_db
    
    def _calculate_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Spektral özellikler hesapla"""
        
        # FFT hesapla
        fft = np.fft.fft(audio_data)
        magnitude_spectrum = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(fft)//2]
        
        # Spektral centroid
        spectral_centroid = np.sum(freqs * magnitude_spectrum) / (np.sum(magnitude_spectrum) + 1e-10)
        
        # Spektral bandwidth
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid)**2) * magnitude_spectrum) / 
            (np.sum(magnitude_spectrum) + 1e-10)
        )
        
        # Spektral rolloff
        cumsum_spectrum = np.cumsum(magnitude_spectrum)
        rolloff_threshold = 0.85 * cumsum_spectrum[-1]
        rolloff_idx = np.where(cumsum_spectrum >= rolloff_threshold)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
        
        return {
            "spectral_centroid_hz": float(spectral_centroid),
            "spectral_bandwidth_hz": float(spectral_bandwidth),
            "spectral_rolloff_hz": float(spectral_rolloff),
            "zero_crossing_rate": float(zcr)
        }
    
    def _calculate_objective_metrics(self, audio_file: str, reference_file: str,
                                   audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Objektif kalite metrikleri (PESQ, STOI)"""
        
        try:
            # Referans dosyasını yükle
            ref_data, ref_sr = librosa.load(reference_file, sr=sample_rate, mono=True)
            
            # Aynı uzunluğa getir
            min_length = min(len(audio_data), len(ref_data))
            audio_data = audio_data[:min_length]
            ref_data = ref_data[:min_length]
            
            # PESQ hesapla (8kHz veya 16kHz gerekli)
            pesq_score = None
            if PESQ_AVAILABLE and sample_rate in [8000, 16000]:
                try:
                    pesq_score = pesq(sample_rate, ref_data, audio_data, 'wb' if sample_rate == 16000 else 'nb')
                except Exception as e:
                    self.log_warning(f"PESQ hesaplama hatası: {e}")
            
            # STOI hesapla
            stoi_score = None
            if STOI_AVAILABLE:
                try:
                    stoi_score = stoi(ref_data, audio_data, sample_rate, extended=False)
                except Exception as e:
                    self.log_warning(f"STOI hesaplama hatası: {e}")
            
            return {
                "pesq_score": float(pesq_score) if pesq_score is not None else None,
                "stoi_score": float(stoi_score) if stoi_score is not None else None,
                "has_reference": True
            }
            
        except Exception as e:
            self.log_warning(f"Objektif metrik hesaplama hatası: {e}")
            return {
                "pesq_score": None,
                "stoi_score": None,
                "has_reference": False,
                "error": str(e)
            }
    
    def _assess_categorical_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Kategorik kalite değerlendirmesi"""
        
        # Eşikleri konfigürasyondan al
        cat_thresh = self.quality_config.get("categorical_thresholds", {})
        mv_var_thresh = cat_thresh.get("multi_voice_variance", 0.5)
        self.log_info(f"Multi-voice variance threshold: {mv_var_thresh}")
        assessment = {
            "belirsiz_ses": False,
            "agir_aksan": False,
            "yanlis_dil": False,
            "sentezlenmis": False,
            "coklu_ses": False,
            "confidence_scores": {}
        }
        
        # Belirsiz ses tespiti
        snr_db = self._estimate_snr(audio_data, sample_rate)
        # SNR loglaması
        self.log_info(f"Hesaplanan SNR: {snr_db:.1f} dB (eşik: {self.snr_min} dB)")
        if snr_db < self.snr_min:
            assessment["belirsiz_ses"] = True
            assessment["confidence_scores"]["belirsiz_ses"] = 1.0 - (snr_db / self.snr_min)
        
        # Sentezlenmiş konuşma tespiti (spektral özellikler)
        spectral_features = self._calculate_spectral_features(audio_data, sample_rate)
        
        # Çok düzenli spektral özellikler sentezlenmiş konuşmayı işaret edebilir
        zcr = spectral_features["zero_crossing_rate"]
        if zcr < 0.01 or zcr > 0.5:  # Çok düşük veya çok yüksek ZCR
            assessment["sentezlenmis"] = True
            assessment["confidence_scores"]["sentezlenmis"] = abs(zcr - 0.1) / 0.1
        
        # Çoklu ses tespiti (basit yöntem - enerji varyasyonu)
        frame_size = int(0.1 * sample_rate)  # 100ms frameler
        energy_frames = []
        
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i:i + frame_size]
            energy = np.sum(frame**2)
            energy_frames.append(energy)
        
        if len(energy_frames) > 0:
            energy_variance = np.var(energy_frames)
            energy_mean = np.mean(energy_frames)
            
            if energy_mean > 0:
                normalized_variance = energy_variance / (energy_mean + 1e-10)
                # Normalized variance loglaması
                self.log_info(f"Normalized variance: {normalized_variance:.2f} (eşik: {mv_var_thresh})")
                if normalized_variance > mv_var_thresh:
                    assessment["coklu_ses"] = True
                    assessment["confidence_scores"]["coklu_ses"] = min(normalized_variance / 2.0, 1.0)
        
        return assessment
    
    def _calculate_overall_quality_score(self, basic_metrics: Dict, 
                                       objective_metrics: Dict,
                                       categorical_assessment: Dict) -> float:
        """Genel kalite skoru hesapla (0-100)"""
        
        score = 100.0  # Başlangıç skoru
        
        # SNR bazlı düşüş
        snr_db = basic_metrics.get("snr_db", 0)
        if snr_db < 20:
            score -= (20 - snr_db) * 2  # Her dB başına 2 puan düşüş
        
        # Clipping cezası
        clipping_ratio = basic_metrics.get("clipping_ratio", 0)
        score -= clipping_ratio * 50  # %1 clipping = 0.5 puan düşüş
        
        # Sessizlik oranı cezası
        silence_ratio = basic_metrics.get("silence_ratio", 0)
        if silence_ratio > 0.5:  # %50'den fazla sessizlik
            score -= (silence_ratio - 0.5) * 40
        
        # Kategorik sorunlar cezası
        for category, detected in categorical_assessment.items():
            if isinstance(detected, bool) and detected:
                score -= 15  # Her kategori sorunu için 15 puan düşüş
        
        # Objektif metrikler bonusu
        if objective_metrics.get("pesq_score"):
            pesq_score = objective_metrics["pesq_score"]
            if pesq_score > 2.5:  # İyi PESQ skoru
                score += (pesq_score - 2.5) * 10
        
        if objective_metrics.get("stoi_score"):
            stoi_score = objective_metrics["stoi_score"]
            if stoi_score > 0.8:  # İyi STOI skoru
                score += (stoi_score - 0.8) * 25
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, basic_metrics: Dict, 
                                categorical_assessment: Dict) -> List[str]:
        """Kalite iyileştirme önerileri"""
        
        recommendations = []
        
        # SNR bazlı öneriler
        snr_db = basic_metrics.get("snr_db", 0)
        if snr_db < 10:
            recommendations.append("Ses kayıt ortamındaki arka plan gürültüsünü azaltın")
            recommendations.append("Mikrofonu konuşmacıya daha yakın konumlandırın")
        
        # Clipping uyarısı
        clipping_ratio = basic_metrics.get("clipping_ratio", 0)
        if clipping_ratio > 0.01:  # %1'den fazla clipping
            recommendations.append("Kayıt seviyesini düşürün, ses kırpılıyor")
        
        # Dinamik aralık önerisi
        dynamic_range = basic_metrics.get("dynamic_range_db", 0)
        if dynamic_range < 6:
            recommendations.append("Dinamik aralık çok düşük, ses işleme (sıkıştırma) azaltın")
        
        # Sessizlik uyarısı
        silence_ratio = basic_metrics.get("silence_ratio", 0)
        if silence_ratio > 0.7:
            recommendations.append("Çok fazla sessizlik var, kayıt ayarlarını kontrol edin")
        
        # Kategorik sorun önerileri
        if categorical_assessment.get("belirsiz_ses"):
            recommendations.append("Ses kalitesi düşük, kayıt ayarlarını iyileştirin")
        
        if categorical_assessment.get("coklu_ses"):
            recommendations.append("Birden fazla konuşmacı tespit edildi, konuşmacı ayırma kullanın")
        
        if categorical_assessment.get("sentezlenmis"):
            recommendations.append("Ses sentezlenmiş olabilir, doğal ses kayıt edin")
        
        return recommendations
    
    def classify_audio_quality(self, quality_score: float) -> str:
        """Kalite skorunu kategoriye dönüştür"""
        
        if quality_score >= 80:
            return "mükemmel"
        elif quality_score >= 65:
            return "iyi"
        elif quality_score >= 50:
            return "orta"
        elif quality_score >= 30:
            return "zayıf"
        else:
            return "çok_zayıf"
    
    def get_quality_summary(self, evaluation_result: Dict) -> Dict:
        """Kalite değerlendirmesi özeti"""
        
        basic_metrics = evaluation_result.get("basic_metrics", {})
        categorical = evaluation_result.get("categorical_assessment", {})
        overall_score = evaluation_result.get("overall_score", 0)
        
        # Tespit edilen sorunlar
        detected_issues = []
        for category, detected in categorical.items():
            if isinstance(detected, bool) and detected:
                issue_names = {
                    "belirsiz_ses": "Belirsiz/Gürültülü Ses",
                    "agir_aksan": "Ağır Aksan",
                    "yanlis_dil": "Yanlış Dil",
                    "sentezlenmis": "Sentezlenmiş Konuşma",
                    "coklu_ses": "Çoklu Konuşmacı"
                }
                detected_issues.append(issue_names.get(category, category))
        
        return {
            "genel_kalite": self.classify_audio_quality(overall_score),
            "kalite_skoru": round(overall_score, 1),
            "tespit_edilen_sorunlar": detected_issues,
            "snr_db": round(basic_metrics.get("snr_db", 0), 1),
            "clipping_var": basic_metrics.get("clipping_ratio", 0) > 0.01,
            "tavsiye_sayisi": len(evaluation_result.get("recommendations", []))
        } 