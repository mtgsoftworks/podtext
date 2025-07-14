"""
Konfigürasyon yönetimi modülü
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Proje konfigürasyon yöneticisi"""
    
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        if not os.path.exists(config_file):
            print(f"Uyarı: {config_file} bulunamadı. Varsayılan konfigürasyon oluşturuluyor.")
            self._create_default_config()
            self.save()
        else:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
        
        # .env dosyasını yükle
        load_dotenv()
        
        # Konfigürasyonu yükle
        self._load_config()
    
    def _load_config(self):
        """YAML konfigürasyon dosyasını yükle"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as file:
                self.config_data = yaml.safe_load(file)
            
            # Environment değişkenlerini yerine koy
            self._substitute_env_vars(self.config_data)
            
        except FileNotFoundError:
            print(f"Uyarı: {self.config_file} dosyası bulunamadı. Varsayılan ayarlar kullanılacak.")
            self._create_default_config()
        except yaml.YAMLError as e:
            print(f"YAML hatası: {e}")
            self._create_default_config()
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """Environment değişkenlerini yerine koy"""
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self._substitute_env_vars(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = self._substitute_env_vars(item)
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        
        return data
    
    def _create_default_config(self):
        """Varsayılan konfigürasyon oluştur"""
        self.config_data = {
            "whisper": {
                "model": "large-v3",
                "language": "tr",
                "threads": 8,
                # pywhispercpp model init parametreleri
                "pywhispercpp_params": {
                    "no_context": True,
                    "n_threads": 6
                },
                # pywhispercpp transcribe parametreleri
                "pywhispercpp_transcribe_params": {
                    "language": "tr",
                    "translate": False,
                    "n_processors": 8
                },
                "temperature": 0.0,
                "beam_size": 5,
                "vad_threshold": 0.5,
                "entropy_threshold": 2.8,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "word_timestamps": True
            },
            "diarization": {
                "model_name": "pyannote/speaker-diarization-3.1",
                "min_speakers": 1,
                "max_speakers": 10
            },
            "audio_quality": {
                "categories": [
                    "belirsiz_ses",
                    "agir_aksan",
                    "yanlis_dil",
                    "sentezlenmis_konusma",
                    "coklu_ses"
                ]
            },
            "gemini": {
                "model": "gemini-1.5-flash"
            },
            "audio_processing": {
                "noise_reduction": { "enabled": True },
                "normalization": { "enabled": True }
            },
            "auto_labeling": {
                "unsure_confidence_threshold": 0.7,
                "truncation_silence_threshold_ms": 2000,
                "stutter_word_count": 3
            },
            "nlp": {
                "model_path": "tr_core_news_lg"
            },
            "enable_model_comparison": False,
            "whisper_openai": {
                "api_key": None,  # OpenAI API anahtarı
                "model": "whisper-1",
                "language": "tr",
                "response_format": "verbose_json",
                "temperature": 0.0
            },
            "google_cloud": {
                "credentials_path": None,  # Service account JSON dosyası yolu
                "project_id": None,
                "language_code": "tr-TR",
                "model": "latest_long",
                "enable_automatic_punctuation": True,
                "enable_word_time_offsets": True
            },
            "performance": {
                "max_parallel_workers": 4,
                "enable_gpu": False,
                "memory_limit_gb": 8,
                "cpu_usage_limit": 0.9,  # CPU kullanımını %90'la sınırla
                "auto_adjust_threads": True,  # CPU kullanımına göre otomatik ayarlama
                "timeout_seconds": 300
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Konfigürasyon değerini anahtarla al"""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_key(self, service: str) -> Optional[str]:
        """API anahtarı al. Öncelikle env, sonrasında config.yaml'daki api_keys alanına bakar."""
        # Öncelik: environment variable
        service_env_key = f"{service.upper()}_API_KEY" if service not in ["google_gemini", "huggingface"] else None
        if service == "huggingface":
            env_val = os.getenv("HUGGINGFACE_TOKEN")
        elif service == "google_gemini":
            env_val = os.getenv("GOOGLE_GEMINI_API_KEY")
        else:
            env_val = os.getenv(service_env_key) if service_env_key else None
        if env_val:
            return env_val
        # İkinci seçenek: config.yaml içindeki api_keys
        api_keys_cfg = self.get("api_keys", {})
        return api_keys_cfg.get(service)
    
    def get_whisper_config(self) -> Dict[str, Any]:
        """Whisper konfigürasyonunu al"""
        return self.get("whisper", {})
    
    def get_speaker_config(self) -> Dict[str, Any]:
        """Konuşmacı tanıma konfigürasyonunu al"""
        return self.get("diarization", {})
    
    def get_quality_categories(self) -> list:
        """Ses kalitesi kategorilerini al"""
        return self.get("audio_quality.categories", [])
    
    def get_labeling_tags(self) -> Dict[str, str]:
        """Etiketleme tag'lerini al"""
        return self.get("labeling.tags", {})
    
    def is_debug(self) -> bool:
        """Debug modu aktif mi?"""
        return os.getenv("DEBUG", "false").lower() == "true"
    
    def get_editor(self) -> str:
        """Varsayılan editörü al"""
        return os.getenv("EDITOR", "nano")
    
    def create_dirs(self):
        """Gerekli dizinleri oluştur"""
        dirs = [
            "logs",
            "cache", 
            "temp",
            "output",
            "models"
        ]
        
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True) 