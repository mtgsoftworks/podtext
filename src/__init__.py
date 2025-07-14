"""
Podcast Transkripsiyon Sistemi

Modern yapay zeka teknolojileri ve Türkçe özel modeller kullanarak 
podcast ses kayıtlarının transkripsiyon işlemini gerçekleştiren kapsamlı bir araç.

Özellikler:
- OpenAI Whisper transkripsiyon
- SpaCy tr_core_news_lg tabanlı Türkçe NLP analizi
- Konuşmacı tanıma ve ayırma
- AI destekli kalite değerlendirmesi
"""

__version__ = "1.1.0"
__author__ = "Podcast Transkripsiyon Sistemi"
__email__ = "info@podcasttranscription.com"

from .core.orchestrator import PodcastTranscriptionOrchestrator
from .models.whisper_local import LocalWhisperModel as WhisperTranscriber
from .models.speaker_diarization import SpeakerDiarizer
from .models.gemini_model import GeminiAnalyzer
from .audio.processor import AudioProcessor
from .quality.evaluator import AudioQualityEvaluator
from .labeling.auto_labeler import AutoLabeler
from .nlp.turkish_analyzer import TurkishNLPAnalyzer
from .utils.config import Config

__all__ = [
    "PodcastTranscriptionOrchestrator",
    "WhisperTranscriber",
    "SpeakerDiarizer", 
    "GeminiAnalyzer",
    "AudioProcessor",
    "AudioQualityEvaluator",
    "AutoLabeler",
    "TurkishNLPAnalyzer",
    "Config"
] 