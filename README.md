# 🎙️ Podcast Transkripsiyon Sistemi

Modern yapay zeka teknolojileri kullanarak podcast ses kayıtlarının yüksek kaliteli transkripsiyon işlemini gerçekleştiren kapsamlı bir araçtır.

## ✨ Özellikler

### 🎯 Transkripsiyon ve Analiz
- **Yüksek Doğrulukta Transkripsiyon**: OpenAI Whisper Large v3 Turbo
- **Konuşmacı Tanıma ve Ayırma**: pyannote.audio ile gelişmiş speaker diarization
- **Ana Konuşmacı Filtreleme**: Sadece ana konuşmacının konuşması yazıya dökülür
- **AI Destekli Kalite Analizi**: Google Gemini 2.5 Pro ile akıllı analiz
- **Dilbilimsel Analiz**: spaCy ile Türkçe NLP işlemleri

### 🔍 Kalite Değerlendirme  
- **Ses Kalitesi Metrikleri**: PESQ, STOI objektif ölçümler
- **Kategorik Değerlendirme**: Belirsiz ses, ağır aksan, sentezlenmiş konuşma tespiti
- **SNR ve Clipping Analizi**: Teknik ses kalitesi kontrolü
- **Otomatik Öneri Sistemi**: Kalite iyileştirme tavsiyeleri

### 🔊 Ses Ön İşleme
- **Gürültü Azaltma**: Arka plan gürültüsü azaltma (spectral_subtraction)
- **Normalizasyon**: Ses seviyesini hedef LUFS -23 ile dengeleme

### 🏷️ Akıllı Etiketleme
- **Otomatik Etiket Önerileri**: [unsure:], [truncated:], [inaudible:], [overlap:]
- **Güven Skoru Analizi**: Düşük güvenilirlikli segmentleri tespit
- **Kalıp Tanıma**: Belirsizlik ve kesinti ifadelerini otomatik algılama
- **İnsan-in-the-Loop**: Manuel doğrulama ve düzeltme
- **POLLY STEP 2 FAQ Uyumluluk**: Endüstri standardı transkripsiyon kuralları

### 🎯 POLLY STEP 2 FAQ Uyumlu Özellikler
- **FAQ 3**: Konuşma dışı gürültü filtreleme (burp, chuckle, kiss, gnaw vb.)
- **FAQ 7**: Uzatılmış kelime normalleştirme ("yessss" → "yes")
- **FAQ 8**: Gerçek vs. sözlü gülme ayrımı
- **FAQ 9**: Özel isim düzeltme (ünlüler, şirketler için güvenilir kaynak kontrolü)
- **FAQ 15**: Ağır aksan tespiti ([unsure: ] ile işaretleme)
- **FAQ 16**: Sadece ana konuşmacı transkripsiyon
- **FAQ 17**: Çoklu ses analizi (anlamlı konuşma süresi bazlı)
- **FAQ 18**: Sayı formatı düzeltme ("50 000" → "elli bin")
- **FAQ 19-20**: Büyük/küçük harf kuralları (ilk kelime küçük harf)

### 🎛️ Kullanıcı Deneyimi
- **Terminal Arayüzü**: Zengin, interaktif CLI
- **Metin Editörü Entegrasyonu**: VS Code, Sublime, Vim desteği
- **İlerleme Gösterimi**: Yüzdelikli çubuk ile gerçek zamanlı işlem takibi
- **Ses Oynatma**: ffplay ile terminal ses kontrolü

## 📋 Gereksinimler

### Sistem Gereksinimleri
- Python 3.8+
- FFmpeg (ses işleme için)
- GPU (opsiyonel, pyannote.audio için hızlandırma)

### API Anahtarları
- OpenAI API Key (Whisper transkripsiyon)
- Google Gemini API Key (AI analiz)
- HuggingFace Token (pyannote.audio modelleri)

## 🚀 Kurulum

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/kullanici/podcast-transcription.git
cd podcast-transcription
```

### 2. Sanal Ortam Oluşturun
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt

# spaCy Türkçe modelini yükleyin
python -m spacy download tr_core_news_lg
```

### 4. FFmpeg Kurulumu

#### Windows
```bash
# Chocolatey ile
choco install ffmpeg

# Scoop ile
scoop install ffmpeg
```

#### macOS
```bash
# Homebrew ile
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

### 5. API Anahtarlarını Ayarlayın
```bash
# .env dosyasını oluşturun
cp .env.example .env

# API anahtarlarınızı .env dosyasına ekleyin
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key_here  
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

### 6. HuggingFace Model Erişimi
[pyannote.audio](https://huggingface.co/pyannote/speaker-diarization-3.1) modellerine erişim için:
1. HuggingFace hesabı oluşturun
2. Model sayfasında kullanım koşullarını kabul edin
3. Access token oluşturup .env dosyasına ekleyin

## 📖 Kullanım

### Temel Kullanım
```bash
# Basit transkripsiyon
python main.py podcast.mp3

# Ses dosyasını dinleme ile
python main.py interview.wav

# Özel konfigürasyon ile
python main.py meeting.m4a --config my_config.yaml
```

### Gelişmiş Kullanım
```bash
# İnteraktif olmayan mod
python main.py podcast.mp3 --no-interactive

# Özel çıktı dizini
python main.py interview.wav --output results/

# Verbose logging
python main.py meeting.m4a --verbose

# Yardım
python main.py --help
```

### Desteklenen Ses Formatları
- MP3
- WAV  
- FLAC
- M4A
- OGG

## ⚙️ Konfigürasyon

`config.yaml` dosyasında ayarlanabilir parametreler:

### Whisper Ayarları
```yaml
whisper:
  model: "large-v3"
  language: "tr"
  temperature: 0.0
```

### Konuşmacı Tanıma
```yaml
speaker_diarization:
  model: "pyannote/speaker-diarization-3.1" 
  min_speakers: 1
  max_speakers: 10
```

### Ses Kalitesi Eşikleri
```yaml
audio_quality:
  thresholds:
    pesq_min: 1.0
    stoi_min: 0.3
    snr_min: 5.0
```

### Etiketleme Sistemi
```yaml
labeling:
  tags:
    unsure: "[unsure:]"
    truncated: "[truncated:]"
    inaudible: "[inaudible:]"
    overlap: "[overlap:]"
  confidence_threshold: 0.6
```

## 📊 Çıktı Formatları

### JSON Çıktısı
```json
{
  "audio_info": {
    "sure_saniye": 1234.5,
    "kalite_skoru": 85.2,
    "format": "mp3"
  },
  "transcription": {
    "text": "Transkripsiyon metni...",
    "segments": [...],
    "confidence": 0.92
  },
  "speaker_diarization": {
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "main_speaker": "SPEAKER_00"
  },
  "quality_assessment": {...},
  "label_suggestions": {...}
}
```

### Metin Çıktısı
```
[00:01.2s - 00:05.8s] Merhaba, bugünkü podcast bölümümüze hoş geldiniz.
[00:06.1s - 00:12.3s] [unsure:] Bugün konuğumuz teknoloji alanında...
[00:13.0s - 00:18.5s] [overlap:] İki kişi aynı anda konuşuyor...
```

## 🏗️ Mimari

### Modüler Yapı
```
src/
├── core/           # Ana orkestratör
├── audio/          # Ses işleme (pydub, librosa)
├── models/         # AI modelleri (Whisper, Gemini, pyannote)
├── quality/        # Kalite değerlendirme (PESQ, STOI)
├── labeling/       # Otomatik etiketleme
├── nlp/           # spaCy NLP analizi
└── utils/         # Yardımcı araçlar
```

### Veri Akışı
1. **Ses Yükleme** → AudioProcessor
2. **Kalite Analizi** → AudioQualityEvaluator  
3. **Konuşmacı Ayırma** → SpeakerDiarizer
4. **Transkripsiyon** → WhisperTranscriber
5. **AI Analizi** → GeminiAnalyzer
6. **NLP İşleme** → SpacyAnalyzer
7. **Etiketleme** → AutoLabeler
8. **İnsan Düzeltmesi** → Terminal Editor

## 🔧 Geliştirme

### Test Çalıştırma
```bash
pytest tests/
```

### Kod Kalitesi
```bash
# Linting
flake8 src/

# Formatting  
black src/
```

### Debug Modu
```bash
DEBUG=true python main.py podcast.mp3 --verbose
```

## 📈 Performans

### Benchmark Sonuçları
| Ses Süresi | İşlem Süresi | Bellek Kullanımı | GPU Hızlandırma |
|------------|--------------|------------------|-----------------|
| 10 dk      | ~2 dk        | 2-4 GB          | 3x hızlı        |
| 30 dk      | ~5 dk        | 3-6 GB          | 3x hızlı        |
| 60 dk      | ~8 dk        | 4-8 GB          | 3x hızlı        |

### Optimizasyon İpuçları
- GPU kullanımı için CUDA kurulumu
- Büyük dosyalar için chunk işleme
- Paralel işlem için batch processing

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`) 
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [OpenAI Whisper](https://github.com/openai/whisper) - Transkripsiyon modeli
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [spaCy](https://spacy.io/) - NLP kütüphanesi
- [Rich](https://github.com/Textualize/rich) - Terminal UI

## 📞 İletişim

- GitHub Issues: [Sorun bildirin](https://github.com/kullanici/podcast-transcription/issues)
- Email: info@podcasttranscription.com
- Dokümantasyon: [Wiki](https://github.com/kullanici/podcast-transcription/wiki)

---

⭐ Bu proje size yardımcı olduysa, GitHub'da yıldız vermeyi unutmayın! 
