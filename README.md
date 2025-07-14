# Podcast Transcription System
A comprehensive tool that performs high-quality transcription of podcast audio recordings using modern artificial intelligence technologies.

✨ **Features**

## 🎯 Transcription and Analysis
- **High Accuracy Transcription**: OpenAI Whisper Large v3 Turbo
- **Speaker Recognition and Separation**: Advanced speaker diarization with pyannote.audio
- **Main Speaker Filtering**: Only the main speaker's speech is transcribed
- **AI-Powered Quality Analysis**: Smart analysis with Google Gemini 2.5 Pro
- **Linguistic Analysis**: Turkish NLP processing with spaCy

## 🔍 Quality Assessment
- **Audio Quality Metrics**: PESQ, STOI objective measurements
- **Categorical Evaluation**: Detection of unclear speech, heavy accents, synthesized speech
- **SNR and Clipping Analysis**: Technical audio quality control
- **Automatic Suggestion System**: Quality improvement recommendations

## 🔊 Audio Preprocessing
- **Noise Reduction**: Background noise reduction (spectral_subtraction)
- **Normalization**: Audio level balancing with target LUFS -23

## 🏷️ Smart Labeling
- **Automatic Label Suggestions**: [unsure:], [truncated:], [inaudible:], [overlap:]
- **Confidence Score Analysis**: Detection of low-confidence segments
- **Pattern Recognition**: Automatic detection of uncertainty and interruption expressions
- **Human-in-the-Loop**: Manual verification and correction
- **POLLY STEP 2 FAQ Compliance**: Industry standard transcription rules

## 🎯 POLLY STEP 2 FAQ Compliant Features
- **FAQ 3**: Non-speech noise filtering (burp, chuckle, kiss, gnaw, etc.)
- **FAQ 7**: Extended word normalization ("yessss" → "yes")
- **FAQ 8**: Real vs. verbal laughter distinction
- **FAQ 9**: Proper name correction (reliable source checking for celebrities, companies)
- **FAQ 15**: Heavy accent detection (marked with [unsure:])
- **FAQ 16**: Main speaker only transcription
- **FAQ 17**: Multi-voice analysis (based on meaningful speech duration)
- **FAQ 18**: Number format correction ("50 000" → "fifty thousand")
- **FAQ 19-20**: Capitalization rules (first word lowercase)

## 🎛️ User Experience
- **Terminal Interface**: Rich, interactive CLI
- **Text Editor Integration**: VS Code, Sublime, Vim support
- **Progress Display**: Real-time processing tracking with percentage bar
- **Audio Playback**: Terminal audio control with ffplay

## 📋 Requirements

### System Requirements
- Python 3.8+
- FFmpeg (for audio processing)
- GPU (optional, for pyannote.audio acceleration)

### API Keys
- OpenAI API Key (Whisper transcription)
- Google Gemini API Key (AI analysis)
- HuggingFace Token (pyannote.audio models)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/username/podcast-transcription.git
cd podcast-transcription
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

# Install spaCy Turkish model
python -m spacy download tr_core_news_lg
```

### 4. FFmpeg Installation

#### Windows
```bash
# With Chocolatey
choco install ffmpeg

# With Scoop
scoop install ffmpeg
```

#### macOS
```bash
# With Homebrew
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

### 5. Set Up API Keys
```bash
# Create .env file
cp .env.example .env

# Add your API keys to .env file
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key_here  
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

### 6. HuggingFace Model Access
For pyannote.audio model access:
1. Create a HuggingFace account
2. Accept terms of use on the model page
3. Create an access token and add it to the .env file

## 📖 Usage

### Basic Usage
```bash
# Simple transcription
python main.py podcast.mp3

# With audio listening
python main.py interview.wav

# With custom configuration
python main.py meeting.m4a --config my_config.yaml
```

### Advanced Usage
```bash
# Non-interactive mode
python main.py podcast.mp3 --no-interactive

# Custom output directory
python main.py interview.wav --output results/

# Verbose logging
python main.py meeting.m4a --verbose

# Help
python main.py --help
```

### Supported Audio Formats
- MP3
- WAV
- FLAC
- M4A
- OGG

## ⚙️ Configuration

Configurable parameters in `config.yaml`:

### Whisper Settings
```yaml
whisper:
  model: "large-v3"
  language: "tr"
  temperature: 0.0
```

### Speaker Diarization
```yaml
speaker_diarization:
  model: "pyannote/speaker-diarization-3.1" 
  min_speakers: 1
  max_speakers: 10
```

### Audio Quality Thresholds
```yaml
audio_quality:
  thresholds:
    pesq_min: 1.0
    stoi_min: 0.3
    snr_min: 5.0
```

### Labeling System
```yaml
labeling:
  tags:
    unsure: "[unsure:]"
    truncated: "[truncated:]"
    inaudible: "[inaudible:]"
    overlap: "[overlap:]"
  confidence_threshold: 0.6
```

## 📊 Output Formats

### JSON Output
```json
{
  "audio_info": {
    "duration_seconds": 1234.5,
    "quality_score": 85.2,
    "format": "mp3"
  },
  "transcription": {
    "text": "Transcription text...",
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

### Text Output
```
[00:01.2s - 00:05.8s] Hello, welcome to today's podcast episode.
[00:06.1s - 00:12.3s] [unsure:] Today our guest is in the technology field...
[00:13.0s - 00:18.5s] [overlap:] Two people speaking simultaneously...
```

## 🏗️ Architecture

### Modular Structure
```
src/
├── core/           # Main orchestrator
├── audio/          # Audio processing (pydub, librosa)
├── models/         # AI models (Whisper, Gemini, pyannote)
├── quality/        # Quality assessment (PESQ, STOI)
├── labeling/       # Automatic labeling
├── nlp/           # spaCy NLP analysis
└── utils/         # Utility tools
```

### Data Flow
1. Audio Loading → AudioProcessor
2. Quality Analysis → AudioQualityEvaluator
3. Speaker Separation → SpeakerDiarizer
4. Transcription → WhisperTranscriber
5. AI Analysis → GeminiAnalyzer
6. NLP Processing → SpacyAnalyzer
7. Labeling → AutoLabeler
8. Human Correction → Terminal Editor

## 🔧 Development

### Run Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Linting
flake8 src/

# Formatting  
black src/
```

### Debug Mode
```bash
DEBUG=true python main.py podcast.mp3 --verbose
```

## 📈 Performance

### Benchmark Results
| Audio Duration | Processing Time | Memory Usage | GPU Acceleration |
|---------------|----------------|--------------|------------------|
| 10 min        | ~2 min         | 2-4 GB       | 3x faster        |
| 30 min        | ~5 min         | 3-6 GB       | 3x faster        |
| 60 min        | ~8 min         | 4-8 GB       | 3x faster        |

### Optimization Tips
- CUDA installation for GPU usage
- Chunk processing for large files
- Batch processing for parallel operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** - Transcription model
- **pyannote.audio** - Speaker diarization
- **spaCy** - NLP library
- **Rich** - Terminal UI

⭐ **If this project helped you, don't forget to give it a star on GitHub!**
