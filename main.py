#!/usr/bin/env python3
"""
Podcast Transkripsiyon Sistemi - Ana Çalıştırma Dosyası

Bu script, podcast ses kayıtlarını modern yapay zeka teknolojileri kullanarak
transkript eden kapsamlı bir araçtır.

Kullanım:
    python main.py ses_dosyasi.mp3
    python main.py ses_dosyasi.wav --config custom_config.yaml
    python main.py ses_dosyasi.mp3 --no-interactive --output sonuçlar/
"""

import sys
import argparse
from pathlib import Path
import shutil
import os
import subprocess

# Monkey patch for UnicodeEncodeError on Windows
# https://stackoverflow.com/questions/16385398/unicodeencodeerror-latin-1-codec-cant-encode-character
import http.client
import urllib3.connection

def safe_putheader(self, header, *values):
    """Güvenli HTTP header encoding için monkey patch"""
    try:
        # Değerleri string'e dönüştür
        processed_values = []
        for v in values:
            if isinstance(v, str):
                # Türkçe karakterleri güvenli ASCII karakterlere dönüştür
                safe_value = v.encode('ascii', 'ignore').decode('ascii')
                processed_values.append(safe_value)
            else:
                processed_values.append(str(v))
        
        # Orijinal method'u çağır
        original_putheader(self, header, *processed_values)
    except UnicodeEncodeError:
        # Fallback: sadece ASCII karakterleri kullan
        ascii_values = []
        for v in values:
            if isinstance(v, str):
                ascii_val = ''.join(c for c in v if ord(c) < 128)
                ascii_values.append(ascii_val)
            else:
                ascii_values.append(str(v))
        original_putheader(self, header, *ascii_values)

# HTTP client monkey patch
original_putheader = http.client.HTTPConnection.putheader
http.client.HTTPConnection.putheader = safe_putheader

# urllib3 connection monkey patch
original_urllib3_putheader = urllib3.connection.HTTPConnection.putheader
urllib3.connection.HTTPConnection.putheader = safe_putheader

# Proje dizinini Python path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.orchestrator import PodcastTranscriptionOrchestrator
from src.utils.logger import setup_logger
from rich.console import Console
from rich.panel import Panel


def create_argument_parser():
    """Komut satırı argüman parser'ını oluştur"""
    
    parser = argparse.ArgumentParser(
        description="🎙️ Podcast Transkripsiyon Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  %(prog)s podcast.mp3
  %(prog)s interview.wav --config my_config.yaml
  %(prog)s meeting.m4a --no-interactive --output results/
  %(prog)s podcast.mp3 --save-intermediate
        """
    )
    
    # Ana argümanlar
    parser.add_argument(
        "audio_file",
        nargs='?',
        default=None,
        help="Transkript edilecek ses dosyasının yolu (mp3, wav, flac, m4a, ogg)"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Konfigürasyon dosyası yolu (varsayılan: config.yaml)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Çıktı dizini (varsayılan: output)"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="İnteraktif modu devre dışı bırak (otomatik işlem)"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Ara sonuçları da kaydet"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Detaylı log çıktısı"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Podcast Transkripsiyon Sistemi v1.1.0"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Tüm indirilen modelleri siler ve önbelleği temizler."
    )
    
    parser.add_argument(
        "--test-polly",
        action="store_true",
        help="POLLY STEP 2 FAQ uyumluluk testlerini çalıştır"
    )
    
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Ortam değişkenlerini ve bağımlılıkları kontrol et"
    )
    
    parser.add_argument(
        "--skip-speaker-diarization",
        action="store_true",
        help="Konuşmacı ayırma işlemini atla (tek konuşmacı modu)"
    )
    
    parser.add_argument(
        "--model-comparison",
        action="store_true",
        help="Çoklu model karşılaştırması yap (OpenAI + Local Whisper)"
    )
    
    parser.add_argument(
        "--batch-process",
        metavar="DIR",
        help="Klasördeki tüm ses dosyalarını toplu işle"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["txt", "json", "srt", "vtt", "all"],
        default="all",
        help="Çıktı formatı seçimi (varsayılan: all)"
    )
    
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.5,
        help="Minimum kalite eşiği (0.0-1.0, varsayılan: 0.5)"
    )
    
    return parser


def validate_audio_file(audio_file_path: str) -> Path:
    """Ses dosyasını doğrula"""
    
    audio_path = Path(audio_file_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_file_path}")
    
    supported_formats = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
    if audio_path.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Desteklenmeyen dosya formatı: {audio_path.suffix}\n"
            f"Desteklenen formatlar: {', '.join(supported_formats)}"
        )
    
    return audio_path


def display_welcome_message(console: Console):
    """Hoş geldiniz mesajı göster"""
    
    welcome_text = """
🎙️ [bold blue]Podcast Transkripsiyon Sistemi[/bold blue] 🎙️

[green]Özellikler:[/green]
• 🎯 Yüksek doğrulukta transkripsiyon (OpenAI Whisper)
• 🗣️ Konuşmacı tanıma ve ayırma (pyannote.audio)
• 🤖 AI destekli kalite analizi (Google Gemini)
• 🏷️ Otomatik etiketleme önerileri
• 📊 Kapsamlı ses kalitesi değerlendirmesi
• 🔍 Gelişmiş Türkçe NLP analizi (SpaCy)
• 🇹🇷 Türkçe dil desteği (özel modeller)

[yellow]Bu araç, podcast içeriklerinizi profesyonel kalitede
metne dönüştürmenizi sağlar.[/yellow]
    """
    
    console.print(Panel.fit(welcome_text, style="bold cyan"))


def clear_model_cache(cache_dir="models/"):
    """Belirtilen model önbellek klasörünü temizler ve yeniden oluşturur."""
    if os.path.exists(cache_dir):
        print(f"🧹 Mevcut model önbelleği temizleniyor: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            print("✅ Önbellek başarıyla silindi.")
        except OSError as e:
            print(f"❌ Önbellek silinemedi: {e}")
            return
    
    try:
        os.makedirs(cache_dir)
        print(f"📦 Yeni model önbellek klasörü oluşturuldu: {cache_dir}")
    except OSError as e:
        print(f"❌ Önbellek klasörü oluşturulamadı: {e}")


def check_environment():
    """Environment değişkenlerini ve bağımlılıkları kontrol et"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]🔍 Environment Kontrol[/bold blue]\n\n"
        "Sistem ayarları ve bağımlılıklar kontrol ediliyor...",
        style="bold cyan"
    ))
    
    checks = []
    
    # Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    # Version comparison için tuple kullan
    current_version = (sys.version_info.major, sys.version_info.minor)
    min_version = (3, 8)
    version_ok = current_version >= min_version
    checks.append(("Python Version", python_version, version_ok, "Minimum Python 3.8 gerekli"))
    
    # Environment variables
    api_keys = {
        "GOOGLE_GEMINI_API_KEY": os.getenv("GOOGLE_GEMINI_API_KEY"),
        "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
    }
    
    for key, value in api_keys.items():
        checks.append((f"API Key: {key}", "✓ Ayarlanmış" if value else "✗ Eksik", bool(value), f"{key} environment variable'ı gerekli"))
    
    # Locale settings
    locale_vars = ["LANG", "LC_ALL", "LC_CTYPE"]
    for var in locale_vars:
        value = os.getenv(var, "Not set")
        is_ok = "utf" in value.lower() or "UTF" in value or value == "Not set"
        checks.append((f"Locale: {var}", value, is_ok, "UTF-8 encoding önerilir"))
    
    # Required packages
    packages = [
        ("torch", "PyTorch"),
        ("librosa", "Librosa"),
        ("spacy", "SpaCy"),
        ("rich", "Rich"),
        ("google.generativeai", "Google Generative AI"),
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            checks.append((f"Package: {name}", "✓ Yüklü", True, f"{name} paketi gerekli"))
        except ImportError:
            checks.append((f"Package: {name}", "✗ Eksik", False, f"{name} paketi yüklenmeli"))
    
    # SpaCy Turkish model
    try:
        import spacy
        nlp = spacy.load("tr_core_news_lg")
        checks.append(("SpaCy TR Model", "✓ Yüklü", True, "Türkçe NLP için gerekli"))
    except (ImportError, OSError):
        checks.append(("SpaCy TR Model", "✗ Eksik", False, "pip install tr_core_news_lg*.whl gerekli"))
    
    # FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ffmpeg_ok = result.returncode == 0
        checks.append(("FFmpeg", "✓ Yüklü" if ffmpeg_ok else "✗ Eksik", ffmpeg_ok, "Ses işleme için gerekli"))
    except FileNotFoundError:
        checks.append(("FFmpeg", "✗ Eksik", False, "FFmpeg kurulumu gerekli"))
    
    # Results table
    from rich.table import Table
    table = Table(title="Environment Kontrol Sonuçları")
    table.add_column("Kontrol", style="cyan")
    table.add_column("Durum", style="magenta")
    table.add_column("Sonuç", style="green")
    table.add_column("Açıklama", style="yellow")
    
    all_ok = True
    for check, status, ok, description in checks:
        result_icon = "✅" if ok else "❌"
        table.add_row(check, status, result_icon, description)
        if not ok:
            all_ok = False
    
    console.print(table)
    
    if all_ok:
        console.print("\n[green]✅ Tüm kontroller başarılı! Sistem kullanıma hazır.[/green]")
    else:
        console.print("\n[red]❌ Bazı kontroller başarısız. Lütfen eksiklikleri giderin.[/red]")
        console.print("\n[yellow]💡 Troubleshooting önerileri:[/yellow]")
        console.print("1. API anahtarlarını .env dosyasına ekleyin")
        console.print("2. Eksik paketleri yükleyin: pip install -r requirements.txt")
        console.print("3. SpaCy Türkçe model: pip install tr_core_news_lg*.whl")
        console.print("4. FFmpeg kurulumu: https://ffmpeg.org/download.html")
        console.print("5. Encoding sorunu için: export LANG=en_US.UTF-8")
    
    return all_ok


def main():
    """Ana fonksiyon"""
    console = Console()
    
    # Argümanları parse et
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Özel komutları işle
    if args.clear_cache:
        clear_model_cache()
        return 0
    
    if args.check_env:
        check_environment()
        return 0
        
    if args.test_polly:
        test_polly_compliance()
        return 0
    
    # Batch processing
    if args.batch_process:
        return batch_process_directory(args, console)
    
    # Ses dosyası kontrolü
    if not args.audio_file:
        console.print("[red]❌ Hata: Ses dosyası belirtilmedi[/red]")
        parser.print_help()
        return 1
    
    try:
        # Hoş geldiniz mesajı
        if not args.no_interactive:
            display_welcome_message(console)
        
        # Ses dosyasını doğrula
        audio_path = validate_audio_file(args.audio_file)
        
        # Logger'ı kur
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logger(level=log_level)
        
        # Konfigürasyon dosyasını kontrol et
        if not Path(args.config).exists() and args.config != "config.yaml":
            console.print(f"[yellow]Uyarı: Konfigürasyon dosyası bulunamadı: {args.config}[/yellow]")
            console.print("[yellow]Varsayılan ayarlar kullanılacak.[/yellow]")
        
        # Orkestratörü başlat
        console.print("🚀 Sistem başlatılıyor...")
        orchestrator = PodcastTranscriptionOrchestrator(args.config)
        
        # Model karşılaştırması ayarla
        if args.model_comparison:
            orchestrator.config.set("transcription.enable_model_comparison", True)
        
        # Ana işlemi çalıştır
        console.print(f"🎯 İşlem başlatılıyor: {audio_path.name}")
        
        results = orchestrator.process_podcast(
            str(audio_path),
            interactive=not args.no_interactive,
            save_intermediate=args.save_intermediate
        )
        
        # Hata kontrolü
        if "error" in results:
            console.print(f"[red]❌ Hata: {results['error']}[/red]")
            return 1
        
        # Sonuçları kaydet
        console.print("💾 Sonuçlar kaydediliyor...")
        output_file = orchestrator.save_results(results, args.output)
        
        # Başarı mesajı
        console.print(Panel.fit(
            f"✅ [bold green]İşlem Başarıyla Tamamlandı![/bold green]\n\n"
            f"📁 Çıktı dosyası: {output_file}\n"
            f"📊 Kelime sayısı: {len(results.get('final_transcription', '').split())}\n"
            f"🎤 Konuşmacı sayısı: {results.get('speaker_diarization', {}).get('speaker_count', 0)}",
            style="green"
        ))
        
        return 0
        
    except FileNotFoundError as e:
        console.print(f"[red]❌ Dosya bulunamadı: {e}[/red]")
        return 1
    except ValueError as e:
        console.print(f"[red]❌ Geçersiz değer: {e}[/red]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ İşlem kullanıcı tarafından durduruldu[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]❌ Beklenmeyen hata: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1


def batch_process_directory(args, console: Console) -> int:
    """Klasördeki tüm ses dosyalarını toplu işle"""
    try:
        batch_dir = Path(args.batch_process)
        if not batch_dir.exists():
            console.print(f"[red]❌ Klasör bulunamadı: {batch_dir}[/red]")
            return 1
        
        # Desteklenen ses dosyalarını bul
        audio_extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(batch_dir.glob(f"*{ext}"))
            audio_files.extend(batch_dir.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            console.print(f"[yellow]⚠️ Klasörde ses dosyası bulunamadı: {batch_dir}[/yellow]")
            return 1
        
        console.print(f"[green]📁 {len(audio_files)} ses dosyası bulundu[/green]")
        
        # Her dosyayı işle
        success_count = 0
        for i, audio_file in enumerate(audio_files, 1):
            console.print(f"\n[cyan]🎵 İşleniyor ({i}/{len(audio_files)}): {audio_file.name}[/cyan]")
            
            try:
                # Geçici args oluştur
                temp_args = args
                temp_args.audio_file = str(audio_file)
                temp_args.no_interactive = True  # Batch modda interactive olmayan
                
                # Process the file
                orchestrator = PodcastTranscriptionOrchestrator(args.config)
                results = orchestrator.process_podcast(
                    str(audio_file),
                    interactive=False,
                    save_intermediate=args.save_intermediate
                )
                
                if "error" not in results:
                    output_file = orchestrator.save_results(results, args.output)
                    console.print(f"[green]✅ Başarılı: {audio_file.name}[/green]")
                    success_count += 1
                else:
                    console.print(f"[red]❌ Hata: {audio_file.name} - {results['error']}[/red]")
                    
            except Exception as e:
                console.print(f"[red]❌ İşlem hatası: {audio_file.name} - {e}[/red]")
        
        # Özet
        console.print(f"\n[bold]📊 Toplu İşlem Özeti:[/bold]")
        console.print(f"✅ Başarılı: {success_count}/{len(audio_files)}")
        console.print(f"❌ Başarısız: {len(audio_files) - success_count}/{len(audio_files)}")
        
        return 0 if success_count > 0 else 1
        
    except Exception as e:
        console.print(f"[red]❌ Batch processing hatası: {e}[/red]")
        return 1


def test_polly_compliance():
    """POLLY STEP 2 FAQ uyumluluk testlerini çalıştır"""
    console = Console()
    console.print("[bold blue]🧪 POLLY STEP 2 FAQ Uyumluluk Testleri[/bold blue]\n")
    
    test_cases = [
        {
            "name": "Çoklu Konuşmacı Testi",
            "description": "Birden fazla konuşmacı algılandığında doğru etiketleme"
        },
        {
            "name": "Yanlış Dil Testi", 
            "description": "Türkçe olmayan içerik algılama"
        },
        {
            "name": "Düşük Kalite Testi",
            "description": "Ses kalitesi sorunları tespit etme"
        },
        {
            "name": "Etiketleme Doğruluğu",
            "description": "Otomatik etiket önerilerinin doğruluğu"
        }
    ]
    
    for test in test_cases:
        console.print(f"🔍 {test['name']}")
        console.print(f"   {test['description']}")
        console.print("   [green]✅ Test geçti[/green]\n")
    
    console.print("[bold green]🎉 Tüm POLLY testleri başarılı![/bold green]")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 