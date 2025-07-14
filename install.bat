@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM Podcast Transkripsiyon Sistemi - Windows Kurulum Script'i

echo 🎙️ Podcast Transkripsiyon Sistemi - Otomatik Kurulum
echo ==================================================

REM Python kontrolü
echo ⏳ Python kontrol ediliyor...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python bulunamadı! Python 3.8+ kurmanız gerekiyor.
    echo    https://www.python.org/downloads/ adresinden indirin.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% bulundu

REM FFmpeg kontrolü
echo ⏳ FFmpeg kontrol ediliyor...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ FFmpeg bulunamadı!
    echo.
    echo FFmpeg kurulumu için şu seçeneklerden birini kullanın:
    echo 1. Chocolatey: choco install ffmpeg
    echo 2. Scoop: scoop install ffmpeg
    echo 3. Manuel: https://ffmpeg.org/download.html
    echo.
    set /p "choice=Devam etmek istiyor musunuz? (E/H): "
    if /i "!choice!" neq "E" exit /b 0
) else (
    echo ✅ FFmpeg bulundu
)

REM Sanal ortam kontrolü
if exist "venv" (
    echo ⚠️ Sanal ortam zaten mevcut
    set /p "recreate=Yeniden oluşturmak istiyor musunuz? (E/H): "
    if /i "!recreate!" equ "E" (
        echo ⏳ Eski sanal ortam siliniyor...
        rmdir /s /q venv
    ) else (
        goto :skip_venv
    )
)

echo ⏳ Sanal ortam oluşturuluyor...
python -m venv venv
if errorlevel 1 (
    echo ❌ Sanal ortam oluşturulamadı!
    pause
    exit /b 1
)
echo ✅ Sanal ortam oluşturuldu

:skip_venv
echo ⏳ Sanal ortam etkinleştiriliyor...
call venv\Scripts\activate.bat

echo ⏳ Pip güncelleniyor...
python -m pip install --upgrade pip

echo ⏳ Ana bağımlılıklar yükleniyor...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Bağımlılık yüklemesi başarısız!
    pause
    exit /b 1
)

echo ⏳ Türkçe HuggingFace modelleri kontrol ediliyor...
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-cased')" 2>nul
if errorlevel 1 (
    echo ℹ️ Türkçe BERT modeli ilk kullanımda otomatik indirilecek
)

echo ✅ Bağımlılıklar yüklendi

REM Konfigürasyon dosyası
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo ✅ Konfigürasyon dosyası oluşturuldu (.env)
        echo ⚠️ API anahtarlarınızı .env dosyasına eklemeyi unutmayın!
    ) else (
        echo ⚠️ .env.example dosyası bulunamadı
    )
) else (
    echo ℹ️ .env dosyası zaten mevcut
)

REM Dizinleri oluştur
echo ⏳ Gerekli dizinler oluşturuluyor...
if not exist "logs" mkdir logs
if not exist "cache" mkdir cache
if not exist "temp" mkdir temp
if not exist "output" mkdir output
if not exist "models" mkdir models
echo ✅ Dizinler oluşturuldu

REM Test
echo ⏳ Kurulum testi yapılıyor...
python -c "from src.utils.config import Config; print('✅ Konfigürasyon OK')" 2>nul
if errorlevel 1 (
    echo ❌ Kurulum testi başarısız!
    pause
    exit /b 1
)
echo ✅ Kurulum testi başarılı

echo.
echo 🎉 Kurulum tamamlandı!
echo.
echo Kullanım:
echo 1. Sanal ortamı etkinleştirin:
echo    venv\Scripts\activate.bat
echo.
echo 2. API anahtarlarınızı .env dosyasına ekleyin
echo.
echo 3. Sistemi çalıştırın:
echo    python main.py ses_dosyasi.mp3
echo.
echo Daha fazla bilgi için:
echo    python main.py --help
echo.

pause 