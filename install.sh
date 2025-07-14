#!/bin/bash

# Podcast Transkripsiyon Sistemi - Otomatik Kurulum Script'i
# Linux ve macOS için

set -e  # Hata durumunda script'i durdur

echo "🎙️ Podcast Transkripsiyon Sistemi - Otomatik Kurulum"
echo "=================================================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonksiyonlar
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Python sürümünü kontrol et
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION bulundu"
            return 0
        else
            print_error "Python 3.8+ gerekli, $PYTHON_VERSION bulundu"
            return 1
        fi
    else
        print_error "Python3 bulunamadı!"
        return 1
    fi
}

# FFmpeg kontrolü ve kurulumu
install_ffmpeg() {
    if command -v ffmpeg &> /dev/null; then
        print_success "FFmpeg zaten kurulu"
        return 0
    fi
    
    print_info "FFmpeg kuruluyor..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux (Ubuntu/Debian)
        if command -v apt &> /dev/null; then
            sudo apt update
            sudo apt install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            sudo pacman -S ffmpeg
        else
            print_error "Paket yöneticisi bulunamadı. FFmpeg'i manuel olarak kurun."
            return 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            print_error "Homebrew bulunamadı. FFmpeg'i manuel olarak kurun."
            return 1
        fi
    else
        print_error "Desteklenmeyen işletim sistemi: $OSTYPE"
        return 1
    fi
    
    if command -v ffmpeg &> /dev/null; then
        print_success "FFmpeg başarıyla kuruldu"
    else
        print_error "FFmpeg kurulumu başarısız"
        return 1
    fi
}

# Sanal ortam oluştur
create_venv() {
    if [ -d "venv" ]; then
        print_warning "Sanal ortam zaten mevcut"
        read -p "Yeniden oluşturmak istiyor musunuz? (e/h): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ee]$ ]]; then
            rm -rf venv
        else
            return 0
        fi
    fi
    
    print_info "Sanal ortam oluşturuluyor..."
    python3 -m venv venv
    print_success "Sanal ortam oluşturuldu"
}

# Bağımlılıkları yükle
install_dependencies() {
    print_info "Sanal ortam etkinleştiriliyor..."
    source venv/bin/activate
    
    print_info "Pip güncelleniyor..."
    pip install --upgrade pip
    
    print_info "Ana bağımlılıklar yükleniyor..."
    pip install -r requirements.txt
    
    print_info "Türkçe HuggingFace modelleri kontrol ediliyor..."
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-cased')" 2>/dev/null || echo "Türkçe BERT modeli ilk kullanımda otomatik indirilecek"
    
    print_success "Bağımlılıklar başarıyla yüklendi"
}

# Konfigürasyon dosyasını kopyala
setup_config() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Konfigürasyon dosyası oluşturuldu (.env)"
            print_warning "API anahtarlarınızı .env dosyasına eklemeyi unutmayın!"
        else
            print_warning ".env.example dosyası bulunamadı"
        fi
    else
        print_info ".env dosyası zaten mevcut"
    fi
}

# Dizinleri oluştur
create_directories() {
    print_info "Gerekli dizinler oluşturuluyor..."
    mkdir -p logs cache temp output models
    print_success "Dizinler oluşturuldu"
}

# Test çalıştır
run_test() {
    print_info "Kurulum testi yapılıyor..."
    source venv/bin/activate
    
    if python -c "from src.utils.config import Config; print('✅ Konfigürasyon OK')"; then
        print_success "Kurulum testi başarılı"
    else
        print_error "Kurulum testi başarısız"
        return 1
    fi
}

# Kullanım bilgileri
show_usage() {
    echo
    echo "🎉 Kurulum tamamlandı!"
    echo
    echo "Kullanım:"
    echo "1. Sanal ortamı etkinleştirin:"
    echo "   source venv/bin/activate"
    echo
    echo "2. API anahtarlarınızı .env dosyasına ekleyin"
    echo
    echo "3. Sistemi çalıştırın:"
    echo "   python main.py ses_dosyasi.mp3"
    echo
    echo "Daha fazla bilgi için:"
    echo "   python main.py --help"
    echo
}

# Ana kurulum fonksiyonu
main() {
    echo "Kurulum başlatılıyor..."
    echo
    
    # Python kontrolü
    if ! check_python; then
        exit 1
    fi
    
    # FFmpeg kurulumu
    if ! install_ffmpeg; then
        print_warning "FFmpeg kurulumu başarısız, manuel kurulum gerekebilir"
    fi
    
    # Sanal ortam
    create_venv
    
    # Bağımlılıklar
    install_dependencies
    
    # Konfigürasyon
    setup_config
    
    # Dizinler
    create_directories
    
    # Test
    if ! run_test; then
        print_error "Kurulum sırasında sorun oluştu"
        exit 1
    fi
    
    # Kullanım bilgileri
    show_usage
}

# Script'i çalıştır
main "$@" 