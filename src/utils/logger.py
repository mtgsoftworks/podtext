"""
Logging utility modülü
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def setup_logger(
    name: str = "podcast_transcription",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size: str = "10MB",
    backup_count: int = 5
) -> logging.Logger:
    """
    Gelişmiş logging sistemi kur
    
    Args:
        name: Logger adı
        level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log dosyası yolu
        max_size: Maksimum dosya boyutu
        backup_count: Backup dosya sayısı
    
    Returns:
        Konfigüre edilmiş logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Mevcut handler'ları temizle
    logger.handlers.clear()
    
    # Console handler (Rich ile)
    console = Console()
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    
    console_format = logging.Formatter(
        "%(message)s",
        datefmt="[%X]"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Dosya handler (eğer belirtilmişse)
    if log_file:
        # Log dizinini oluştur
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dosya boyutunu byte'a çevir
        size_map = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        size_str = max_size.upper()
        
        size_bytes = 10 * 1024 * 1024  # Varsayılan 10MB
        for suffix, multiplier in size_map.items():
            if size_str.endswith(suffix):
                size_value = float(size_str[:-len(suffix)])
                size_bytes = int(size_value * multiplier)
                break
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=size_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "podcast_transcription") -> logging.Logger:
    """
    Mevcut logger'ı al veya varsayılan logger oluştur
    
    Args:
        name: Logger adı
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Henüz konfigüre edilmemişse varsayılan ayarlarla kur
        setup_logger(
            name=name,
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_file="logs/podcast_transcription.log"
        )
    
    return logger


class LoggerMixin:
    """Logger mixin sınıfı"""
    
    @property
    def logger(self) -> logging.Logger:
        """Logger property'si"""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """Info log"""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Warning log"""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Error log"""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Debug log"""
        self.logger.debug(message, **kwargs) 