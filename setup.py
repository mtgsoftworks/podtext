"""
Podcast Transkripsiyon Sistemi - Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# README dosyasını oku
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Requirements dosyasını oku
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split("\n")
    requirements = [req for req in requirements if req and not req.startswith("#")]

setup(
    name="podcast-transcription",
    version="1.1.0",
    author="Podcast Transkripsiyon Ekibi",
    author_email="info@podcasttranscription.com",
    description="Modern yapay zeka teknolojileri ve Türkçe BERT modeli kullanarak podcast transkripsiyon sistemi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kullanici/podcast-transcription",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    py_modules=["main"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch[cuda]",
            "torchaudio[cuda]",
        ],
    },
    entry_points={
        "console_scripts": [
            "podcast-transcription=main:main",
            "podtext=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "podcast",
        "transcription", 
        "speech-to-text",
        "ai",
        "whisper",
        "speaker-diarization",
        "nlp",
        "turkish",
        "turkish-bert",
        "huggingface",
        "transformers",
        "audio-processing"
    ],
    project_urls={
        "Bug Reports": "https://github.com/kullanici/podcast-transcription/issues",
        "Source": "https://github.com/kullanici/podcast-transcription",
        "Documentation": "https://github.com/kullanici/podcast-transcription/wiki",
    },
) 