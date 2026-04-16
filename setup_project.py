"""
Akıllı Lojistik ve Rota Optimizasyonu
Proje klasör yapısını kuran setup scripti.
Çalıştır: python setup_project.py
"""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent

DIRECTORIES = [
    # Config
    "config",
    # Data katmanları
    "data/raw",
    "data/processed",
    "data/external",
    # Kaynak kod modülleri
    "src/data",
    "src/features",
    "src/models",
    "src/optimization",
    "src/api",
    "src/utils",
    # Notebook'lar (EDA, deneyler)
    "notebooks",
    # Testler
    "tests",
    # Log dosyaları
    "logs",
    # Çıktılar
    "outputs/models",
    "outputs/reports",
    "outputs/routes",
]

# Her modül altına __init__.py eklenecek src dizinleri
SRC_MODULES = [
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/optimization",
    "src/api",
    "src/utils",
    "tests",
]

# Oluşturulacak boş placeholder dosyalar
PLACEHOLDER_FILES = [
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",
    "logs/.gitkeep",
    "outputs/models/.gitkeep",
    "outputs/reports/.gitkeep",
    "outputs/routes/.gitkeep",
    "notebooks/.gitkeep",
]


def create_directories():
    print("Klasörler oluşturuluyor...")
    for directory in DIRECTORIES:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {directory}/")


def create_init_files():
    print("\n__init__.py dosyaları oluşturuluyor...")
    for module in SRC_MODULES:
        init_file = PROJECT_ROOT / module / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
            print(f"  [OK] {module}/__init__.py")
        else:
            print(f"  [--] {module}/__init__.py (zaten var)")


def create_placeholder_files():
    print("\nPlaceholder dosyalar oluşturuluyor...")
    for filepath in PLACEHOLDER_FILES:
        path = PROJECT_ROOT / filepath
        if not path.exists():
            path.touch()
            print(f"  [OK] {filepath}")
        else:
            print(f"  [--] {filepath} (zaten var)")


def create_gitignore():
    gitignore_path = PROJECT_ROOT / ".gitignore"
    if gitignore_path.exists():
        print("\n  [--] .gitignore (zaten var)")
        return

    content = """\
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
.eggs/

# Sanal ortam
venv/
.venv/
env/

# Jupyter
.ipynb_checkpoints/

# Veri dosyaları (büyük CSV'ler)
data/raw/*.csv
data/processed/*.csv
data/external/*.csv

# Modeller (büyük dosyalar)
outputs/models/*.pkl
outputs/models/*.joblib
outputs/models/*.h5

# Log dosyaları
logs/*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Environment variables
.env
*.env
"""
    gitignore_path.write_text(content, encoding="utf-8")
    print("\n  [OK] .gitignore")


def print_tree():
    print("\n" + "=" * 55)
    print("  Proje Yapısı Kurulumu Tamamlandı")
    print("=" * 55)
    print(f"""
hackathon-smart-logistic/
├── config/
│   ├── settings.yaml          ← Tüm konfigürasyon
│   └── config_loader.py       ← YAML yükleyici
├── data/
│   ├── raw/                   ← Ham CSV dosyaları
│   ├── processed/             ← İşlenmiş veriler
│   └── external/              ← Dış veri kaynakları
├── src/
│   ├── data/                  ← Veri yükleme & ön işleme
│   ├── features/              ← Feature engineering
│   ├── models/                ← ML modelleri
│   ├── optimization/          ← Rota optimizasyonu
│   ├── api/                   ← FastAPI endpoint'leri
│   └── utils/                 ← Yardımcı fonksiyonlar
├── notebooks/                 ← EDA & deneyler
├── tests/                     ← Unit testler
├── logs/                      ← Uygulama logları
├── outputs/
│   ├── models/                ← Eğitilmiş model dosyaları
│   ├── reports/               ← Analiz raporları
│   └── routes/                ← Optimize edilmiş rotalar
├── setup_project.py           ← Bu script
├── config/settings.yaml
└── .gitignore
""")
    print("Sonraki adım: CSV dosyalarını data/raw/ klasörüne koy.")
    print("=" * 55)


def main():
    print("=" * 55)
    print("  Akıllı Lojistik - Proje Kurulumu Başlıyor")
    print("=" * 55 + "\n")

    create_directories()
    create_init_files()
    create_placeholder_files()
    create_gitignore()
    print_tree()


if __name__ == "__main__":
    main()
