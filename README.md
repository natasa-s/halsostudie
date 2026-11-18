# Individuell uppgift – Hälsostudie

Analys av data från en hälsostudie (800 deltagare) i Python och Jupyter Notebook.

## Python-version
Python 3.14.10

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Köra notebooken
Starta Jupyter och öppna notebooken (eller öppna den i VS Code):
```bash
jupyter notebook report_del1.ipynb
```
Kör **Restart & Run All** för att reproducera alla resultat. Datan läses med relativ sökväg (`data/health_study_dataset.csv`) och slumpfröet är satt till 42.

## Innehåll
- `data/health_study_dataset.csv` – datasetet
- `report_del1.ipynb` – Del 1: Grundläggande analys och statistik (branch `del1`)
- `requirements.txt` – nödvändiga paket