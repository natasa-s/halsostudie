# Individuell uppgift – Hälsostudie

Analys av data från en hälsostudie (800 deltagare) i Python och Jupyter Notebook.
Del 1 är en grundläggande statistisk analys. Del 2 fördjupar med linjär regression och samlar analyserna i klassen `HealthAnalyzer`.

## Python-version
Python 3.14.10

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Köra notebookarna
Starta Jupyter och öppna önskad notebook (eller öppna den i VS Code):
```bash
jupyter notebook
```
Kör **Restart & Run All** för att reproducera alla resultat. Datan läses med relativ sökväg (`data/health_study_dataset.csv`) och slumpfröet är satt till 42.

## Innehåll
- `data/health_study_dataset.csv` – datasetet
- `report_del1.ipynb` – Del 1: Grundläggande analys och statistik (branch `del1`)
- `report_del2.ipynb` – Del 2: Linjär regression där systoliskt blodtryck analyseras utifrån ålder, BMI, rökning, kön och kolesterol (branch `del2`)
- `health_analyzer.py` – klassen `HealthAnalyzer` (datakontroll, nya variabler, konfidensintervall, Welchs t-test, linjär regression, bootstrap av koefficienter, grafer)
- `requirements.txt` – nödvändiga paket