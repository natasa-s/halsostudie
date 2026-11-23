"""Verktyg för analys av hälsostudiens dataset."""

import pandas as pd

class HealthAnalyzer:
    """
    Samlar analyser och grafer för hälsostudiens dataset.

    Parametrar:
        csv_path: sökväg till CSV-filen med datan.
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)