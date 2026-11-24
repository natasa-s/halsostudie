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

    def check_data(self):
        """Skriver ut kontroll av saknade värden, dubletter och kategorier."""
        print(f"Antal rader och kolumner: {self.df.shape}")
        print(f"Saknade värden: {self.df.isna().sum().sum()}")  # why 2 sum?
        print(f"Antal dubbletter av id: {self.df["id"].duplicated().sum()}")
        print(f"Antal dubbletter av rader (exklusive id): {self.df.drop(columns="id").duplicated().sum()}")
        for column in ["sex", "smoker", "disease"]:
            print(f"Värden i {column}: {sorted(self.df[column].unique().tolist())}")

    def add_bmi(self):
        """Lägger till kolumnen bmi (vikt i kg / längd i meter i kvadrat)."""
        self.df["bmi"] = self.df["weight"] / (self.df["height"] / 100) ** 2

    def add_smoker_yes(self):
        """Lägger till kolumnen smoker_yes (1 = rökare, 0 = icke-rökare)."""
        self.df["smoker_yes"] = (self.df["smoker"] == "Yes").astype(int)

    def add_sex_male(self):
        """Lägger till kolumnen sex_male (1 = man, 0 = kvinna)."""
        self.df["sex_male"] = (self.df["sex"] == "M").astype(int)