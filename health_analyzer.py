"""Verktyg för analys av hälsostudiens dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
        print(f"Saknade värden: {self.df.isna().sum().sum()}")
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

    def fit_linear_regression(self, predictors, target="systolic_bp"):
        """
        Anpassar en linjär regression med scikit-learn.
        Returnerar dictionary med intercept, coefficients, y_hat, residuals och r2.
        """
        X = self.df[predictors].to_numpy()
        y = self.df[target].to_numpy()

        model = LinearRegression().fit(X, y)
        y_hat = model.predict(X)
        residuals = y - y_hat
        r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)

        return {
            "intercept": model.intercept_,
            "coefficients": dict(zip(predictors, model.coef_)),
            "y_hat": y_hat,
            "residuals": residuals,
            "r2": r2
        }

    def plot_fit(self, predictor, result, target="systolic_bp"):
        """Ritar datan som punter med den anpassade regressionslinjen."""
        x = self.df[predictor].to_numpy()
        order = np.argsort(x)

        fig, ax = plt.subplots()
        ax.scatter(x, self.df[target], alpha=0.5)
        ax.plot(x[order], result["y_hat"][order], color="red", label="Regressionslinje")
        ax.set_title("Systoliskt blodtryck mot ålder med regressionslinje")
        ax.set_xlabel("Ålder (år)")
        ax.set_ylabel("Systoliskt blodtryck (mmHg)")
        ax.legend()
        plt.show()

    def plot_residuals(self, result):
        """Ritar residualer mot anpassade värden för att kontrollera modellen."""
        fig, ax = plt.subplots()
        ax.scatter(result["y_hat"], result["residuals"], alpha=0.5)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("Residualdiagram")
        ax.set_xlabel("Anpassat värde (mmHg)")
        ax.set_ylabel("Residual (mmHg)")
        plt.show()