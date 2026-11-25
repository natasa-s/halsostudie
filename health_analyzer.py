"""Verktyg för analys av hälsostudiens dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

class HealthAnalyzer:
    """
    Samlar analyser och grafer för hälsostudiens dataset.

    Parametrar:
        csv_path: sökväg till CSV-filen med datan.
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    # --- Dataförberedelse ---

    def check_data(self):
        """Skriver ut kontroll av saknade värden, dubletter och kategorier."""
        print(f"Antal rader och kolumner: {self.df.shape}")
        print(f"Saknade värden: {self.df.isna().sum().sum()}")
        print(f"Antal dubbletter av id: {self.df["id"].duplicated().sum()}")
        print(f"Antal dubbletter av rader (exklusive id): {self.df.drop(columns="id").duplicated().sum()}")
        for column in ["sex", "smoker", "disease"]:
            print(f"Värden i {column}: {sorted(self.df[column].unique().tolist())}")

    def add_bmi(self):
        """
        Lägger till kolumnen bmi (vikt i kg / längd i meter i kvadrat).
        """
        self.df["bmi"] = self.df["weight"] / (self.df["height"] / 100) ** 2

    def add_smoker_yes(self):
        """
        Lägger till kolumnen smoker_yes (1 = rökare, 0 = icke-rökare).
        """
        self.df["smoker_yes"] = (self.df["smoker"] == "Yes").astype(int)

    def add_sex_male(self):
        """
        Lägger till kolumnen sex_male (1 = man, 0 = kvinna).
        """
        self.df["sex_male"] = (self.df["sex"] == "M").astype(int)

    # --- Analyser från del 1 ---

    def confidence_interval_normal(self, column="systolic_bp", z=1.96):
        """
        Beräknar ett 95 %-igt konfidensintervall för medelvärdet med normalapproximation.
        Returnerar tuple med nedre och övre gräns.
        """
        values = self.df[column].to_numpy()
        standard_error = values.std(ddof=1) / np.sqrt(len(values))
        return values.mean() - z * standard_error, values.mean() + z * standard_error

    def confidence_interval_bootstrap(self, column="systolic_bp", n_bootstrap=10_000):
        """
        Beräknar ett 95 %-igt konfidensintervall för medelvärdet bed bootstrap.
        Returnerar tuple med nedre och övre gräns (percentilerna 2,5 och 97,5).
        """
        values = self.df[column].to_numpy()
        bootstrap_means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            bootstrap_means[i] = np.random.choice(values, size=len(values), replace=True).mean()
        lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
        return lower, upper

    def welch_test(self, target="systolic_bp"):
        """
        Ensidigt Welch t-test av H1: rökare har högre medelvärde än icke-rökare.
        Returnerar dictionary med difference, standard_error, t, dof och p_value.
        """
        smokers = self.df.loc[self.df["smoker"] == "Yes", target].to_numpy()
        nonsmokers = self.df.loc[self.df["smoker"] == "No", target].to_numpy()

        var_smoker, var_nonsmoker = smokers.var(ddof=1), nonsmokers.var(ddof=1)
        n_smoker, n_nonsmoker = len(smokers), len(nonsmokers)

        difference = smokers.mean() - nonsmokers.mean()
        standard_error = np.sqrt(var_smoker / n_smoker + var_nonsmoker / n_nonsmoker)
        t = difference / standard_error

        # Welch-approximation av frihetsgraderna
        dof = (var_smoker / n_smoker + var_nonsmoker / n_nonsmoker) ** 2 / (
            (var_smoker / n_smoker) ** 2 / (n_smoker - 1)
            + (var_nonsmoker / n_nonsmoker) ** 2 / (n_nonsmoker - 1)
        )
        p_value = stats.t.sf(t, dof)

        return {"difference": difference, "standard_error": standard_error, "t": t, "dof": dof, "p_value": p_value}

    # --- Regression ---

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

    def bootstrap_coefficients(self, predictors, target="systolic_bp", n_bootstrap=2000):
        """
        Beräknar 95 %-iga bootstrapintervall för regressionskoefficienterna.
        Returnerar DataFrame med koefficient, nedre och övre gräns för varje variabel.
        """
        X = self.df[predictors].to_numpy()
        y = self.df[target].to_numpy()
        n = len(y)

        boot_coefficients = np.empty((n_bootstrap, len(predictors)))
        for i in range(n_bootstrap):
            rows = np.random.choice(n, size=n, replace=True)
            boot_coefficients[i] = LinearRegression().fit(X[rows], y[rows]).coef_

        coefficients = LinearRegression().fit(X, y).coef_
        lower, upper = np.percentile(boot_coefficients, [2.5, 97.5], axis=0)

        return pd.DataFrame(
            {"Koefficient": coefficients, "Nedre gräns": lower, "Övre gräns": upper},
            index=predictors
        )

    # --- Grafer ---

    def plot_fit(self, predictor, result, target="systolic_bp"):
        """
        Ritar datan som punter med den anpassade regressionslinjen.
        """
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
        """
        Ritar residualer mot anpassade värden för att kontrollera modellen.
        """
        fig, ax = plt.subplots()
        ax.scatter(result["y_hat"], result["residuals"], alpha=0.5)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("Residualdiagram")
        ax.set_xlabel("Anpassat värde (mmHg)")
        ax.set_ylabel("Residual (mmHg)")
        plt.show()

    def plot_coefficients(self, table, labels=None):
        """
        Ritar koefficienterna med konfidensintervall och en linje vid noll.
        """
        positions = np.arange(len(table))
        lower_errors = (table["Koefficient"] - table["Nedre gräns"]).to_numpy()
        upper_errors = (table["Övre gräns"] - table["Koefficient"]).to_numpy()

        fig, ax = plt.subplots()
        ax.errorbar(table["Koefficient"].to_numpy(), positions, xerr=[lower_errors, upper_errors], fmt="o", capsize=6)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels if labels is not None else table.index)
        ax.invert_yaxis()
        ax.set_title("Regressionskoefficienter med 95 %-iga konfidensintervall")
        ax.set_xlabel("Förändring i systoliskt blodtryck (mmHg) per enhet")
        plt.show()