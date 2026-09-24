"""
Treinamento de modelos de Regressão Linear para prever o preço de imóveis.

Pipeline:
    1. Carrega e inspeciona 'House Price Prediction Dataset.csv'.
    2. Limpa dados (duplicatas, valores ausentes, valores inconsistentes).
    3. Cria novas features (idade do imóvel, razão banheiros/quartos, área por quarto).
    4. Pré-processa: escalonamento numérico, OrdinalEncoder em 'Condition',
       One-Hot Encoding em 'Location' e 'Garage'.
    5. Separa treino (80%) e teste (20%).
    6. Ajusta LinearRegression, Ridge, Lasso e ElasticNet com GridSearchCV (5 folds).
    7. Escolhe o melhor modelo pelo RMSE de validação cruzada e avalia no teste.
    8. Salva métricas, coeficientes, gráficos e o modelo (.joblib) na pasta do script.

Uso:
    python train_model.py
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # salva figuras sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = "House Price Prediction Dataset.csv"
# Procura o CSV na pasta do script e, se não achar, na pasta acima
DATA_PATH = next((p for p in (BASE_DIR / DATA_FILE, BASE_DIR.parent / DATA_FILE)
                  if p.exists()), BASE_DIR / DATA_FILE)
OUTPUT_DIR = BASE_DIR  # resultados ficam junto do script

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
REFERENCE_YEAR = 2026  # ano usado para calcular a idade do imóvel

TARGET = "Price"
NUMERIC_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors",
                    "HouseAge", "BathBedRatio", "AreaPerRoom"]
ORDINAL_FEATURES = ["Condition"]
CONDITION_ORDER = [["Poor", "Fair", "Good", "Excellent"]]
NOMINAL_FEATURES = ["Location", "Garage"]


# --------------------------------------------------------------------------- #
# 1. Carga, análise e limpeza
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    """Lê o CSV e exibe um resumo exploratório."""
    df = pd.read_csv(path)
    print("=" * 70)
    print(f"Dataset: {path.name}  |  {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("=" * 70)
    print("\n--- info() ---")
    df.info()
    print("\n--- describe() ---")
    print(df.describe(include="all").T.to_string())
    print("\n--- Valores nulos por coluna ---")
    print(df.isna().sum().to_string())
    print("\n--- Correlação (Pearson) das variáveis numéricas com o preço ---")
    print(df.select_dtypes("number").drop(columns="Id").corr()[TARGET].round(3).to_string())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas, linhas inválidas e imputa ausentes."""
    df = df.drop(columns="Id").copy()
    n0 = len(df)

    df = df.drop_duplicates()

    # Padroniza texto das categóricas (espaços / capitalização)
    for col in ORDINAL_FEATURES + NOMINAL_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.title()

    # Regras de consistência de domínio
    valid = (
        (df["Area"] > 0)
        & (df["Bedrooms"] >= 0)
        & (df["Bathrooms"] >= 0)
        & (df["Floors"] >= 1)
        & (df["YearBuilt"].between(1800, REFERENCE_YEAR))
        & (df[TARGET] > 0)
        & (df["Condition"].isin(CONDITION_ORDER[0]))
    )
    df = df[valid]

    # Imputação (o dataset atual não tem nulos, mas o script fica robusto)
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = ORDINAL_FEATURES + NOMINAL_FEATURES
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    print(f"\nLimpeza: {n0} -> {len(df)} linhas ({n0 - len(df)} removidas)")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Engenharia de recursos e pré-processamento
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas.

    'HouseAge' substitui 'YearBuilt' (são colinearmente perfeitas; manter as
    duas só adicionaria redundância ao modelo linear).
    """
    df = df.copy()
    df["HouseAge"] = REFERENCE_YEAR - df["YearBuilt"]
    df["BathBedRatio"] = df["Bathrooms"] / df["Bedrooms"].clip(lower=1)
    df["AreaPerRoom"] = df["Area"] / (df["Bedrooms"] + df["Bathrooms"]).clip(lower=1)
    return df.drop(columns="YearBuilt")


def build_preprocessor() -> ColumnTransformer:
    """Escalonamento numérico + encoding das categóricas."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("ord", Pipeline([
                ("enc", OrdinalEncoder(categories=CONDITION_ORDER)),
                ("scale", StandardScaler()),
            ]), ORDINAL_FEATURES),
            ("nom", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- #
# 3. Modelagem e otimização
# --------------------------------------------------------------------------- #
def model_search_spaces() -> dict:
    """Modelos candidatos e suas grades de hiperparâmetros.

    O tipo de scaler (Standard vs Robust) também é tratado como hiperparâmetro.
    """
    scalers = [StandardScaler(), RobustScaler()]
    alphas = np.logspace(-3, 4, 30)
    return {
        "LinearRegression": (LinearRegression(), {
            "prep__num": scalers,
        }),
        "Ridge": (Ridge(random_state=RANDOM_STATE), {
            "prep__num": scalers,
            "model__alpha": alphas,
        }),
        "Lasso": (Lasso(max_iter=50_000, random_state=RANDOM_STATE), {
            "prep__num": scalers,
            "model__alpha": np.logspace(-1, 5, 30),
        }),
        "ElasticNet": (ElasticNet(max_iter=50_000, random_state=RANDOM_STATE), {
            "prep__num": scalers,
            "model__alpha": np.logspace(-3, 3, 20),
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        }),
    }


def tune_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Roda GridSearchCV para cada modelo e devolve os buscadores ajustados."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    print("\n" + "=" * 70)
    print(f"Validação cruzada ({CV_FOLDS} folds) - métrica: RMSE")
    print("=" * 70)
    for name, (estimator, grid) in model_search_spaces().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        search = GridSearchCV(
            pipe, grid, cv=cv, n_jobs=-1,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            refit="rmse",
        )
        search.fit(X_train, y_train)
        best = search.best_index_
        cv_rmse = -search.cv_results_["mean_test_rmse"][best]
        cv_r2 = search.cv_results_["mean_test_r2"][best]
        params = {k.replace("model__", "").replace("prep__num", "scaler"):
                  (round(float(v), 4) if isinstance(v, (int, float)) else type(v).__name__)
                  for k, v in search.best_params_.items()}
        print(f"{name:<17} CV RMSE = {cv_rmse:>11,.0f} | CV R² = {cv_r2:+.4f} | {params}")
        results[name] = search
    return results


# --------------------------------------------------------------------------- #
# 4. Avaliação
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
    }


def get_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Coeficientes do modelo, ordenados por impacto absoluto."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_
    return (pd.DataFrame({"feature": names, "coef": coefs})
            .assign(abs_coef=lambda d: d["coef"].abs())
            .sort_values("abs_coef", ascending=False)
            .drop(columns="abs_coef")
            .reset_index(drop=True))


def plot_results(y_test, y_pred, model_name: str, out_dir: Path) -> None:
    """Salva gráficos de Previsto vs Real e de resíduos."""
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolor="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Previsão perfeita (y = x)")
    ax.set_xlabel("Preço real")
    ax.set_ylabel("Preço previsto")
    ax.set_title(f"Previsões vs Valores Reais - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_real.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolor="none")
    axes[0].axhline(0, color="r", linestyle="--")
    axes[0].set_xlabel("Preço previsto")
    axes[0].set_ylabel("Resíduo (real - previsto)")
    axes[0].set_title("Resíduos vs Previsões")
    axes[1].hist(residuals, bins=40, edgecolor="white")
    axes[1].set_xlabel("Resíduo")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title("Distribuição dos resíduos")
    fig.suptitle(f"Análise de resíduos - {model_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = engineer_features(clean_data(load_data(DATA_PATH)))
    X = df[NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"\nTreino: {len(X_train)} | Teste: {len(X_test)}")

    searches = tune_models(X_train, y_train)

    # Melhor modelo escolhido pelo CV (não pelo teste, para evitar vazamento)
    best_name = min(searches, key=lambda n: -searches[n].best_score_)
    best_model = searches[best_name].best_estimator_

    # Avaliação no teste: todos os modelos + baseline que prevê a média
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    rows = {"Baseline (média)": regression_metrics(y_test, baseline.predict(X_test))}
    for name, search in searches.items():
        rows[name] = regression_metrics(y_test, search.predict(X_test))
    metrics = pd.DataFrame(rows).T

    print("\n" + "=" * 70)
    print(f"Métricas no conjunto de TESTE  (melhor pelo CV: {best_name})")
    print("=" * 70)
    print(metrics.to_string(formatters={
        "R2": "{:+.4f}".format, "MAE": "{:,.0f}".format,
        "MSE": "{:,.0f}".format, "RMSE": "{:,.0f}".format}))

    coefs = get_coefficients(best_model)
    print(f"\n--- Coeficientes do {best_name} (escala padronizada, ordem de impacto) ---")
    print(coefs.to_string(index=False, formatters={"coef": "{:+,.0f}".format}))

    y_pred = best_model.predict(X_test)
    plot_results(y_test, y_pred, best_name, OUTPUT_DIR)

    metrics.to_csv(OUTPUT_DIR / "metrics.csv")
    coefs.to_csv(OUTPUT_DIR / "coefficients.csv", index=False)
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")
    print(f"\nArquivos salvos em: {OUTPUT_DIR}")
    print("  best_model.joblib, metrics.csv, coefficients.csv, pred_vs_real.png, residuals.png")


if __name__ == "__main__":
    main()
