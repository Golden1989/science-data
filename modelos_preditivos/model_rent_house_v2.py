"""
TAREFA 1 (versão 2) - Tentativa de melhorar o modelo de aluguel, ainda com regressão linear.

Reaproveita a carga e a limpeza de 'model_rent_house.py' (sem duplicar código).
A versão original fica intacta para comparação.

Ideias testadas, todas com a variável-alvo em reais (sem log no alvo):
    1. LinearRegression (base, igual à versão original)
    2. LinearRegression com log das variáveis de área, condomínio e IPTU
       (log só nas explicativas, para tratar o efeito multiplicativo da área)
    3. HuberRegressor: regressão linear robusta, menos sensível aos aluguéis extremos
    4. Polinomial de grau 2 + Ridge, com log nas explicativas
       (captura curvaturas que a reta não consegue)

Regra de escolha: menor RMSE na validação cruzada, calculada só com o treino.
O conjunto de teste é usado apenas para o relatório final.

Saídas (na mesma pasta do script):
    best_rent_model_v2.joblib
    metrics_aluguel_v2.csv

Uso:
    python model_rent_house_v2.py

Descobertas do modelo v2 (polinomial + Ridge):
- Melhora real no teste: R² subiu de 0,501 para 0,567. O erro médio caiu de R$ 1.687 para R$ 1.512 (−10%), e o RMSE de R$ 2.416 para R$ 2.252.
- O problema estava nas entradas extremas: só aplicar log nas explicativas (área, condomínio e IPTU) já derrubou o RMSE da validação cruzada de R$ 5.695 para R$ 2.353. A validação ficou bem mais estável, sem a instabilidade de antes.
- Um pouco de não linearidade ajudou: o polinômio de grau 2 melhorou o R² em mais 0,026 em relação ao modelo só com log, o que confirma que a relação entre área e preço não é uma reta perfeita.
- Robustez no erro não resolveu: o Huber, que tenta ignorar os extremos na função de custo, ficou com R² de 0,467, pior que a versão base. O problema estava mais nas escalas das entradas do que nos poucos casos extremos.
- A regra de escolha funcionou: o modelo escolhido pela validação cruzada também foi o melhor no teste. Ou seja, não precisamos olhar o teste para decidir.

Pontos de atenção:
- O ganho é modesto: o modelo explica cerca de 57% da variação do aluguel. Melhorou, mas continua longe de um modelo forte.
- O bairro continua faltando: é o limite mais importante, e nenhuma transformação resolve isso sem a informação.
- Interpretação mais difícil: com termos polinomiais, os coeficientes não se leem como "quanto cada variável soma no aluguel". Por isso o script não imprime a tabela de coeficientes.
- Um só corte dos dados: não testamos com outras divisões aleatórias. Para afirmar que a melhora é estável, seria preciso repetir o experimento com outras sementes.

Resumo: a versão 2 é melhor que a original nos três números (R², MAE e RMSE), e a diferença veio principalmente de tratar as variáveis com cauda longa.
O polinômio ajudou um pouco a mais. O maior limite continua sendo a falta do bairro, e isso precisa ser dito na apresentação.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

from model_rent_house import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    load_data,
    print_title,
    regression_metrics,
)

# Variáveis com cauda longa (muitos valores pequenos e alguns muito grandes).
LOG_FEATURES = ["area", "hoa", "property tax"]
PLAIN_NUMERIC = [c for c in NUMERIC_FEATURES if c not in LOG_FEATURES]


# --------------------------------------------------------------------------- #
# Pré-processamento
# --------------------------------------------------------------------------- #
def build_preprocessor(log_features: list[str]) -> ColumnTransformer:
    """Monta o pré-processamento.

    Variáveis em 'log_features' passam por log1p antes de escalonar.
    As demais numéricas só são imputadas e escalonadas.
    As categóricas viram One-Hot.
    """
    transformers = []
    if log_features:
        transformers.append(("log", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), log_features))
    plain = [c for c in NUMERIC_FEATURES if c not in log_features]
    transformers.append(("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), plain))
    transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
                         CATEGORICAL_FEATURES))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def build_candidates() -> dict:
    """Cada candidato é (pipeline, grade de hiperparâmetros)."""
    base = build_preprocessor([])
    logged = build_preprocessor(LOG_FEATURES)
    return {
        "LinearRegression (base)": (
            Pipeline([("prep", base), ("model", LinearRegression())]),
            {}),
        "LinearRegression + log": (
            Pipeline([("prep", logged), ("model", LinearRegression())]),
            {}),
        "Huber (robusta)": (
            Pipeline([("prep", base), ("model", HuberRegressor(max_iter=2000))]),
            {"model__epsilon": [1.1, 1.35, 2.0, 3.0],
             "model__alpha": [1e-4, 1e-2, 1.0]}),
        "Polinomial grau 2 + Ridge": (
            Pipeline([("prep", logged),
                      ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                      ("model", Ridge())]),
            {"model__alpha": np.logspace(-2, 4, 13)}),
    }


# --------------------------------------------------------------------------- #
# Treino com validação cruzada (só com o treino)
# --------------------------------------------------------------------------- #
def tune(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print_title("Validação cruzada (5 folds, RMSE em R$)")
    searches = {}
    for name, (pipe, grid) in build_candidates().items():
        search = GridSearchCV(pipe, grid, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train)
        searches[name] = search
        params = {k.replace("model__", ""): (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in search.best_params_.items()}
        print(f"{name:<28} RMSE CV = {-search.best_score_:>12,.2f} | {params}")
    return searches


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    searches = tune(X_train, y_train)
    best_name = min(searches, key=lambda n: -searches[n].best_score_)
    best_model = searches[best_name].best_estimator_

    # Teste: só para relatório. A escolha já foi feita pela validação.
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    rows = {"Baseline (média)": regression_metrics(y_test, baseline.predict(X_test))}
    for name, search in searches.items():
        rows[name] = regression_metrics(y_test, search.predict(X_test))
    metrics = pd.DataFrame(rows).T

    print_title(f"Métricas no TESTE (escolhido pela validação: {best_name})")
    print(metrics.to_string(formatters={
        "R2": "{:+.4f}".format, "MAE": "{:,.2f}".format,
        "MSE": "{:,.2f}".format, "RMSE": "{:,.2f}".format}))

    joblib.dump(best_model, BASE_DIR / "best_rent_model_v2.joblib")
    metrics.to_csv(BASE_DIR / "metrics_aluguel_v2.csv")
    print("\nSalvo: best_rent_model_v2.joblib e metrics_aluguel_v2.csv")


if __name__ == "__main__":
    main()
