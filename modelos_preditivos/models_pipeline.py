"""
Pipeline com duas tarefas de modelagem preditiva.

TAREFA 1 - Regressão linear para o valor do aluguel
    Dataset: houses_to_rent.csv   |   Alvo: 'rent amount' (em R$)
    Modelos candidatos (escolhidos por validação cruzada de 5 folds):
        - LinearRegression
        - LinearRegression com log das explicativas (área, condomínio, IPTU)
        - Huber (regressão linear robusta)
        - Polinomial de grau 2 + Ridge, com log das explicativas

TAREFA 2 - Regressão logística para diagnóstico de câncer de mama
    Dataset: BreastCancer.csv   |   Alvo: 'Class' (0 = benigno, 1 = maligno)

Tratamento da coluna 'floor' (aluguel): veja o bloco de comentário sobre FLOOR_MODE no código.

Saídas (na mesma pasta do script):
    best_rent_model.joblib, metrics_aluguel.csv, coefficients_aluguel.csv
    best_cancer_model.joblib, metrics_cancer.csv

Uso:
    python models_pipeline.py                 (rodada normal, opção padrão "flag")
    python models_pipeline.py onehot          (rodada normal, alternativa "onehot")
    python models_pipeline.py estabilidade    (repete os experimentos com 10 sementes,
                                               sem gravar arquivos)
"""

import contextlib
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, HuberRegressor, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
# ---------------------------------------------------------------------------
# Tratamento da coluna 'floor' (aluguel). Há duas opções, e as duas estão prontas.
#
# "flag" (PADRÃO, usada na entrega):
#   - 'floor' continua numérica (1, 2, 3...).
#   - Uma coluna extra, 'floor_missing', vale 1 quando o valor original é '-'
#     (casa sem andar). Para essas casas, 'floor' recebe a mediana, e o
#     indicador deixa claro que aquele valor não é um andar de verdade.
#   - Poucas colunas (+1) e coeficientes fáceis de explicar.
#
# "onehot" (alternativa):
#   - 'floor' vira categoria: cada andar é uma coluna 0/1 (cerca de 36 colunas).
#   - '-' é uma categoria própria e é a referência (drop="first"), então os
#     coeficientes dos andares são comparados com "casa sem andar".
#   - É mais fiel ao enunciado (One-Hot), mas gera muitas colunas, algumas com
#     poucos exemplos, e o polinômio de grau 2 fica bem maior.
#
# Resultados nos testes (mesma divisão, seed 42):
#   flag:   CV RMSE R$ 2.260,48 | teste R² 0,570 | MAE R$ 1.512 | RMSE R$ 2.244
#   onehot: CV RMSE R$ 2.274,04 | teste R² 0,570 | MAE R$ 1.501 | RMSE R$ 2.242
# A escolha de 'flag' vem da validação cruzada (o critério do script), não do teste.
#
# Para rodar a alternativa sem editar o código:
#     python models_pipeline.py onehot
# As saídas da rodada são sobrescritas. Para voltar à opção padrão:
#     python models_pipeline.py flag
# ---------------------------------------------------------------------------
FLOOR_MODE = "flag"

RENT_TARGET = "rent amount"
RENT_MONEY_COLS = ["hoa", "rent amount", "property tax"]
RENT_LOG_COLS = ["area", "hoa", "property tax"]  # cauda longa: log1p antes de escalonar

BREAST_TARGET = "Class"
BREAST_FEATURES = ["Cl.thickness", "Cell.size", "Cell.shape", "Marg.adhesion",
                   "Epith.c.size", "Bare.nuclei", "Bl.cromatin",
                   "Normal.nucleoli", "Mitoses"]
BREAST_CLASS_NAMES = {0: "Benigno", 1: "Maligno"}


def find_csv(name: str) -> Path:
    """Procura o CSV na pasta do script e, se não achar, na pasta acima."""
    for path in (BASE_DIR / name, BASE_DIR.parent / name):
        if path.exists():
            return path
    raise FileNotFoundError(f"'{name}' não encontrado em {BASE_DIR} nem na pasta acima.")


def print_title(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ========================================================================== #
# TAREFA 1 - Aluguel
# ========================================================================== #
# ---------------------------------------------------------------------------
# DESCOBERTAS - TAREFA 1 (aluguel)
#
# 1. Vazamento de dados: no começo, o modelo tinha R² ≈ 0,96, o que parecia ótimo.
#    Mas 'fire insurance' é ≈ 1,3% do aluguel (correlação 0,986) e 'total' é a soma
#    do aluguel com as taxas. O modelo estava "adivinhando" o alvo a partir de uma
#    conta feita com o próprio alvo. Ao removê-las, o R² caiu para ≈ 0,50 (valor real).
#
# 2. Log no alvo piorou o modelo: o R² da LinearRegression caiu de 0,50 para 0,29, e o
#    RMSE da validação saiu como 2 × 10^17, porque o exp de um log muito alto estourou.
#    O erro em log vira erro multiplicativo ao voltar para reais, e pesa nos imóveis
#    caros. Como a métrica é RMSE em reais, o alvo ficou em reais.
#
# 3. Log nas explicativas (área, condomínio, IPTU) funcionou: o RMSE da validação caiu
#    de R$ 5.695 para R$ 2.353. O problema estava nas entradas de cauda longa.
#
# 4. Um pouco de não linearidade ajudou: o polinômio de grau 2 melhorou o R² em ≈ 0,026
#    em relação ao modelo só com log. A relação área × preço não é uma reta perfeita.
#
# 5. Huber (robusta a extremos) não ajudou: R² de 0,47, pior que a regressão base.
#
# 6. Limites que não se resolvem com transformações:
#    - Aluguel com cauda longa: 90% dos imóveis custam até ~R$ 9.900, mas há casos de
#      R$ 45.000. Os extremos continuam pesando no RMSE.
#    - Falta o bairro: 'city' é só um código numérico sem significado documentado.
#      O bairro é um dos fatores mais fortes no preço de aluguel.
#    - Vazios: 'floor' tem '-' em 1.555 linhas e 'hoa' tem 202 vazios. Tratados com a
#      mediana (e com o indicador 'floor_missing', na opção padrão).
#
# 7. Estabilidade (10 sementes diferentes para a divisão e a validação cruzada):
#    - O polinômio com Ridge foi escolhido em todas as 10 e ficou sempre à frente da
#      regressão base. R² médio de 0,589 ± 0,028 (de 0,535 a 0,629).
#    - A regressão base é instável: em duas sementes o RMSE foi de R$ 13.450 e R$ 9.518,
#      e o R² médio ficou negativo. O R² de 0,501 da semente 42 não representa bem essa
#      regressão. A diferença real entre os modelos é maior do que a da semente 42 mostra.
#
# Resultado final (semente 42, teste): R² 0,570 | MAE R$ 1.512 | RMSE R$ 2.244.
# A regressão linear simples chegou a R² 0,501 | RMSE R$ 2.416 (nessa semente).
# R² ≈ 0,57 significa que o modelo explica cerca de 57% da variação do aluguel.
# ---------------------------------------------------------------------------


def parse_brl(series: pd.Series) -> pd.Series:
    """Converte textos no formato 'R$8,000' para número. Valores inválidos viram NaN."""
    cleaned = (series.astype(str)
               .str.replace("R$", "", regex=False)
               .str.replace(",", "", regex=False)
               .str.strip())
    return pd.to_numeric(cleaned, errors="coerce")


def load_rent(floor_mode: str) -> pd.DataFrame:
    """Carrega e limpa o dataset de aluguel, tratando 'floor' conforme o modo escolhido."""
    if floor_mode not in ("flag", "onehot"):
        raise ValueError(f"floor_mode inválido: {floor_mode}")

    df = pd.read_csv(find_csv("houses_to_rent.csv"))
    print_title("TAREFA 1 - Dataset de aluguel")
    print(f"Shape original: {df.shape} | tratamento de 'floor': {floor_mode}")

    # 'total' é a soma do aluguel com as taxas, e 'fire insurance' é ~1,3% do aluguel
    # (correlação 0,986). Ambas são calculadas a partir do alvo: usá-las é vazamento.
    df = df.drop(columns=["Unnamed: 0", "total", "fire insurance"])

    for col in RENT_MONEY_COLS:
        df[col] = parse_brl(df[col])

    # 'city' são códigos numéricos sem significado documentado: tratamos como categoria.
    df["city"] = df["city"].astype(str)

    # 'floor' tem '-' em 1.555 casas, que provavelmente são casas sem andar.
    floor_text = df["floor"].astype(str).str.strip()
    if floor_mode == "flag":
        df["floor_missing"] = (floor_text == "-").astype(int)
        df["floor"] = pd.to_numeric(floor_text, errors="coerce")
    else:
        df["floor"] = floor_text

    before = len(df)
    df = df.dropna(subset=[RENT_TARGET]).reset_index(drop=True)
    print(f"Linhas sem alvo removidas: {before - len(df)}")
    return df


def rent_feature_groups(floor_mode: str) -> tuple[list, list, list]:
    """Devolve (colunas com log, numéricas, categóricas) para o modo escolhido."""
    numeric = ["rooms", "bathroom", "parking spaces"]
    categorical = ["city", "animal", "furniture"]
    if floor_mode == "flag":
        numeric += ["floor", "floor_missing"]
    else:
        categorical += ["floor"]
    return RENT_LOG_COLS, numeric, categorical


def build_rent_preprocessor(floor_mode: str, use_log: bool) -> ColumnTransformer:
    """Imputação, log (opcional), escalonamento e One-Hot.

    Fica dentro do Pipeline para que medianas e médias sejam calculadas só com o treino
    de cada rodada da validação cruzada.
    """
    log_cols, numeric_cols, categorical_cols = rent_feature_groups(floor_mode)
    transformers = []
    if use_log:
        transformers.append(("log", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), log_cols))
    else:
        numeric_cols = log_cols + numeric_cols
    transformers.append(("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_cols))
    transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
                         categorical_cols))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def build_rent_candidates(floor_mode: str) -> dict:
    """Cada candidato é (pipeline, grade de hiperparâmetros)."""
    base = build_rent_preprocessor(floor_mode, use_log=False)
    logged = build_rent_preprocessor(floor_mode, use_log=True)
    return {
        "LinearRegression": (
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


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
    }


def get_rent_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Coeficientes do modelo final, ordenados pelo impacto absoluto.

    Com etapa polinomial, os nomes incluem termos combinados (ex.: 'area^2').
    Os valores estão na escala padronizada, então comparam-se pela ordem de grandeza.
    """
    names = pipeline.named_steps["prep"].get_feature_names_out()
    if "poly" in pipeline.named_steps:
        names = pipeline.named_steps["poly"].get_feature_names_out(names)
    coefs = pipeline.named_steps["model"].coef_
    return (pd.DataFrame({"feature": names, "coef": coefs})
            .assign(abs_coef=lambda d: d["coef"].abs())
            .sort_values("abs_coef", ascending=False)
            .drop(columns="abs_coef")
            .reset_index(drop=True))


def run_rent(floor_mode: str, save: bool) -> dict:
    """Treina os candidatos, escolhe pela validação e avalia no teste.

    Devolve um resumo com o nome do modelo escolhido e as métricas de todos.
    """
    df = load_rent(floor_mode)
    X = df.drop(columns=[RENT_TARGET])
    y = df[RENT_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    searches = {}
    print_title(f"TAREFA 1 - Validação cruzada (5 folds, RMSE em R$) | floor = {floor_mode}")
    for name, (pipe, grid) in build_rent_candidates(floor_mode).items():
        search = GridSearchCV(pipe, grid, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train)
        searches[name] = search
        params = {k.replace("model__", ""): (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in search.best_params_.items()}
        print(f"{name:<28} RMSE CV = {-search.best_score_:>12,.2f} | {params}")

    # A escolha é feita só pela validação. O teste é usado apenas para o relatório.
    best_name = min(searches, key=lambda n: -searches[n].best_score_)
    best_model = searches[best_name].best_estimator_

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    rows = {"Baseline (média)": regression_metrics(y_test, baseline.predict(X_test))}
    for name, search in searches.items():
        rows[name] = regression_metrics(y_test, search.predict(X_test))
    metrics = pd.DataFrame(rows).T

    print_title(f"TAREFA 1 - Métricas no TESTE (escolhido pela validação: {best_name})")
    print(metrics.to_string(formatters={
        "R2": "{:+.4f}".format, "MAE": "{:,.2f}".format,
        "MSE": "{:,.2f}".format, "RMSE": "{:,.2f}".format}))

    coefs = get_rent_coefficients(best_model)
    print_title(f"TAREFA 1 - Coeficientes mais influentes ({best_name})")
    print(coefs.head(12).to_string(index=False, formatters={"coef": "{:+,.2f}".format}))

    if save:
        joblib.dump(best_model, BASE_DIR / "best_rent_model.joblib")
        # utf-8-sig: o BOM faz o Excel do Windows ler os acentos corretamente
        metrics.to_csv(BASE_DIR / "metrics_aluguel.csv", encoding="utf-8-sig")
        coefs.to_csv(BASE_DIR / "coefficients_aluguel.csv", index=False, encoding="utf-8-sig")
        print("\nSalvo: best_rent_model.joblib, metrics_aluguel.csv e coefficients_aluguel.csv")

    return {"best": best_name, "metrics": metrics, "searches": searches}


# ========================================================================== #
# TAREFA 2 - Câncer de mama
# ========================================================================== #
# ---------------------------------------------------------------------------
# DESCOBERTAS - TAREFA 2 (câncer de mama)
#
# 1. Desempenho forte no teste (140 casos): acurácia 0,957, precisão, recall e F1
#    de 0,938 e AUC-ROC de 0,995. Chutar sempre "benigno" acertaria só 65,7%, então
#    o modelo aprendeu com as variáveis.
#
# 2. Os erros mais graves são os falsos negativos: 3 malignos foram classificados como
#    benignos (de 48 malignos no teste). Num diagnóstico, isso significa um paciente que
#    deixaria de ser investigado. Os 3 falsos positivos gerariam só exames extras.
#    Por isso o recall é a métrica que mais importa aqui.
#
# 3. Variáveis mais influentes (coeficientes padronizados, todos positivos):
#    Bare.nuclei (+1,39), Cl.thickness (+1,01) e Bl.cromatin (+0,74).
#    Valor positivo empurra a previsão para Maligno. Faz sentido: células mais alteradas
#    aumentam a chance de malignidade.
#
# 4. Sem vazamento de dados: nenhuma variável é calculada a partir do alvo. O 'Id' é só
#    um identificador e foi removido. Os 16 vazios de 'Bare.nuclei' foram preenchidos
#    com a mediana dentro do Pipeline, usando só dados de treino.
#
# 5. Por que funcionou melhor que o aluguel: as variáveis de célula têm relação forte e
#    clara com o diagnóstico, e o alvo é razoavelmente equilibrado (65% benignos,
#    35% malignos), sem os extremos do aluguel.
#
# Pontos de atenção:
#    - Amostra pequena: 3 erros em 48 malignos no teste é ≈ 6%, com intervalo de confiança
#      grande. Com 10 sementes diferentes, o resultado é estável: acurácia 0,957 ± 0,014,
#      recall 0,938 ± 0,017, AUC 0,993 ± 0,004, e de 2 a 4 falsos negativos por divisão.
#    - Rótulos inferidos: o arquivo não diz o que é 0 e 1. Assumimos 0 = benigno e
#      1 = maligno pela contagem (458 e 241). Vale confirmar com a fonte.
#    - Correlação não é causa: os coeficientes mostram associação, não causalidade.
#    - Não é um diagnóstico clínico: é um exercício com um dataset histórico.
#
# Melhoria possível (não testada): ajustar o limiar de decisão para aumentar o recall,
# aceitando mais falsos positivos em troca de detectar mais malignos.
# ---------------------------------------------------------------------------


def load_breast() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega o dataset, remove o Id e garante o alvo binário 0/1."""
    df = pd.read_csv(find_csv("BreastCancer.csv")).drop(columns="Id")
    print_title("TAREFA 2 - Dataset de câncer de mama")
    print(f"Shape: {df.shape}")

    y = df[BREAST_TARGET]
    if set(y.unique()) == {2, 4}:
        y = y.map({2: 0, 4: 1})
    assert set(y.unique()) == {0, 1}, f"Alvo não binário: {y.unique()}"

    print("Distribuição do alvo:")
    for code, n in y.value_counts().sort_index().items():
        print(f"  {code} ({BREAST_CLASS_NAMES[code]}): {n} ({n / len(y):.1%})")
    nulls = df[BREAST_FEATURES].isna().sum()
    print(f"Valores nulos: {nulls[nulls > 0].to_dict()}")
    return df[BREAST_FEATURES], y


def build_breast_pipeline() -> Pipeline:
    """Imputação e padronização dentro do Pipeline, para não vazar informação do teste."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def run_breast(save: bool) -> None:
    X, y = load_breast()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"\nTreino: {len(X_train)} | Teste: {len(X_test)}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(build_breast_pipeline(), {"model__C": np.logspace(-3, 3, 13)},
                          cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"Melhor C: {search.best_params_['model__C']:.4f} | "
          f"AUC-ROC na validação: {search.best_score_:.4f}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    }
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

    print_title("TAREFA 2 - Métricas no TESTE")
    for name, value in metrics.items():
        print(f"{name:<10} {value:.4f}")
    print(f"\nReferência (sempre 'Benigno'): acurácia = {dummy_acc:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print_title("TAREFA 2 - Matriz de confusão (teste)")
    print("                   Previsto Benigno   Previsto Maligno")
    print(f"Real Benigno       {tn:>16}   {fp:>16}")
    print(f"Real Maligno       {fn:>16}   {tp:>16}")
    print(f"\nFalsos negativos (malignos não detectados): {fn}  <- o erro mais grave aqui")

    coefs = (pd.Series(best_model.named_steps["model"].coef_[0], index=BREAST_FEATURES)
             .sort_values(key=np.abs, ascending=False))
    print_title("TAREFA 2 - Coeficientes (ordem de impacto)")
    print(coefs.to_string(float_format="{:+.4f}".format))

    if save:
        joblib.dump(best_model, BASE_DIR / "best_cancer_model.joblib")
        pd.Series(metrics).to_csv(BASE_DIR / "metrics_cancer.csv", header=["valor"],
                                  encoding="utf-8-sig")
        print("\nSalvo: best_cancer_model.joblib e metrics_cancer.csv")


# ========================================================================== #
# Main
# ========================================================================== #
# ========================================================================== #
# Estabilidade: os resultados mudam com a semente?
# ========================================================================== #
def breast_metrics_for_seed(seed: int) -> dict:
    """Mesmo procedimento do run_breast, mas com outra semente. Devolve só as métricas."""
    with contextlib.redirect_stdout(io.StringIO()):
        X, y = load_breast()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    search = GridSearchCV(build_breast_pipeline(), {"model__C": np.logspace(-3, 3, 13)},
                          cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "acuracia": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "falsos_negativos": int(((y_pred == 0) & (y_test.values == 1)).sum()),
    }


def run_stability(floor_mode: str = "flag", seeds=range(10)) -> None:
    """Repete os dois experimentos com sementes diferentes.

    Cada semente muda a divisão treino/teste e a divisão da validação cruzada.
    Não grava arquivos: só imprime o resumo. A semente padrão do script é 42.
    """
    global RANDOM_STATE
    original_seed = RANDOM_STATE
    rent_rows, cancer_rows = [], []
    try:
        for seed in seeds:
            RANDOM_STATE = seed
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_rent(floor_mode, save=False)
                cancer = breast_metrics_for_seed(seed)
            m, best = res["metrics"], res["best"]
            rent_rows.append({
                "semente": seed,
                "modelo_escolhido": best,
                "R2": m.loc[best, "R2"],
                "RMSE": m.loc[best, "RMSE"],
                "MAE": m.loc[best, "MAE"],
                "R2_base": m.loc["LinearRegression", "R2"],
                "RMSE_base": m.loc["LinearRegression", "RMSE"],
            })
            cancer_rows.append({"semente": seed, **cancer})
            print(f"semente {seed} concluída", flush=True)
    finally:
        RANDOM_STATE = original_seed

    rent = pd.DataFrame(rent_rows)
    cancer = pd.DataFrame(cancer_rows)

    print_title("ESTABILIDADE - Aluguel (por semente)")
    print(rent.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print("\nModelo escolhido por semente:", rent["modelo_escolhido"].value_counts().to_dict())

    print_title("ESTABILIDADE - Aluguel (média ± desvio | min | max)")
    for col in ["R2", "RMSE", "MAE", "R2_base", "RMSE_base"]:
        s = rent[col]
        print(f"{col:<10} {s.mean():>14,.4f} ± {s.std():,.4f} | {s.min():,.4f} | {s.max():,.4f}")

    print_title("ESTABILIDADE - Câncer (por semente)")
    print(cancer.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_title("ESTABILIDADE - Câncer (média ± desvio | min | max)")
    for col in ["acuracia", "recall", "precisao", "f1", "auc", "falsos_negativos"]:
        s = cancer[col]
        print(f"{col:<18} {s.mean():>10.4f} ± {s.std():.4f} | {s.min():.4f} | {s.max():.4f}")


def main() -> None:
    # Sem argumento: rodada normal com FLOOR_MODE.
    # Com 'onehot' ou 'flag': usa essa opção de 'floor' na rodada normal.
    # Com 'estabilidade': repete os experimentos com 10 sementes, sem gravar arquivos.
    arg = sys.argv[1] if len(sys.argv) > 1 else FLOOR_MODE
    if arg == "estabilidade":
        run_stability("flag")
        return
    if arg not in ("flag", "onehot"):
        raise SystemExit("Uso: python models_pipeline.py [flag|onehot|estabilidade]")
    run_rent(arg, save=True)
    run_breast(save=True)


if __name__ == "__main__":
    main()
