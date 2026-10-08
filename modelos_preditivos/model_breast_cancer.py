"""
TAREFA 2 - Regressão Logística para diagnosticar câncer de mama.

Dataset: BreastCancer.csv
Alvo:    'Class' (0 = benigno, 1 = maligno)

Modelo: LogisticRegression, com o parâmetro C escolhido por GridSearchCV
        estratificado de 5 folds, usando AUC-ROC como critério.

Saídas (na mesma pasta do script):
    best_cancer_model.joblib   modelo treinado
    metrics_cancer.csv         acurácia, precisão, recall, F1 e AUC-ROC no teste

Uso:
    python model_breast_cancer.py

Descobertas:
- Desempenho forte no teste: acurácia de 0,957, precisão, recall e F1 de 0,938 e AUC-ROC de 0,995. Um modelo que chutasse sempre "benigno" acertaria só 65,7%, então o modelo aprendeu de fato.
- Os erros mais graves são os falsos negativos: 3 casos malignos foram classificados como benignos, de um total de 48 malignos no teste. Num diagnóstico, esse erro significa um paciente que deixaria de ser investigado. Já os 3 falsos positivos gerariam apenas exames extras.
- Variáveis mais influentes: Bare.nuclei (+1,39), Cl.thickness (+1,01) e Bl.cromatin (+0,74) lideram. Todos os coeficientes são positivos, o que faz sentido: células mais alteradas aumentam a chance de malignidade.
- Sem vazamento de dados: nenhuma variável é calculada a partir do alvo. O Id foi removido por ser só um identificador.
- Imputação feita corretamente: os 16 vazios de Bare.nuclei foram preenchidos com a mediana dentro do Pipeline, então só com dados de treino.

Por que esse modelo funciona melhor que o do aluguel:
- As variáveis de célula têm relação forte e clara com o diagnóstico. No aluguel, faltam dados importantes, como o bairro.
- O alvo é binário e equilibrado o suficiente (65% benignos e 35% malignos), sem os extremos que atrapalharam a regressão do aluguel.

Pontos de atenção:
- Amostra pequena: o teste tem só 140 casos. Com 3 erros em 48 malignos, o intervalo de confiança é grande, e outra divisão dos dados poderia dar um resultado diferente.
- Rótulos inferidos: o arquivo não documenta o significado de 0 e 1. Assumimos 0 = benigno e 1 = maligno pela contagem (458 e 241), o que precisa ser confirmado.
- Associação não é causa: os coeficientes mostram quais variáveis andam junto com o diagnóstico, não que causam o câncer.
- Não é um diagnóstico clínico: é um exercício com um dataset histórico.

Resumo: o modelo separa muito bem malignos de benignos (AUC-ROC de 0,995), e as variáveis de célula fazem sentido biologicamente. 
O ponto a melhorar é o recall, porque 3 malignos não detectados são 3 casos que mereceriam atenção. Uma forma de reduzir esses falsos negativos seria ajustar o limiar de decisão do modelo, aceitando mais falsos positivos em troca de detectar mais malignos. 
Isso fica como sugestão para a apresentação, e não foi testado nesta versão.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Configuração

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "Class"
FEATURES = ["Cl.thickness", "Cell.size", "Cell.shape", "Marg.adhesion",
            "Epith.c.size", "Bare.nuclei", "Bl.cromatin",
            "Normal.nucleoli", "Mitoses"]
# O arquivo não documenta o significado de 0 e 1. Pela contagem (458 e 241),
# 0 é benigno e 1 é maligno, que é a proporção do dataset original.
CLASS_NAMES = {0: "Benigno", 1: "Maligno"}


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


# 1. Carga e limpeza

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega o CSV, remove o Id e garante que o alvo seja binário 0/1."""
    df = pd.read_csv(find_csv("BreastCancer.csv")).drop(columns="Id")
    print_title("Dataset de câncer de mama")
    print(f"Shape: {df.shape}")

    y = df[TARGET]
    # O arquivo já vem com 0/1. Se viesse com a codificação original (2/4), convertemos.
    if set(y.unique()) == {2, 4}:
        y = y.map({2: 0, 4: 1})
    assert set(y.unique()) == {0, 1}, f"Alvo não binário: {y.unique()}"

    print("Distribuição do alvo:")
    for code, n in y.value_counts().sort_index().items():
        print(f"  {code} ({CLASS_NAMES[code]}): {n} ({n / len(y):.1%})")

    nulls = df[FEATURES].isna().sum()
    print(f"Valores nulos: {nulls[nulls > 0].to_dict()}")
    return df[FEATURES], y


# 2. Pipeline e treino

def build_pipeline() -> Pipeline:
    """Imputação, padronização e modelo.

    A imputação fica dentro do Pipeline para que a mediana seja calculada
    só com o treino de cada rodada, sem vazar informação do teste.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Escolhe o C por validação cruzada estratificada (mantém a proporção das classes)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(),
        {"model__C": np.logspace(-3, 3, 13)},
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Melhor C: {search.best_params_['model__C']:.4f} | "
          f"AUC-ROC na validação: {search.best_score_:.4f}")
    return search


# 3. Avaliação

def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
    }


def print_confusion_matrix(y_true, y_pred) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print_title("Matriz de confusão (teste)")
    print("                   Previsto Benigno   Previsto Maligno")
    print(f"Real Benigno       {tn:>16}   {fp:>16}")
    print(f"Real Maligno       {fn:>16}   {tp:>16}")
    print(f"\nFalsos negativos (malignos não detectados): {fn}  <- o erro mais grave aqui")


def print_coefficients(model: Pipeline) -> None:
    """Coeficientes padronizados: valor positivo empurra a previsão para Maligno."""
    coefs = (pd.Series(model.named_steps["model"].coef_[0], index=FEATURES)
             .sort_values(key=np.abs, ascending=False))
    print_title("Coeficientes (ordem de impacto)")
    print(coefs.to_string(float_format="{:+.4f}".format))



# Main

def main() -> None:
    X, y = load_data()

    # stratify=y mantém a mesma proporção de benignos e malignos no treino e no teste.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"\nTreino: {len(X_train)} | Teste: {len(X_test)}")

    search = tune_model(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_pred, y_proba)

    # Referência: um modelo que chuta sempre a classe mais comum (Benigno).
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

    print_title("Métricas no TESTE")
    for name, value in metrics.items():
        print(f"{name:<10} {value:.4f}")
    print(f"\nReferência (sempre 'Benigno'): acurácia = {dummy_acc:.4f}")

    print_confusion_matrix(y_test, y_pred)
    print_coefficients(best_model)

    joblib.dump(best_model, BASE_DIR / "best_cancer_model.joblib")
    pd.Series(metrics).to_csv(BASE_DIR / "metrics_cancer.csv", header=["valor"])
    print("\nSalvo: best_cancer_model.joblib e metrics_cancer.csv")


if __name__ == "__main__":
    main()
