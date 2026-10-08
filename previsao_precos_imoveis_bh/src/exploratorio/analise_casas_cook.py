"""Etapa 4 – itens 1 e 2: área das casas (terreno × construída) e pontos influentes (distância de Cook).

Protocolo: separação treino/teste igual à do modelagem.py (80/20, seed 42). Toda comparação usa
validação cruzada SÓ no treino (5-fold × 3 repetições); o teste não é tocado aqui.

Entrada: dados/processados/imoveis_limpo.csv | Saída: resultados/exploratorio/item1_comparacao_casas.csv
Caminhos relativos à raiz do projeto; o script roda de qualquer pasta.

Uso: python src/exploratorio/analise_casas_cook.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

SEED = 42
MIN_ANUNCIOS_BAIRRO = 5
REPETICOES = 3
OPCAO_COOK = "a"  # opção do item 1 usada na análise de Cook (definida após ver a tabela)

#dados
RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "resultados" / "exploratorio"
SAIDA.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(RAIZ / "dados" / "processados" / "imoveis_limpo.csv", encoding="utf-8-sig")
df["tipo"] = df["tipo"].replace({"sobrado": "casa"})
freq = df["bairro"].value_counts()
df["bairro_grp"] = df["bairro"].where(df["bairro"].map(freq) >= MIN_ANUNCIOS_BAIRRO, "Outros")
df["log_area"] = np.log(df["area_m2"])
for t in ("casa", "cobertura"):  # termos de interação área × tipo (referência: apartamento)
    df[f"area_x_{t}"] = df["area_m2"] * (df["tipo"] == t)
    df[f"log_area_x_{t}"] = df["log_area"] * (df["tipo"] == t)

# Casas suspeitas de informar área de TERRENO: área >= 300 m² e R$/m² abaixo do P10 das casas
casas = df["tipo"] == "casa"
p10_casas = df.loc[casas, "preco_m2"].quantile(0.10)
df["suspeita_terreno"] = casas & (df["area_m2"] >= 300) & (df["preco_m2"] < p10_casas)
print(f"Imóveis: {len(df)} | casas: {casas.sum()} | P10 do R$/m² das casas: R$ {p10_casas:,.0f}")
print(f"Casas suspeitas (área >= 300 m² e R$/m² < P10): {df['suspeita_terreno'].sum()}")
print(df.loc[df["suspeita_terreno"], ["preco", "area_m2", "quartos", "banheiros", "vagas", "bairro", "preco_m2"]]
      .sort_values("preco_m2").to_string())

treino, teste = train_test_split(df, test_size=0.2, random_state=SEED)
treino = treino.reset_index(drop=True)
print(f"\nTreino: {len(treino)} (CV só aqui) | Teste: {len(teste)} (reservado, não usado)")
print(f"Suspeitas no treino: {treino['suspeita_terreno'].sum()} | no teste: {teste['suspeita_terreno'].sum()}")

#  opções do item 1
NUM = ["area_m2", "quartos", "banheiros", "vagas"]
NUM_LOG = ["log_area", "quartos", "banheiros", "vagas"]
CAT = ["bairro_grp", "tipo"]
INTER = {False: ["area_x_casa", "area_x_cobertura"], True: ["log_area_x_casa", "log_area_x_cobertura"]}
apto_cob = df["tipo"].isin(["apartamento", "cobertura"])

OPCOES = {
    #       (descrição, usa interação, filtro de linhas que o modelo cobre/treina)
    "base": ("atual (todos os imóveis)", False, lambda d: np.ones(len(d), bool)),
    "a": ("interação área × tipo", True, lambda d: np.ones(len(d), bool)),
    "b": ("só apartamentos + coberturas", False, lambda d: d["tipo"].isin(["apartamento", "cobertura"]).values),
    "c": ("sem as casas suspeitas", False, lambda d: ~d["suspeita_terreno"].values),
}


def montar(log, interacao, cats):
    num = (NUM_LOG if log else NUM) + (INTER[log] if interacao else [])
    ct = ColumnTransformer([("num", StandardScaler(), num),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)])
    return make_pipeline(ct, LinearRegression())


def oof(dados, opcao, log, seed):
    """Previsões fora da dobra (R$) para as linhas cobertas pela opção; NaN nas demais."""
    _, interacao, filtro = OPCOES[opcao]
    cobre = filtro(dados)
    # Só apto+cobertura: 'tipo' perde utilidade como dummy de casa, mas mantém cobertura × apartamento
    prev = np.full(len(dados), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(dados):
        tr = tr[cobre[tr]]
        te = te[cobre[te]]
        m = montar(log, interacao, CAT)
        alvo = np.log(dados["preco"].iloc[tr]) if log else dados["preco"].iloc[tr]
        m.fit(dados.iloc[tr], alvo)
        p = m.predict(dados.iloc[te])
        prev[te] = np.exp(p) if log else p
    return prev


def metricas(real, prev):
    ok = ~np.isnan(prev)
    if ok.sum() < 5:  # opção não cobre este recorte (ex.: (b) não prevê casas)
        return {"n": ok.sum()}
    real, prev = real[ok], prev[ok]
    return {"n": ok.sum(), "R2": r2_score(real, prev), "RMSE": np.sqrt(mean_squared_error(real, prev)),
            "MAE": mean_absolute_error(real, prev)}


# Recortes de avaliação: comparar opções só onde todas preveem, além do escopo de cada uma
RECORTES = {
    "escopo próprio": lambda d: np.ones(len(d), bool),
    "apto + cobertura": lambda d: d["tipo"].isin(["apartamento", "cobertura"]).values,
    "casas não suspeitas": lambda d: ((d["tipo"] == "casa") & ~d["suspeita_terreno"]).values,
}

linhas = []
for log in (False, True):
    for op, (desc, _, _) in OPCOES.items():
        reps = [oof(treino, op, log, SEED + r) for r in range(REPETICOES)]
        for nome_rec, rec in RECORTES.items():
            mask = rec(treino)
            ms = [metricas(treino["preco"].values[mask], p[mask]) for p in reps]
            if ms[0]["n"] < 5:
                continue
            linhas.append({"alvo": "log-log" if log else "linear", "opcao": f"({op}) {desc}", "recorte": nome_rec,
                           "n": ms[0]["n"], "R2_cv": np.mean([m["R2"] for m in ms]),
                           "R2_dp": np.std([m["R2"] for m in ms]), "RMSE_cv": np.mean([m["RMSE"] for m in ms]),
                           "MAE_cv": np.mean([m["MAE"] for m in ms])})

res = pd.DataFrame(linhas)
res.to_csv(SAIDA / "item1_comparacao_casas.csv", index=False, encoding="utf-8-sig")
fmt = res.copy()
fmt["R2_cv"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(res["R2_cv"], res["R2_dp"])]
for c in ("RMSE_cv", "MAE_cv"):
    fmt[c] = res[c].map(lambda v: f"R$ {v/1e3:,.0f} mil")
fmt = fmt.drop(columns="R2_dp")
print("\n=== Item 1 – CV no treino (5-fold × 3), métricas em R$ (log-log ainda SEM smearing; item 3) ===")
for rec in RECORTES:
    print(f"\n--- recorte de avaliação: {rec} ---")
    print(fmt[fmt["recorte"] == rec].drop(columns="recorte").to_string(index=False))

# Coeficientes da interação (log-log, treino, opção a): área de casa vale menos?
X = sm.add_constant(pd.get_dummies(treino[["tipo", "bairro_grp"]], drop_first=False, dtype=float)
                    .drop(columns=["tipo_apartamento", "bairro_grp_Outros"])
                    .join(treino[NUM_LOG + INTER[True]]))
fit_a = sm.OLS(np.log(treino["preco"]), X).fit(cov_type="HC3")
print("\nInteração (a), log-log no treino – elasticidade da área por tipo:")
e_apto = fit_a.params["log_area"]
for t in ("casa", "cobertura"):
    b, p = fit_a.params[f"log_area_x_{t}"], fit_a.pvalues[f"log_area_x_{t}"]
    print(f"  {t}: {e_apto:.3f} + ({b:+.3f}) = {e_apto + b:.3f}  (p da diferença = {p:.3f})")
print(f"  apartamento: {e_apto:.3f}")

# item 2: distância de Cook
print(f"\n=== Item 2 – Distância de Cook (OLS log-log, opção ({OPCAO_COOK}), ajustado no treino) ===")
cobre = OPCOES[OPCAO_COOK][2](treino)
base_cook = treino[cobre].reset_index(drop=True)


INTER_COOK = OPCOES[OPCAO_COOK][1]


def matriz(d):
    dm = pd.get_dummies(d[CAT], dtype=float)
    dm = dm.drop(columns=[c for c in ("tipo_apartamento", "bairro_grp_Outros") if c in dm])
    return sm.add_constant(d[NUM_LOG + (INTER[True] if INTER_COOK else [])].astype(float).join(dm))


Xc, yc = matriz(base_cook), np.log(base_cook["preco"])
fit = sm.OLS(yc, Xc).fit()
cook = fit.get_influence().cooks_distance[0]
n = len(base_cook)
lim = 4 / n
infl = base_cook.assign(cook=cook, residuo_log=fit.resid, previsto=np.exp(fit.fittedvalues))[cook > lim]
print(f"n = {n} | limiar 4/n = {lim:.4f} | influentes: {len(infl)}")
print(infl.sort_values("cook", ascending=False)[["preco", "previsto", "area_m2", "quartos", "banheiros", "vagas",
                                                  "tipo", "bairro", "preco_m2", "cook"]]
      .round({"previsto": 0, "preco_m2": 0, "cook": 4}).to_string())

fit_sem = sm.OLS(yc[cook <= lim], Xc[cook <= lim]).fit()
comp = pd.DataFrame({"coef_todos": fit.params, "p_todos": fit.pvalues,
                     "coef_sem_influentes": fit_sem.params, "p_sem": fit_sem.pvalues})
comp["variacao_%"] = (comp["coef_sem_influentes"] / comp["coef_todos"] - 1) * 100
print(f"\nSensibilidade dos coeficientes (R² log: {fit.rsquared:.3f} -> {fit_sem.rsquared:.3f} sem influentes):")
print(comp.round(4).to_string())

# Sensibilidade preditiva: treinar sem os influentes, avaliar fora da dobra em TODAS as linhas
ids_infl = set(infl.index)
reps_todos, reps_sem = [], []
for r in range(REPETICOES):
    p1, p2 = np.full(n, np.nan), np.full(n, np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=SEED + r).split(base_cook):
        for prev, tr_ in ((p1, tr), (p2, np.array([i for i in tr if i not in ids_infl]))):
            m = montar(True, INTER_COOK, CAT).fit(base_cook.iloc[tr_], np.log(base_cook["preco"].iloc[tr_]))
            prev[te] = np.exp(m.predict(base_cook.iloc[te]))
    reps_todos.append(metricas(base_cook["preco"].values, p1))
    reps_sem.append(metricas(base_cook["preco"].values, p2))
for nome, reps in (("treinando com todos", reps_todos), ("treinando sem influentes", reps_sem)):
    print(f"  CV (avaliado em todas as {n} linhas), {nome}: R² = {np.mean([m['R2'] for m in reps]):.3f} | "
          f"RMSE = R$ {np.mean([m['RMSE'] for m in reps])/1e3:,.0f} mil | "
          f"MAE = R$ {np.mean([m['MAE'] for m in reps])/1e3:,.0f} mil")
print("\nNada foi removido da base: esta é só a análise de sensibilidade.")
