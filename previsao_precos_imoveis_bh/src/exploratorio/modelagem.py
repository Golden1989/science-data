"""Etapa 4 – Regressão linear para prever o preço dos imóveis.

Primeira versão da Etapa 4 (exploratória; o modelo final está em src/modelo_final.py).
Entrada: dados/processados/imoveis_limpo.csv (gerado por src/limpeza.py)
Saída:   resultados/exploratorio/resultados_modelos.csv, resultados/exploratorio/coeficientes_ols_*.csv,
         graficos/exploratorio/5_*.png e 6_*.png
Caminhos relativos à raiz do projeto; o script roda de qualquer pasta.

Compara:
  - conjuntos de variáveis: base (área, quartos, banheiros, vagas) -> + bairro (one-hot) -> + tipo
  - estimadores: OLS, Ridge e Lasso (alpha escolhido por validação cruzada)
  - alvo: preco e log(preco)
e verifica multicolinearidade (VIF).

Uso: python src/exploratorio/modelagem.py
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
MIN_ANUNCIOS_BAIRRO = 5  # bairros com menos anúncios viram "Outros"
RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "resultados" / "exploratorio"
PASTA = RAIZ / "graficos" / "exploratorio"  # fora de graficos/ para não misturar com as figuras do relatório
for _p in (SAIDA, PASTA):
    _p.mkdir(parents=True, exist_ok=True)

AZUL, LARANJA, TINTA, TINTA_2, FUNDO, GRADE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
    "axes.edgecolor": GRADE, "axes.grid": True, "grid.color": GRADE, "axes.axisbelow": True,
    "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left", "font.size": 10,
})
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)

# dados
df = pd.read_csv(RAIZ / "dados" / "processados" / "imoveis_limpo.csv", encoding="utf-8-sig")
freq = df["bairro"].value_counts()
df["bairro_grp"] = df["bairro"].where(df["bairro"].map(freq) >= MIN_ANUNCIOS_BAIRRO, "Outros")
df["tipo"] = df["tipo"].replace({"sobrado": "casa"})  # só 1 sobrado: agrupa com casa
# Nos modelos com alvo log(preco) a área entra em log (modelo log-log): com a área linear, exp() da
# previsão explode para imóveis muito grandes. O coeficiente vira elasticidade (+1% área -> +b% preço).
df["log_area"] = np.log(df["area_m2"])

NUM = ["area_m2", "quartos", "banheiros", "vagas"]
CONJUNTOS = {
    "base": (NUM, []),
    "base + bairro": (NUM, ["bairro_grp"]),
    "base + bairro + tipo": (NUM, ["bairro_grp", "tipo"]),
}
REFERENCIA = {"bairro_grp": "Outros", "tipo": "apartamento"}  # categorias de referência (dummy omitida)

print(f"Imóveis: {len(df)}")
print(f"Bairros com >= {MIN_ANUNCIOS_BAIRRO} anúncios (demais = 'Outros'): "
      f"{df['bairro_grp'].nunique() - 1} | 'Outros': {(df['bairro_grp'] == 'Outros').sum()} imóveis")

X, y = df[NUM + ["log_area", "bairro_grp", "tipo"]], df["preco"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"Treino: {len(X_tr)} | Teste: {len(X_te)} | + validação cruzada 5-fold na base inteira")


# VIF
def tabela_vif(dados, cols):
    m = sm.add_constant(dados[cols].astype(float))
    return pd.Series([variance_inflation_factor(m.values, i) for i in range(1, m.shape[1])], index=cols)


print("\n=== Multicolinearidade (VIF) ===")
vif = pd.DataFrame({"VIF": tabela_vif(df, NUM)})
print(vif.round(2).to_string())
print("Referência: VIF < 5 aceitável; 5–10 moderado; > 10 grave")
print("Correlações entre preditoras:\n" + df[NUM].corr().round(2).to_string())


# modelos
def montar(num, cat, estimador, log):
    if log:
        num = ["log_area" if c == "area_m2" else c for c in num]
    etapas = [("num", StandardScaler(), num)]
    if cat:
        etapas.append(("cat", OneHotEncoder(drop=[REFERENCIA[c] for c in cat], sparse_output=False), cat))
    pipe = make_pipeline(ColumnTransformer(etapas), estimador)
    if log:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


ESTIMADORES = {
    "OLS": lambda: LinearRegression(),
    "Ridge": lambda: RidgeCV(alphas=np.logspace(-3, 3, 50)),
    "Lasso": lambda: LassoCV(alphas=100, cv=5, random_state=SEED, max_iter=50000),
}


def metricas(real, prev):
    return r2_score(real, prev), np.sqrt(mean_squared_error(real, prev)), mean_absolute_error(real, prev)


linhas, ajustados = [], {}
for nome_conj, (num, cat) in CONJUNTOS.items():
    for nome_est, fabrica in ESTIMADORES.items():
        for log in (False, True):
            modelo = montar(num, cat, fabrica(), log)
            modelo.fit(X_tr, y_tr)
            r2, rmse, mae = metricas(y_te, modelo.predict(X_te))
            prev_cv = cross_val_predict(montar(num, cat, fabrica(), log), X, y, cv=kfold)
            r2_cv, rmse_cv, mae_cv = metricas(y, prev_cv)
            alvo = "log(preco)" if log else "preco"
            linhas.append({"variaveis": nome_conj, "modelo": nome_est, "alvo": alvo,
                           "R2_teste": r2, "RMSE_teste": rmse, "MAE_teste": mae,
                           "R2_cv": r2_cv, "RMSE_cv": rmse_cv, "MAE_cv": mae_cv})
            ajustados[(nome_conj, nome_est, alvo)] = (modelo, prev_cv)

res = pd.DataFrame(linhas)
res.to_csv(SAIDA / "resultados_modelos.csv", index=False, encoding="utf-8-sig")
print("\n=== Comparação de modelos (métricas em R$; modelos log são convertidos de volta com exp) ===")
fmt = res.copy()
for c in ["R2_teste", "R2_cv"]:
    fmt[c] = fmt[c].map("{:.3f}".format)
for c in ["RMSE_teste", "MAE_teste", "RMSE_cv", "MAE_cv"]:
    fmt[c] = fmt[c].map(lambda v: f"{v/1e3:,.0f} mil")
print(fmt.to_string(index=False))

melhor = res.loc[res["R2_cv"].idxmax()]
print(f"\nMelhor pela validação cruzada: {melhor['modelo']} | {melhor['variaveis']} | alvo {melhor['alvo']} "
      f"(R² cv = {melhor['R2_cv']:.3f}, MAE cv = R$ {melhor['MAE_cv']:,.0f})")

# Hiperparâmetros e variáveis zeradas pelo Lasso no conjunto completo
for alvo in ("preco", "log(preco)"):
    for nome_est in ("Ridge", "Lasso"):
        m = ajustados[("base + bairro + tipo", nome_est, alvo)][0]
        pipe = m.regressor_ if hasattr(m, "regressor_") else m
        est = pipe[-1]
        nomes = pipe[0].get_feature_names_out()
        zerados = [n.split("__")[1] for n, c in zip(nomes, est.coef_) if abs(c) < 1e-10]
        extra = f" | coeficientes zerados: {len(zerados)}/{len(nomes)} {zerados}" if nome_est == "Lasso" else ""
        print(f"{nome_est} ({alvo}): alpha = {est.alpha_:.4g}{extra}")


# interpretação (statsmodels, base inteira)
NUM_LOG = ["log_area", "quartos", "banheiros", "vagas"]


def matriz(dados, num=NUM):
    X_ = dados[num].astype(float)
    for c in ("bairro_grp", "tipo"):
        d = pd.get_dummies(dados[c], prefix=c.replace("_grp", ""), dtype=float)
        X_ = X_.join(d.drop(columns=f"{c.replace('_grp', '')}_{REFERENCIA[c]}"))
    return sm.add_constant(X_)


Xc = matriz(df)
print("\nVIF com as dummies de bairro e tipo (maiores):")
print(pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
                index=Xc.columns[1:]).sort_values(ascending=False).head(6).round(2).to_string())

# Erros-padrão robustos (HC3): os resíduos crescem com o preço (heterocedasticidade)
ols_lin = sm.OLS(df["preco"], Xc).fit(cov_type="HC3")
ols_log = sm.OLS(np.log(df["preco"]), matriz(df, NUM_LOG)).fit(cov_type="HC3")


def tabela_coef(fit, log):
    t = pd.DataFrame({"coef": fit.params, "erro_padrao": fit.bse, "p_valor": fit.pvalues})
    if log:
        # Efeito % no preço de +1 unidade (dummies: diferença vs. referência). log_area é elasticidade
        # (lê-se o próprio coef: +1% de área -> +coef% no preço) e o intercepto não tem leitura percentual.
        t["efeito_%"] = (np.exp(t["coef"]) - 1) * 100
        t.loc[["const", "log_area"], "efeito_%"] = np.nan
    return t


coef_lin, coef_log = tabela_coef(ols_lin, False), tabela_coef(ols_log, True)
coef_lin.to_csv(SAIDA / "coeficientes_ols_preco.csv", encoding="utf-8-sig")
coef_log.to_csv(SAIDA / "coeficientes_ols_log_preco.csv", encoding="utf-8-sig")

print(f"\n=== OLS preco ~ base + bairro + tipo (base inteira, n={len(df)}) | "
      f"R² = {ols_lin.rsquared:.3f} | R² ajustado = {ols_lin.rsquared_adj:.3f} ===")
print(coef_lin.round({"coef": 0, "erro_padrao": 0, "p_valor": 4}).to_string())
print(f"\n=== OLS log(preco) ~ log(area) + quartos + banheiros + vagas + bairro + tipo | R² (escala log) = {ols_log.rsquared:.3f} | "
      f"R² ajustado = {ols_log.rsquared_adj:.3f} ===")
print(coef_log.round({"coef": 4, "erro_padrao": 4, "p_valor": 4, "efeito_%": 1}).to_string())

print("\nLeitura (mantidas as demais variáveis constantes):")
print(f"  +1 m²: {coef_lin.loc['area_m2', 'coef']:+,.0f} R$ (modelo linear, p={coef_lin.loc['area_m2', 'p_valor']:.3f}) | "
      f"+1% de área: {coef_log.loc['log_area', 'coef']:+.2f}% no preço (modelo log-log, p={coef_log.loc['log_area', 'p_valor']:.3f})")
for v, unid in [("quartos", "quarto"), ("banheiros", "banheiro"), ("vagas", "vaga")]:
    print(f"  +1 {unid}: {coef_lin.loc[v, 'coef']:+,.0f} R$ (modelo linear, p={coef_lin.loc[v, 'p_valor']:.3f}) | "
          f"{coef_log.loc[v, 'efeito_%']:+.2f}% (modelo log, p={coef_log.loc[v, 'p_valor']:.3f})")

# gráficos (previsões da validação cruzada)
prev_lin = ajustados[("base + bairro + tipo", "OLS", "preco")][1]
prev_log = ajustados[("base + bairro + tipo", "OLS", "log(preco)")][1]
mi = lambda v, _: f"{v/1e6:.1f} mi"

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
lim = [0, max(y.max(), prev_lin.max(), prev_log.max()) * 1.05]
for ax, prev, titulo in [(axes[0], prev_lin, "OLS – alvo preco"), (axes[1], prev_log, "OLS – alvo log(preco)")]:
    ax.plot(lim, lim, color=TINTA, lw=1.5, ls="--", label="previsão perfeita")
    ax.scatter(y, prev, s=26, color=AZUL, alpha=0.6, edgecolor=FUNDO, linewidth=0.8)
    r2, _, mae = metricas(y, prev)
    ax.text(0.03, 0.95, f"R² = {r2:.2f}\nMAE = R$ {mae/1e3:,.0f} mil", transform=ax.transAxes, va="top", color=TINTA_2)
    ax.set(title=titulo, xlabel="Preço real (R$)", xlim=lim, ylim=lim)
    ax.xaxis.set_major_formatter(mi)
    ax.yaxis.set_major_formatter(mi)
axes[0].set_ylabel("Preço previsto (R$)")
axes[1].legend(loc="lower right", frameon=False)
fig.suptitle("Real × previsto (validação cruzada 5-fold, base + bairro + tipo)", x=0.01, ha="left", color=TINTA_2)
fig.tight_layout()
fig.savefig(PASTA / "5_real_x_previsto.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(prev_lin, y - prev_lin, s=24, color=AZUL, alpha=0.6, edgecolor=FUNDO, linewidth=0.8)
axes[0].set(title="Resíduos – alvo preco", xlabel="Preço previsto (R$)", ylabel="Resíduo (R$)")
axes[0].xaxis.set_major_formatter(mi)
axes[0].yaxis.set_major_formatter(mi)
prev_log_ln = ajustados[("base + bairro + tipo", "OLS", "log(preco)")][1]
axes[1].scatter(np.log(prev_log_ln), np.log(y) - np.log(prev_log_ln), s=24, color=AZUL, alpha=0.6,
                edgecolor=FUNDO, linewidth=0.8)
axes[1].set(title="Resíduos – alvo log(preco)", xlabel="log(preço previsto)", ylabel="Resíduo (log)")
for ax in axes:
    ax.axhline(0, color=TINTA, lw=1.5, ls="--")
fig.tight_layout()
fig.savefig(PASTA / "6_residuos.png", dpi=150)
plt.close(fig)
print(f"\n-> {PASTA / '5_real_x_previsto.png'}\n-> {PASTA / '6_residuos.png'}")
