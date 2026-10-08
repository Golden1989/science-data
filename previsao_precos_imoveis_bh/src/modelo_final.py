"""Etapa 4 – modelo final de ponta a ponta.

Especificação (decidida na validação cruzada no treino):
  Principal:    preco      ~ area_m2 + area_m2×casa + casa + quartos + banheiros + vagas + regional [+ lancamento]
  Complementar: log(preco) ~ log(area) + log(area)×casa + casa + quartos + banheiros + vagas + regional [+ lancamento]
                (volta para R$ com smearing de Duan; usado para ler efeitos em %)
  Regional administrativa de BH (regionais_bh.csv), referência = Oeste. Cobertura conta como apartamento.
  Pontos influentes mantidos. Critério de escolha: métricas em R$ (R², RMSE, MAE).

Protocolo:
  1. treino/teste 80/20 (seed 42);
  2. `lancamento` (link /imoveis-lancamentos/), identificada DEPOIS da 1ª avaliação no teste: entra só se o
     linear melhorar R², RMSE e MAE na CV 5×3 do treino; o log-log segue a mesma decisão;
  3. OLS × Ridge × Lasso por CV 5-fold × 3 repetições só no treino. Regra fixada antes: Ridge/Lasso só
     substituem o OLS se o R² em R$ subir mais de 2 desvios-padrão (entre repetições) E RMSE e MAE caírem;
     Random Forest com as mesmas variáveis entra só como referência;
  4. ajuste final no treino inteiro; o teste é reportado nas duas avaliações (sem e com `lancamento`);
  5. diagnósticos (Breusch-Pagan, VIF, QQ-plot, resíduos) e robustez (Cook) em src/diagnosticos.py;
     o texto de relatorio/resumo_etapa4.md é escrito por src/gerar_resumo.py. Este script chama os dois.

Entradas: dados/processados/imoveis_limpo.csv, dados/regionais_bh.csv
Saídas:   resultados/coeficientes_final_*.csv (aqui); graficos/5_–7_*.png, resultados/vif_final.csv,
          resultados/breusch_pagan.csv (diagnosticos.py); relatorio/resumo_etapa4.md (gerar_resumo.py)
Caminhos relativos à raiz do projeto; o script roda de qualquer pasta.

Uso: python src/modelo_final.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import diagnosticos  # mesma pasta (src/)
import gerar_resumo

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

SEED = 42
REPETICOES = 3
REF_REGIONAL = "Oeste"
RAIZ = Path(__file__).resolve().parents[1]
DADOS, RESULTADOS, RELATORIO = RAIZ / "dados", RAIZ / "resultados", RAIZ / "relatorio"
PASTA = RAIZ / "graficos"
for _p in (PASTA, RESULTADOS, RELATORIO):
    _p.mkdir(exist_ok=True)

def br(v, casas=0):
    """Número no formato brasileiro: 1234567.8 -> '1.234.567,8'."""
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def reais(v):
    return f"R$ {br(v)}"


def pv(p):
    """p-valor no formato brasileiro."""
    return "< 0,001" if p < 0.001 else br(p, 3)


# p-valor com o sinal, para textos de gráfico e do resumo ("< 0,001" ou "= 0,030")
p_txt = lambda p: "< 0,001" if p < 0.001 else f"= {br(p, 3)}"


# dados
df = pd.read_csv(DADOS / "processados" / "imoveis_limpo.csv", encoding="utf-8-sig")
df["tipo"] = df["tipo"].replace({"sobrado": "casa"})
reg = pd.read_csv(DADOS / "regionais_bh.csv", encoding="utf-8").fillna("")
faltando = sorted(set(df["bairro"]) - set(reg["bairro"]))
if faltando:
    raise SystemExit(f"Bairros sem regional em regionais_bh.csv: {faltando}")
df = df.merge(reg[["bairro", "regional"]], on="bairro", how="left")
df["casa"] = (df["tipo"] == "casa").astype(float)
df["log_area"] = np.log(df["area_m2"])
df["area_x_casa"] = df["area_m2"] * df["casa"]
df["log_area_x_casa"] = df["log_area"] * df["casa"]
REGIONAIS = sorted(set(df["regional"]) - {REF_REGIONAL})
for r in REGIONAIS:  # dummies fixas (mapeamento determinístico, não usa o preço)
    df[f"reg_{r}"] = (df["regional"] == r).astype(float)

DUMMIES = [f"reg_{r}" for r in REGIONAIS]
X_LIN = ["area_m2", "area_x_casa", "casa", "quartos", "banheiros", "vagas"] + DUMMIES
X_LOG = ["log_area", "log_area_x_casa", "casa", "quartos", "banheiros", "vagas"] + DUMMIES
X_LIN_BASE, X_LOG_BASE = list(X_LIN), list(X_LOG)  # especificação avaliada no teste na 1ª rodada

treino, teste = train_test_split(df, test_size=0.2, random_state=SEED)
treino, teste = treino.reset_index(drop=True), teste.reset_index(drop=True)
piso = treino["preco"].min()
print(f"Imóveis: {len(df)} | treino: {len(treino)} | teste: {len(teste)} | menor preço do treino: {reais(piso)}")

# CV no treino: OLS × Ridge × Lasso
ESTIMADORES = {
    "OLS": lambda: LinearRegression(),
    "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 60))),
    "Lasso": lambda: make_pipeline(StandardScaler(), LassoCV(alphas=100, cv=5, max_iter=50000, random_state=SEED)),
}


def cv(cols, log, fabrica):
    y = treino["preco"].values
    m_rep = []
    abaixo = 0
    for r in range(REPETICOES):
        prev = np.full(len(treino), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=SEED + r).split(treino):
            Xtr, Xte = treino.loc[tr, cols], treino.loc[te, cols]
            ytr = np.log(y[tr]) if log else y[tr]
            m = fabrica().fit(Xtr, ytr)
            p = m.predict(Xte)
            if log:
                p = np.exp(p) * np.mean(np.exp(ytr - m.predict(Xtr)))  # Duan, fator do fold
            else:
                abaixo += (p < y[tr].min()).sum()
            prev[te] = p
        m_rep.append((r2_score(y, prev), np.sqrt(mean_squared_error(y, prev)), mean_absolute_error(y, prev),
                      np.median(np.abs(y - prev))))
    a = np.array(m_rep)
    return {"R2": a[:, 0].mean(), "R2_dp": a[:, 0].std(), "RMSE": a[:, 1].mean(), "MAE": a[:, 2].mean(),
            "MedAE": a[:, 3].mean(), "abaixo_min_por_rep": abaixo / REPETICOES if not log else np.nan}


# variável `lancamento` (pós-teste)
# Identificada DEPOIS da 1ª avaliação no teste (link com /imoveis-lancamentos/). Regra fixada antes de rodar:
# entra no modelo se o linear (principal) melhorar R², RMSE e MAE na CV do treino; o log-log segue a mesma decisão.
lanc_tab = pd.DataFrame([
    {"modelo": nome, "variaveis": rot, **cv(cols + extra, log, lambda: LinearRegression())}
    for nome, (cols, log) in (("linear", (X_LIN_BASE, False)), ("log-log", (X_LOG_BASE, True)))
    for rot, extra in (("atual", []), ("+ lancamento", ["lancamento"]))
])
lanc_atual = lanc_tab[(lanc_tab.modelo == "linear") & (lanc_tab.variaveis == "atual")].iloc[0]
lanc_novo = lanc_tab[(lanc_tab.modelo == "linear") & (lanc_tab.variaveis == "+ lancamento")].iloc[0]
ADOTA_LANC = bool(lanc_novo.R2 > lanc_atual.R2 and lanc_novo.RMSE < lanc_atual.RMSE
                  and lanc_novo.MAE < lanc_atual.MAE)
print("\n=== `lancamento`: CV 5×3 no treino, OLS (métricas em R$) ===")
print(lanc_tab.assign(R2=lanc_tab.R2.round(3), RMSE=lanc_tab.RMSE.map(reais), MAE=lanc_tab.MAE.map(reais),
                      MedAE=lanc_tab.MedAE.map(reais)).drop(columns=["R2_dp", "abaixo_min_por_rep"]).to_string(index=False))
print(f"Adota `lancamento`: {'SIM' if ADOTA_LANC else 'NÃO'} [regra: linear melhora R², RMSE e MAE]")
if ADOTA_LANC:
    X_LIN, X_LOG = X_LIN_BASE + ["lancamento"], X_LOG_BASE + ["lancamento"]
MODELOS = {"linear": (X_LIN, False), "log-log": (X_LOG, True)}

linhas = []
for nome, (cols, log) in MODELOS.items():
    for est, fab in ESTIMADORES.items():
        linhas.append({"modelo": nome, "estimador": est, **cv(cols, log, fab)})
cv_tab = pd.DataFrame(linhas)
print("\n=== CV 5×3 no treino (métricas em R$; log-log com Duan) ===")
print(cv_tab.assign(R2=[f"{a:.3f} ± {b:.3f}" for a, b in zip(cv_tab.R2, cv_tab.R2_dp)],
                    RMSE=cv_tab.RMSE.map(lambda v: f"{v/1e3:,.0f} mil"), MAE=cv_tab.MAE.map(lambda v: f"{v/1e3:,.0f} mil"),
                    MedAE=cv_tab.MedAE.map(lambda v: f"{v/1e3:,.0f} mil")).drop(columns="R2_dp").to_string(index=False))

escolha = {}
for nome in MODELOS:
    t = cv_tab[cv_tab["modelo"] == nome].set_index("estimador")
    ols = t.loc["OLS"]
    escolha[nome] = "OLS"
    for est in ("Ridge", "Lasso"):
        o = t.loc[est]
        if o.R2 - ols.R2 > 2 * ols.R2_dp and o.RMSE < ols.RMSE and o.MAE < ols.MAE:
            escolha[nome] = est
    print(f"Escolha ({nome}): {escolha[nome]}  [regra: ganho de R² > 2 dp ({2 * ols.R2_dp:.3f}) e RMSE e MAE menores]")
if any(v != "OLS" for v in escolha.values()):
    print("Decisão registrada: OLS mantido (ganho marginal; o dp entre repetições subestima a incerteza; ver resumo).")

# Benchmark não linear (só referência; não substitui o modelo principal): mesmas variáveis e mesma CV do linear
bench = cv(X_LIN, False, lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED,
                                                       n_jobs=-1))
print(f"Benchmark Random Forest (variáveis do linear): R² {bench['R2']:.3f} ± {bench['R2_dp']:.3f} | "
      f"RMSE {reais(bench['RMSE'])} | MAE {reais(bench['MAE'])} | erro mediano {reais(bench['MedAE'])}")

# ajuste final no treino inteiro
y_tr, y_te = treino["preco"], teste["preco"]
Xl_tr, Xl_te = sm.add_constant(treino[X_LIN]), sm.add_constant(teste[X_LIN], has_constant="add")
Xg_tr, Xg_te = sm.add_constant(treino[X_LOG]), sm.add_constant(teste[X_LOG], has_constant="add")
# Erros-padrão robustos HC3 (a heterocedasticidade é testada abaixo com Breusch-Pagan)
fit_lin = sm.OLS(y_tr, Xl_tr).fit(cov_type="HC3")
fit_log = sm.OLS(np.log(y_tr), Xg_tr).fit(cov_type="HC3")
duan = np.mean(np.exp(fit_log.resid))

# teste: avaliação única
prev_lin = fit_lin.predict(Xl_te).values
prev_log = np.exp(fit_log.predict(Xg_te).values) * duan


def metricas(y, p):
    return {"R2": r2_score(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p),
            "MedAE": np.median(np.abs(y - p))}


teste_tab = pd.DataFrame([{"modelo": "linear (principal)", **metricas(y_te, prev_lin)},
                          {"modelo": "log-log + Duan (complementar)", **metricas(y_te, prev_log)}])
abaixo_teste = int((prev_lin < piso).sum())
negativas_teste = int((prev_lin <= 0).sum())

# 1ª avaliação no teste (especificação sem `lancamento`), reproduzida para comparar antes × depois
f0_lin = sm.OLS(y_tr, sm.add_constant(treino[X_LIN_BASE])).fit()
f0_log = sm.OLS(np.log(y_tr), sm.add_constant(treino[X_LOG_BASE])).fit()
p0_lin = f0_lin.predict(sm.add_constant(teste[X_LIN_BASE], has_constant="add")).values
p0_log = np.exp(f0_log.predict(sm.add_constant(teste[X_LOG_BASE], has_constant="add")).values) \
    * np.mean(np.exp(f0_log.resid))
teste_antes = pd.DataFrame([{"modelo": "linear (principal)", **metricas(y_te, p0_lin)},
                            {"modelo": "log-log + Duan (complementar)", **metricas(y_te, p0_log)}])
abaixo_teste_antes = int((p0_lin < piso).sum())

print("\n=== TESTE ===")
for rot, tab in (("1ª avaliação (sem lancamento)", teste_antes),
                 (f"2ª avaliação ({'com' if ADOTA_LANC else 'sem'} lancamento)", teste_tab)):
    print(f"-- {rot}")
    print(tab.assign(R2=tab.R2.round(3), RMSE=tab.RMSE.map(reais), MAE=tab.MAE.map(reais),
                     MedAE=tab.MedAE.map(reais)).to_string(index=False))
print(f"Previsões do linear abaixo do menor preço do treino ({reais(piso)}): {abaixo_teste} de {len(teste)} "
      f"(<= 0: {negativas_teste}) | fator de Duan: {duan:.4f}")
erros_log = (teste.assign(previsto_log=prev_log, previsto_lin=prev_lin, erro_log=prev_log - y_te)
             .reindex((prev_log - y_te).abs().sort_values(ascending=False).index).head(3))
print("Maiores erros do log-log no teste:")
print(erros_log[["preco", "previsto_log", "previsto_lin", "area_m2", "tipo", "bairro", "regional"]]
      .round({"preco": -3, "previsto_log": -3, "previsto_lin": -3}).to_string())

# coeficientes interpretados
coef_lin = pd.DataFrame({"coef": fit_lin.params, "erro_padrao_HC3": fit_lin.bse, "p_valor": fit_lin.pvalues})
coef_log = pd.DataFrame({"coef": fit_log.params, "erro_padrao_HC3": fit_log.bse, "p_valor": fit_log.pvalues})
coef_log["efeito_%"] = (np.exp(coef_log["coef"]) - 1) * 100
coef_log.loc[["const", "log_area", "log_area_x_casa", "casa"], "efeito_%"] = np.nan  # sem leitura isolada
coef_lin.to_csv(RESULTADOS / "coeficientes_final_linear.csv", encoding="utf-8-sig")
coef_log.to_csv(RESULTADOS / "coeficientes_final_loglog.csv", encoding="utf-8-sig")
print("\n=== Coeficientes – linear (treino, HC3) ===")
print(coef_lin.round({"coef": 0, "erro_padrao_HC3": 0, "p_valor": 4}).to_string())
print("\n=== Coeficientes – log-log (treino, HC3) ===")
print(coef_log.round(4).to_string())


def combinacao(fit, pesos):
    """Estimativa, erro-padrão e p-valor de uma combinação linear de coeficientes."""
    r = np.zeros(len(fit.params))
    for nome, w in pesos.items():
        r[list(fit.params.index).index(nome)] = w
    t = fit.t_test(r)
    return float(np.squeeze(t.effect)), float(np.squeeze(t.sd)), float(np.squeeze(t.pvalue))


area_casa_tip = treino.loc[treino["casa"] == 1, "area_m2"].median()
m2_apto = (coef_lin.loc["area_m2", "coef"], coef_lin.loc["area_m2", "erro_padrao_HC3"], coef_lin.loc["area_m2", "p_valor"])
m2_casa = combinacao(fit_lin, {"area_m2": 1, "area_x_casa": 1})
casa_vs_apto = combinacao(fit_lin, {"casa": 1, "area_x_casa": area_casa_tip})
el_apto = (coef_log.loc["log_area", "coef"], coef_log.loc["log_area", "erro_padrao_HC3"], coef_log.loc["log_area", "p_valor"])
el_casa = combinacao(fit_log, {"log_area": 1, "log_area_x_casa": 1})
casa_vs_apto_log = combinacao(fit_log, {"casa": 1, "log_area_x_casa": np.log(area_casa_tip)})
n_reg = treino["regional"].value_counts()

# Efeito marginal em R$/m² a partir da elasticidade do log-log, no ponto mediano de cada tipo:
# dP/dA = b × P/A. IC 95% pelo método delta tratando as medianas como fixas (só b é incerto) -> (b ± 1,96·ep)·P/A
def marginal(el, mascara):
    P, A = treino.loc[mascara, "preco"].median(), treino.loc[mascara, "area_m2"].median()
    b, ep = el[0], el[1]
    return {"P": P, "A": A, "ef": b * P / A, "ic": ((b - 1.96 * ep) * P / A, (b + 1.96 * ep) * P / A), "p": el[2]}


mg_apto = marginal(el_apto, treino["casa"] == 0)
mg_casa = marginal(el_casa, treino["casa"] == 1)
print(f"Efeito marginal (log-log, ponto mediano): apto {reais(mg_apto['ef'])}/m² "
      f"[IC95% {reais(mg_apto['ic'][0])}; {reais(mg_apto['ic'][1])}] | casa {reais(mg_casa['ef'])}/m² "
      f"[IC95% {reais(mg_casa['ic'][0])}; {reais(mg_casa['ic'][1])}]")

print(f"\nm² no apartamento: {reais(m2_apto[0])} (p={m2_apto[2]:.3g}) | m² na casa: {reais(m2_casa[0])} (p={m2_casa[2]:.3g})")
print(f"Elasticidade da área: apto {el_apto[0]:.3f} (p={el_apto[2]:.3g}) | casa {el_casa[0]:.3f} (p={el_casa[2]:.3g})")
print(f"Casa × apto com {br(area_casa_tip)} m²: {reais(casa_vs_apto[0])} (p={casa_vs_apto[2]:.3g}) | "
      f"{(np.exp(casa_vs_apto_log[0]) - 1) * 100:+.1f}% (p={casa_vs_apto_log[2]:.3g})")

# diagnósticos e resumo
# Os dois módulos recebem todos os valores calculados acima (sem recalcular o modelo)
ctx = SimpleNamespace(**{k: v for k, v in globals().items() if not k.startswith("__")})
vars(ctx).update(diagnosticos.executar(ctx))
gerar_resumo.escrever(ctx)
