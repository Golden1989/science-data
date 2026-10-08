"""Etapa 4 – diagnósticos do modelo final (chamado por src/modelo_final.py).

Breusch-Pagan, VIF (bruto, com a área centrada e na forma com a área separada por tipo), figuras 5–7 do relatório
(real × previsto no teste, resíduos × previsto, QQ-plot), distância de Cook e a análise dos maiores erros do
log-log na CV.

Saídas: graficos/5_teste_real_x_previsto.png, graficos/6_residuos_x_previsto.png, graficos/7_qqplot_residuos.png,
        resultados/breusch_pagan.csv, resultados/vif_final.csv

Uso: não é executado sozinho; rode `python src/modelo_final.py`.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

AZUL, TINTA, TINTA_2, FUNDO, GRADE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
    "axes.edgecolor": GRADE, "axes.grid": True, "grid.color": GRADE, "axes.axisbelow": True,
    "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left", "font.size": 10,
})


def executar(ctx):
    """Roda os diagnósticos a partir do contexto do modelo_final e devolve os valores usados pelo resumo."""
    # Valores calculados em modelo_final.py
    br, DUMMIES, fit_lin, fit_log, m2_apto = ctx.br, ctx.DUMMIES, ctx.fit_lin, ctx.fit_log, ctx.m2_apto
    metricas, p_txt, PASTA, prev_lin = ctx.metricas, ctx.p_txt, ctx.PASTA, ctx.prev_lin
    prev_log, reais, REPETICOES, RESULTADOS = ctx.prev_log, ctx.reais, ctx.REPETICOES, ctx.RESULTADOS
    SEED, teste, treino, X_LIN, X_LOG = ctx.SEED, ctx.teste, ctx.treino, ctx.X_LIN, ctx.X_LOG
    Xg_tr, Xl_tr, y_te, y_tr = ctx.Xg_tr, ctx.Xl_tr, ctx.y_te, ctx.y_tr

    # diagnósticos
    bp = []
    for nome, fit in (("linear", fit_lin), ("log-log", fit_log)):
        lm, lm_p, f, f_p = het_breuschpagan(fit.resid, fit.model.exog)
        bp.append({"modelo": nome, "LM": lm, "p_valor_LM": lm_p, "F": f, "p_valor_F": f_p,
                   "conclusao": "heterocedástico (rejeita H0 a 5%)" if lm_p < 0.05 else "não rejeita homocedasticidade "
                                                                                        "a 5%"})
    bp = pd.DataFrame(bp)
    bp.to_csv(RESULTADOS / "breusch_pagan.csv", index=False, encoding="utf-8-sig")
    print("\n=== Breusch-Pagan (resíduos do ajuste no treino) ===")
    print(bp.round(4).to_string(index=False))


    def vif(X):
        return pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=X.columns[1:])


    # VIF bruto e com a área centrada (separa a colinearidade estrutural da interação da colinearidade real)
    def centrada(cols, area, inter):
        X = treino[cols].copy()
        X[area] = X[area] - X[area].mean()
        X[inter] = X[area] * X["casa"]
        return sm.add_constant(X)


    NOMES_LOG = {"log_area": "area_m2", "log_area_x_casa": "area_x_casa"}  # alinha as linhas das duas tabelas
    vif_tab = pd.DataFrame({
        "VIF_linear": vif(Xl_tr),
        "VIF_linear_area_centrada": vif(centrada(X_LIN, "area_m2", "area_x_casa")),
        "VIF_loglog": vif(Xg_tr).rename(NOMES_LOG),
        "VIF_loglog_area_centrada": vif(centrada(X_LOG, "log_area", "log_area_x_casa")).rename(NOMES_LOG),
    })
    vif_tab.index.name = "variavel"
    # Forma equivalente com a área separada por tipo (area_apto, area_casa): mesmo ajuste, sem o termo que soma as duas
    Xr = treino[X_LIN].drop(columns=["area_m2", "area_x_casa"]).assign(
        area_apto=treino["area_m2"] * (1 - treino["casa"]), area_casa=treino["area_m2"] * treino["casa"])
    vif_reparam = vif(sm.add_constant(Xr))
    vif_sem_inter = vif(sm.add_constant(treino[["area_m2", "quartos", "banheiros", "vagas"] + DUMMIES]))
    corr_apto = (treino.loc[treino["casa"] == 0, ["area_m2", "quartos", "banheiros", "vagas"]]
                 .corr().loc["area_m2"].drop("area_m2"))
    print(f"VIF da área: sem interação {vif_sem_inter['area_m2']:.2f} | forma separada: area_apto "
          f"{vif_reparam['area_apto']:.2f}, area_casa {vif_reparam['area_casa']:.2f} | "
          f"corr. da área nos aptos: {corr_apto.round(2).to_dict()}")
    vif_tab.to_csv(RESULTADOS / "vif_final.csv", encoding="utf-8-sig")
    print("\n=== VIF final (treino) ===")
    print(vif_tab.round(2).to_string())

    mi = lambda v, _: f"{v/1e6:.1f} mi".replace(".", ",")


    def virgula_eixos(fig):
        """Troca o ponto decimal por vírgula nos eixos que usam o formatador numérico padrão."""
        for ax in fig.axes:
            for eixo in (ax.xaxis, ax.yaxis):
                if isinstance(eixo.get_major_formatter(), ScalarFormatter):
                    eixo.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))

    # Figura 5. Real × previsto no teste
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    lim = [0, max(y_te.max(), prev_lin.max(), prev_log.max()) * 1.05]
    for ax, p, tit in ((axes[0], prev_lin, "Linear (principal)"), (axes[1], prev_log, "Log-log + Duan (complementar)")):
        ax.plot(lim, lim, color=TINTA, lw=1.5, ls="--", label="previsão perfeita")
        ax.scatter(y_te, p, s=34, color=AZUL, alpha=0.75, edgecolor=FUNDO, linewidth=0.8)
        m = metricas(y_te, p)
        ax.text(0.03, 0.96, f"R² = {br(m['R2'], 2)}\nMAE = R$ {br(m['MAE']/1e3)} mil", transform=ax.transAxes,
                va="top", color=TINTA_2)
        ax.set(title=tit, xlabel="Preço real (R$)", xlim=lim, ylim=lim)
        ax.xaxis.set_major_formatter(mi)
        ax.yaxis.set_major_formatter(mi)
    axes[0].set_ylabel("Preço previsto (R$)")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle(f"Real × previsto no conjunto de teste (n = {len(teste)})", x=0.01, ha="left", color=TINTA_2)
    virgula_eixos(fig)
    fig.tight_layout()
    fig.savefig(PASTA / "5_teste_real_x_previsto.png", dpi=150)
    plt.close(fig)

    # Figura 6. Resíduos × previsto (ajuste no treino)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].scatter(fit_lin.fittedvalues, fit_lin.resid, s=24, color=AZUL, alpha=0.6, edgecolor=FUNDO, linewidth=0.8)
    axes[0].set(title="Linear – resíduos × previsto", xlabel="Preço previsto (R$)", ylabel="Resíduo (R$)")
    axes[0].xaxis.set_major_formatter(mi)
    axes[0].yaxis.set_major_formatter(mi)
    axes[1].scatter(fit_log.fittedvalues, fit_log.resid, s=24, color=AZUL, alpha=0.6, edgecolor=FUNDO, linewidth=0.8)
    axes[1].set(title="Log-log – resíduos × previsto", xlabel="log(preço previsto)", ylabel="Resíduo (log)")
    for ax, b in zip(axes, bp["p_valor_LM"]):
        ax.axhline(0, color=TINTA, lw=1.5, ls="--")
        ax.text(0.98, 0.96, f"Breusch-Pagan p {p_txt(b)}", transform=ax.transAxes, ha="right", va="top", color=TINTA_2)
    virgula_eixos(fig)
    fig.tight_layout()
    fig.savefig(PASTA / "6_residuos_x_previsto.png", dpi=150)
    plt.close(fig)

    # Figura 7. QQ-plot dos resíduos padronizados
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, fit, tit in ((axes[0], fit_lin, "Linear"), (axes[1], fit_log, "Log-log")):
        rp = fit.get_influence().resid_studentized_internal
        (teo, obs), (incl, inter, _) = stats.probplot(rp, dist="norm")
        ax.plot(teo, incl * teo + inter, color=TINTA, lw=1.5, ls="--")
        ax.scatter(teo, obs, s=22, color=AZUL, alpha=0.7, edgecolor=FUNDO, linewidth=0.8)
        sw = stats.shapiro(rp).pvalue
        ax.text(0.03, 0.96, f"Shapiro-Wilk p {p_txt(sw)}", transform=ax.transAxes, va="top", color=TINTA_2)
        ax.set(title=f"QQ-plot dos resíduos – {tit}", xlabel="Quantis teóricos (normal)")
    axes[0].set_ylabel("Resíduos padronizados")
    virgula_eixos(fig)
    fig.tight_layout()
    fig.savefig(PASTA / "7_qqplot_residuos.png", dpi=150)
    plt.close(fig)
    print(f"\n-> graficos/5_teste_real_x_previsto.png, 6_residuos_x_previsto.png, 7_qqplot_residuos.png | "
          f"resultados/vif_final.csv, breusch_pagan.csv")

    # robustez: distância de Cook (linear)
    cook = fit_lin.get_influence().cooks_distance[0]
    lim_cook = 4 / len(treino)
    infl = cook > lim_cook
    fit_sem = sm.OLS(y_tr[~infl], Xl_tr[~infl]).fit(cov_type="HC3")
    fit_log_sem = sm.OLS(np.log(y_tr[~infl]), Xg_tr[~infl]).fit(cov_type="HC3")
    r2_cook = {}
    for rot, excluir in (("com todos", np.zeros(len(treino), bool)), ("sem influentes", infl)):
        reps = []
        for r in range(REPETICOES):
            prev = np.full(len(treino), np.nan)
            for tr, te in KFold(5, shuffle=True, random_state=SEED + r).split(treino):
                tr = tr[~excluir[tr]]  # treina sem os influentes; avalia em TODAS as linhas
                prev[te] = LinearRegression().fit(treino.loc[tr, X_LIN], y_tr[tr]).predict(treino.loc[te, X_LIN])
            reps.append((r2_score(y_tr, prev), mean_absolute_error(y_tr, prev)))
        r2_cook[rot] = np.mean(reps, axis=0)
    print(f"Cook – CV do linear avaliada em todas as linhas: com todos R²={r2_cook['com todos'][0]:.3f} "
          f"MAE={reais(r2_cook['com todos'][1])} | sem influentes R²={r2_cook['sem influentes'][0]:.3f} "
          f"MAE={reais(r2_cook['sem influentes'][1])}")
    print(f"Cook > 4/n: {infl.sum()} imóveis | m² apto {reais(m2_apto[0])} -> {reais(fit_sem.params['area_m2'])}")

    # Quanto o R² em R$ do log-log na CV depende dos 2 maiores erros de cada repetição (mesmo método da tabela de CV)
    r2_com, r2_sem, piores = [], [], []
    # linear × log-log nos MESMOS imóveis, sem os 2 maiores erros do log-log: (R² lin, R² log, RMSE lin, RMSE log)
    mesmos = []
    for r in range(REPETICOES):
        prev = np.full(len(treino), np.nan)
        prev_l = np.full(len(treino), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=SEED + r).split(treino):
            ytr = np.log(y_tr.values[tr])
            m = LinearRegression().fit(treino.loc[tr, X_LOG], ytr)
            prev[te] = (np.exp(m.predict(treino.loc[te, X_LOG]))
                        * np.mean(np.exp(ytr - m.predict(treino.loc[tr, X_LOG]))))
            prev_l[te] = LinearRegression().fit(treino.loc[tr, X_LIN], y_tr.values[tr]).predict(treino.loc[te, X_LIN])
        erro = np.abs(prev - y_tr.values)
        top2 = np.argsort(-erro)[:2]
        manter = np.ones(len(treino), bool)
        manter[top2] = False
        r2_com.append(r2_score(y_tr, prev))
        r2_sem.append(r2_score(y_tr[manter], prev[manter]))
        mesmos.append((r2_score(y_tr[manter], prev_l[manter]), r2_score(y_tr[manter], prev[manter]),
                       np.sqrt(mean_squared_error(y_tr[manter], prev_l[manter])),
                       np.sqrt(mean_squared_error(y_tr[manter], prev[manter]))))
        piores.append(top2[0])
    r2_log_com, r2_log_sem = np.mean(r2_com), np.mean(r2_sem)
    r2m_lin, r2m_log, rmsem_lin, rmsem_log = np.mean(mesmos, axis=0)
    i_pior = max(set(piores), key=piores.count)
    n_maior_erro = piores.count(i_pior)
    c = treino.loc[i_pior]
    maior_erro_log = f"uma {c.tipo} de {br(c.area_m2)} m² no {c.bairro}"
    print(f"Log-log na CV: R² {r2_log_com:.3f} -> {r2_log_sem:.3f} sem os 2 maiores erros por repetição | "
          f"maior erro: {maior_erro_log} ({n_maior_erro}/{REPETICOES}) | mesmos imóveis: R² linear {r2m_lin:.3f} "
          f"vs. log-log {r2m_log:.3f}")

    # Perfil dos influentes (Cook) e sensibilidade de banheiros nos dois modelos
    infl_preco_med, demais_preco_med = treino.loc[infl, "preco"].median(), treino.loc[~infl, "preco"].median()
    infl_banh, demais_banh = treino.loc[infl, "banheiros"].mean(), treino.loc[~infl, "banheiros"].mean()
    infl_acima_p75 = int((treino.loc[infl, "preco"] > treino["preco"].quantile(0.75)).sum())
    var_banh_lin = (fit_sem.params["banheiros"] / fit_lin.params["banheiros"] - 1) * 100
    var_banh_log = (fit_log_sem.params["banheiros"] / fit_log.params["banheiros"] - 1) * 100

    return {k: v for k, v in locals().items() if k != "ctx"}


if __name__ == "__main__":
    raise SystemExit("Rode `python src/modelo_final.py`, que chama os diagnósticos e o resumo.")
