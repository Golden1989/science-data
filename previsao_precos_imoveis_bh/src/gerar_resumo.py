"""Etapa 4 – escreve relatorio/resumo_etapa4.md (chamado por src/modelo_final.py).

Só formata texto e tabelas com os valores já calculados em modelo_final.py e diagnosticos.py; não ajusta modelos.

Uso: não é executado sozinho; rode `python src/modelo_final.py`.
"""
import numpy as np
import pandas as pd


def escrever(ctx):
    """Monta o resumo em Markdown e grava em relatorio/resumo_etapa4.md."""
    # Valores calculados em modelo_final.py e diagnosticos.py
    abaixo_teste, ADOTA_LANC, area_casa_tip = ctx.abaixo_teste, ctx.ADOTA_LANC, ctx.area_casa_tip
    bench, bp, br, casa_vs_apto = ctx.bench, ctx.bp, ctx.br, ctx.casa_vs_apto
    casa_vs_apto_log, coef_lin, coef_log = ctx.casa_vs_apto_log, ctx.coef_lin, ctx.coef_log
    corr_apto, cv_tab, demais_banh = ctx.corr_apto, ctx.cv_tab, ctx.demais_banh
    demais_preco_med, df, duan, DUMMIES = ctx.demais_preco_med, ctx.df, ctx.duan, ctx.DUMMIES
    el_apto, el_casa, f0_log, fit_lin = ctx.el_apto, ctx.el_casa, ctx.f0_log, ctx.fit_lin
    fit_log, fit_log_sem, fit_sem, infl = ctx.fit_log, ctx.fit_log_sem, ctx.fit_sem, ctx.infl
    infl_acima_p75, infl_banh, infl_preco_med = ctx.infl_acima_p75, ctx.infl_banh, ctx.infl_preco_med
    lanc_atual, lanc_novo, lanc_tab, lim_cook = ctx.lanc_atual, ctx.lanc_novo, ctx.lanc_tab, ctx.lim_cook
    m2_apto, m2_casa, maior_erro_log, mg_apto = ctx.m2_apto, ctx.m2_casa, ctx.maior_erro_log, ctx.mg_apto
    mg_casa, n_maior_erro, n_reg, p_txt, piso = ctx.mg_casa, ctx.n_maior_erro, ctx.n_reg, ctx.p_txt, ctx.piso
    pv, r2_cook, r2_log_com, r2_log_sem = ctx.pv, ctx.r2_cook, ctx.r2_log_com, ctx.r2_log_sem
    r2m_lin, r2m_log, reais, REF_REGIONAL = ctx.r2m_lin, ctx.r2m_log, ctx.reais, ctx.REF_REGIONAL
    reg, RELATORIO, REPETICOES, rmsem_lin = ctx.reg, ctx.RELATORIO, ctx.REPETICOES, ctx.rmsem_lin
    rmsem_log, SEED, teste, teste_antes = ctx.rmsem_log, ctx.SEED, ctx.teste, ctx.teste_antes
    teste_tab, treino, var_banh_lin = ctx.teste_tab, ctx.treino, ctx.var_banh_lin
    var_banh_log, vif_reparam, vif_sem_inter = ctx.var_banh_log, ctx.vif_reparam, ctx.vif_sem_inter
    vif_tab = ctx.vif_tab

    sig = lambda p: "sim" if p < 0.05 else "**não**"
    pct = lambda b: (np.exp(b) - 1) * 100
    cv_lin = cv_tab[(cv_tab.modelo == "linear") & (cv_tab.estimador == "OLS")].iloc[0]
    cv_log = cv_tab[(cv_tab.modelo == "log-log") & (cv_tab.estimador == "OLS")].iloc[0]
    t_lin, t_log = teste_tab.iloc[0], teste_tab.iloc[1]
    corr_q_area = treino["quartos"].corr(treino["area_m2"])
    buritis = (df["bairro"] == "Buritis").mean() * 100
    duvidosos = reg[reg["duvidoso"] == "sim"]
    n_duv = df["bairro"].isin(duvidosos["bairro"]).sum()
    maior_casa = df.loc[df["area_m2"].idxmax()]

    L = []
    w = L.append
    w("# Etapa 4 – Modelagem: resumo do modelo final\n")
    w("_Gerado automaticamente por `src/gerar_resumo.py` (chamado por `src/modelo_final.py`). Números do treino (n = "
      f"{len(treino)}) e do teste (n = {len(teste)}), separados 80/20 com seed {SEED}._\n")

    w("## 1. Especificação\n")
    w("**Modelo principal (linear, OLS):**\n")
    w("```\npreco ~ area_m2 + area_m2×casa + casa + quartos + banheiros + vagas + regional"
      + (" + lancamento" if ADOTA_LANC else "") + "\n```\n")
    w("**Modelo complementar (log-log, OLS + smearing de Duan):** mesmas variáveis, com `log(preco)` e `log(area_m2)`; "
      "usado para ler os efeitos em percentual.\n")
    w("- `casa`: 1 para casa/sobrado; **cobertura conta como apartamento** (a elasticidade da área da cobertura não "
      "diferiu da do apartamento, p = 0,50).")
    w(f"- `regional`: regional administrativa de BH (9 categorias, referência = **{REF_REGIONAL}**), a partir de "
      "`regionais_bh.csv`.")
    n_lanc = int(df["lancamento"].sum())
    redondo = (df["preco"] % 10000 == 0).groupby(df["lancamento"]).mean()  # fração de preços múltiplos de R$ 10 mil
    if ADOTA_LANC:
        w(f"- `lancamento`: 1 quando o link do anúncio é de `/imoveis-lancamentos/` ({n_lanc} imóveis, todos "
          f"apartamentos; {int(treino['lancamento'].sum())} no treino). **Variável identificada depois da primeira "
          "avaliação no teste**; entrou porque melhorou R², RMSE e MAE na CV do treino (seção 2).")
    w("- Pontos influentes (Cook > 4/n) **mantidos**; ver robustez na seção 5.")
    w("- Erros-padrão robustos **HC3**, porque o teste de Breusch-Pagan indica heterocedasticidade (seção 4).")
    w("- Escolhas feitas por validação cruzada 5-fold × 3 repetições **só no treino**. Critério definido antes do "
      "teste: desempenho em R$ (R², RMSE, MAE)."
      + (" O teste foi avaliado duas vezes: na especificação original e, depois, com `lancamento`; as duas "
         "avaliações estão na seção 2.\n"
         if ADOTA_LANC else " O teste foi avaliado uma única vez, sem alterar a escolha.\n"))
    w("Caminho até a especificação (scripts `analise_casas_cook.py` e `etapa4_itens3_4.py`): a interação área × casa "
      "entrou porque nas casas a área informada se comporta como área de terreno; a regional venceu o bairro com "
      "limiar de 5 ou 3 anúncios e o target encoding em R² e MAE na CV, com coeficientes mais estáveis.\n")

    w("## 2. Métricas\n")
    w("### Validação cruzada no treino (5-fold × 3)\n")
    w("| Modelo | Estimador | R² | RMSE | MAE | Erro mediano |")
    w("|---|---|---|---|---|---|")
    for _, r in cv_tab.iterrows():
        w(f"| {r.modelo} | {r.estimador} | {br(r.R2, 3)} ± {br(r.R2_dp, 3)} | {reais(r.RMSE)} | {reais(r.MAE)} | "
          f"{reais(r.MedAE)} |")
    w(f"| *referência* | Random Forest | {br(bench['R2'], 3)} ± {br(bench['R2_dp'], 3)} | {reais(bench['RMSE'])} | "
      f"{reais(bench['MAE'])} | {reais(bench['MedAE'])} |")
    ols_l = cv_tab[(cv_tab.modelo == "linear") & (cv_tab.estimador == "OLS")].iloc[0]
    reg_l = cv_tab[(cv_tab.modelo == "linear") & (cv_tab.estimador != "OLS")].set_index("estimador")
    reg_g = cv_tab[(cv_tab.modelo == "log-log") & (cv_tab.estimador != "OLS")]
    ols_g = cv_tab[(cv_tab.modelo == "log-log") & (cv_tab.estimador == "OLS")].iloc[0]
    w(f"\n**OLS mantido.** No linear, a diferença de R² para o OLS foi de {br(reg_l.loc['Ridge', 'R2'] - ols_l.R2, 3)} "
      f"(Ridge) e {br(reg_l.loc['Lasso', 'R2'] - ols_l.R2, 3)} (Lasso), com MAE de {reais(reg_l.loc['Ridge', 'MAE'])} "
      f"e {reais(reg_l.loc['Lasso', 'MAE'])} contra {reais(ols_l.MAE)}. Diferenças dessa ordem não justificam a troca: "
      f"o desvio entre repetições ({br(ols_l.R2_dp, 3)}) só mede a variação de embaralhar os mesmos dados e subestima "
      "a incerteza real. Mantido o OLS por parcimônia e interpretabilidade. No log-log, Ridge e Lasso "
      + ("foram piores que o OLS." if (reg_g.R2 < ols_g.R2).all() else "não superaram o OLS de forma clara."))
    w(f"\n**Referência não linear:** um Random Forest com as mesmas variáveis e a mesma CV chega a R² "
      f"{br(bench['R2'], 3)} e MAE {reais(bench['MAE'])}, contra {br(ols_l.R2, 3)} e {reais(ols_l.MAE)} do linear: o "
      f"modelo linear perde {br(bench['R2'] - ols_l.R2, 3)} de R² e {reais(ols_l.MAE - bench['MAE'])} de MAE, em troca "
      "de coeficientes interpretáveis. Ele é só referência e não substitui o modelo principal.\n")

    if ADOTA_LANC:
        w("### Variável `lancamento` (identificada depois da primeira avaliação no teste)\n")
        w("Depois da primeira avaliação no teste, notamos que os links com `/imoveis-lancamentos/` têm preço por m² "
          f"muito maior (mediana de {reais(df.loc[df.lancamento == 1, 'preco_m2'].median())}/m², contra "
          f"{reais(df.loc[df.lancamento == 0, 'preco_m2'].median())}/m² nos demais). A variável foi testada com o "
          "mesmo protocolo (CV só no treino, OLS) e a regra fixada antes de rodar: entra se o linear melhorar R², RMSE "
          "e MAE.\n")
        w("| Modelo | Variáveis | R² (CV) | RMSE (CV) | MAE (CV) | Erro mediano (CV) |")
        w("|---|---|---|---|---|---|")
        for _, r in lanc_tab.iterrows():
            w(f"| {r.modelo} | {r.variaveis} | {br(r.R2, 3)} | {reais(r.RMSE)} | {reais(r.MAE)} | {reais(r.MedAE)} |")
        w("\nO linear melhorou nas três métricas da regra"
          + (f" (o erro mediano piorou de {reais(lanc_atual.MedAE)} para {reais(lanc_novo.MedAE)})"
             if lanc_novo.MedAE > lanc_atual.MedAE else "")
          + ", e a variável foi adotada nos dois modelos.\n")

    w("### Teste\n")
    w("| Avaliação | Modelo | R² | RMSE | MAE | Erro mediano |")
    w("|---|---|---|---|---|---|")
    if ADOTA_LANC:
        for _, r in teste_antes.iterrows():
            w(f"| 1ª (sem `lancamento`) | {r.modelo} | {br(r.R2, 3)} | {reais(r.RMSE)} | {reais(r.MAE)} | "
              f"{reais(r.MedAE)} |")
    for _, r in teste_tab.iterrows():
        rot = "2ª (com `lancamento`)" if ADOTA_LANC else "única"
        w(f"| {rot} | {r.modelo} | {br(r.R2, 3)} | {reais(r.RMSE)} | {reais(r.MAE)} | {reais(r.MedAE)} |")
    if ADOTA_LANC:
        w("\nA 2ª avaliação **não é uma estimativa independente**: a variável foi encontrada depois de o teste já ter "
          "sido usado uma vez, e a melhora no teste deve ser lida com essa ressalva. A melhora na CV do treino, onde a "
          "decisão foi tomada, é a evidência principal.")
    cv_lin_ols = cv_tab[(cv_tab.modelo == "linear") & (cv_tab.estimador == "OLS")].iloc[0]
    cv_log_ols = cv_tab[(cv_tab.modelo == "log-log") & (cv_tab.estimador == "OLS")].iloc[0]
    ganha_lin = [n for n, a, b in (("R²", cv_lin_ols.R2, cv_log_ols.R2), ("RMSE", -cv_lin_ols.RMSE, -cv_log_ols.RMSE),
                                    ("MAE", -cv_lin_ols.MAE, -cv_log_ols.MAE)) if a > b]
    ganha_log = [n for n in ("R²", "RMSE", "MAE") if n not in ganha_lin]
    w(f"\n**Linear × log-log.** Na CV do treino o resultado se divide: o linear vence em {' e '.join(ganha_lin)} "
      f"(R² {br(cv_lin_ols.R2, 3)} vs. {br(cv_log_ols.R2, 3)}; RMSE {reais(cv_lin_ols.RMSE)} vs. "
      f"{reais(cv_log_ols.RMSE)}), e o log-log vence em {' e '.join(ganha_log)} (MAE {reais(cv_log_ols.MAE)} vs. "
      f"{reais(cv_lin_ols.MAE)}). No teste, o log-log fica à frente nas três métricas (R² {br(t_log.R2, 3)} vs. "
      f"{br(t_lin.R2, 3)}; MAE {reais(t_log.MAE)} vs. {reais(t_lin.MAE)}). O critério fixado antes do teste (R², RMSE "
      "e MAE em R$) não previa um resultado dividido; **mantivemos o linear como principal** porque ele vence nas duas "
      "métricas que penalizam erros grandes (R² e RMSE), e essa decisão não foi revista depois de ver o teste.")
    w(f"\nA desvantagem do log-log nessas duas métricas continua vindo de 1–2 casas grandes: {maior_erro_log} é o "
      f"maior erro em {n_maior_erro} de {REPETICOES} repetições. Tirando os 2 maiores erros do log-log de cada "
      f"repetição e comparando os dois modelos **nos mesmos imóveis**, o log-log passa à frente também em R² "
      f"({br(r2m_log, 3)} vs. {br(r2m_lin, 3)}) e RMSE ({reais(rmsem_log)} vs. {reais(rmsem_lin)}). O teste "
      f"({len(teste)} imóveis, maior área {br(teste['area_m2'].max())} m²) não tem casos desse tipo, o que explica a "
      f"inversão.")
    w(f"\nFator de Duan (treino): {br(duan, 3)}, ou seja, sem a correção o log-log subestimaria os preços em cerca de "
      f"{br((duan - 1) * 100, 1)}%.\n")

    w("## 3. Coeficientes interpretados\n")
    w("Sempre **mantidas as demais variáveis constantes**. Significância a 5% com erros-padrão HC3.\n")
    w("### Modelo principal (linear, em R$)\n")
    w("| Efeito | Estimativa | p-valor | Significativo? |")
    w("|---|---|---|---|")
    w(f"| Cada m² adicional – **apartamento/cobertura** | {reais(m2_apto[0])} | {pv(m2_apto[2])} | {sig(m2_apto[2])} |")
    w(f"| Cada m² adicional – **casa** | {reais(m2_casa[0])} | {pv(m2_casa[2])} | {sig(m2_casa[2])} |")
    w(f"| Casa × apartamento, ambos com {br(area_casa_tip)} m² (área mediana das casas) | {reais(casa_vs_apto[0])} | "
      f"{pv(casa_vs_apto[2])} | {sig(casa_vs_apto[2])} |")
    for v, nome in (("banheiros", "+1 banheiro"), ("vagas", "+1 vaga"), ("quartos", "+1 quarto")):
        r = coef_lin.loc[v]
        w(f"| {nome} | {reais(r.coef)} | {pv(r.p_valor)} | {sig(r.p_valor)} |")
    for d in DUMMIES:
        r = coef_lin.loc[d]
        w(f"| {d[4:]} vs. {REF_REGIONAL} (n treino = {n_reg[d[4:]]}) | {reais(r.coef)} | {pv(r.p_valor)} | "
          f"{sig(r.p_valor)} |")
    if ADOTA_LANC:
        r = coef_lin.loc["lancamento"]
        w(f"| Lançamento vs. demais | {reais(r.coef)} | {pv(r.p_valor)} | {sig(r.p_valor)} |")
    w("")
    w(f"- **Área:** no linear, o m² sai {reais(m2_apto[0])} no apartamento (p = {pv(m2_apto[2])}) e "
      f"{reais(m2_casa[0])} na casa (p = {pv(m2_casa[2])}). A estimativa em R$/m² usada na interpretação vem do "
      "log-log (abaixo), que lida melhor com a heterocedasticidade.")
    w(f"- Um banheiro a mais: {reais(coef_lin.loc['banheiros', 'coef'])}; uma vaga a mais: "
      f"{reais(coef_lin.loc['vagas', 'coef'])}.")
    top = coef_lin.loc[DUMMIES, "coef"].sort_values()
    w(f"- Regional: o imóvel no **{top.index[-1][4:]}** custa {reais(top.iloc[-1])} a mais que um idêntico na "
      f"{REF_REGIONAL}; no **{top.index[0][4:]}**, {reais(abs(top.iloc[0]))} a menos.")
    w(f"- O coeficiente isolado de `casa` ({reais(coef_lin.loc['casa', 'coef'])}) é a diferença com área zero e não "
      "tem leitura própria; a comparação útil é a da linha \"Casa × apartamento\" acima.\n")

    w("### Modelo complementar (log-log, em %)\n")
    w("| Efeito | Estimativa | p-valor | Significativo? |")
    w("|---|---|---|---|")
    w(f"| +1% de área – **apartamento/cobertura** | +{br(el_apto[0], 2)}% | {pv(el_apto[2])} | {sig(el_apto[2])} |")
    w(f"| +1% de área – **casa** | +{br(el_casa[0], 2)}% | {pv(el_casa[2])} | {sig(el_casa[2])} |")
    w(f"| Casa × apartamento com {br(area_casa_tip)} m² | {br(pct(casa_vs_apto_log[0]), 1)}% | "
      f"{pv(casa_vs_apto_log[2])} | {sig(casa_vs_apto_log[2])} |")
    for v, nome in (("banheiros", "+1 banheiro"), ("vagas", "+1 vaga"), ("quartos", "+1 quarto")):
        r = coef_log.loc[v]
        w(f"| {nome} | {br(r['efeito_%'], 1)}% | {pv(r.p_valor)} | {sig(r.p_valor)} |")
    for d in DUMMIES:
        r = coef_log.loc[d]
        w(f"| {d[4:]} vs. {REF_REGIONAL} | {br(r['efeito_%'], 1)}% | {pv(r.p_valor)} | {sig(r.p_valor)} |")
    if ADOTA_LANC:
        r = coef_log.loc["lancamento"]
        w(f"| Lançamento vs. demais | {br(r['efeito_%'], 1)}% | {pv(r.p_valor)} | {sig(r.p_valor)} |")
    w("")
    w(f"- **+1% de área → +{br(el_apto[0], 2)}% no preço do apartamento e +{br(el_casa[0], 2)}% no da casa.**")
    w(f"- **Cada m² adicional eleva o preço do apartamento em cerca de {reais(mg_apto['ef'])}** (IC 95%: "
      f"{reais(mg_apto['ic'][0])} a {reais(mg_apto['ic'][1])}), mantidas as demais constantes. Cálculo: elasticidade × "
      f"preço mediano / área mediana dos apartamentos do treino ({br(el_apto[0], 3)} × {reais(mg_apto['P'])} / "
      f"{br(mg_apto['A'])} m²), com IC pelo método delta tratando as medianas como fixas. Na casa, o mesmo cálculo dá "
      f"{reais(mg_casa['ef'])}/m² (IC 95%: {reais(mg_casa['ic'][0])} a {reais(mg_casa['ic'][1])}), não significativo. "
      f"Para comparação, o coeficiente do modelo linear é {reais(m2_apto[0])}/m² (p = {br(m2_apto[2], 2)}).")
    w(f"- **+1 banheiro → {br(coef_log.loc['banheiros', 'efeito_%'], 1)}%; +1 vaga → "
      f"{br(coef_log.loc['vagas', 'efeito_%'], 1)}%.**")
    w(f"- **Centro-Sul ≈ {br(coef_log.loc['reg_Centro-Sul', 'efeito_%'], 0)}% vs. {REF_REGIONAL}** para um imóvel "
      "com as mesmas características" + ("." if not ADOTA_LANC else
      f"; sem `lancamento`, o prêmio estimado era de {br(pct(f0_log.params['reg_Centro-Sul']), 0)}%, porque parte dos "
      f"lançamentos ({int(df.loc[df.lancamento == 1, 'regional'].eq('Centro-Sul').sum())} de {n_lanc}) fica no "
      "Centro-Sul e o efeito deles era atribuído à regional."))
    if ADOTA_LANC:
        w(f"- **Lançamento:** com as demais variáveis iguais, o anúncio de lançamento custa "
          f"**{br(coef_log.loc['lancamento', 'efeito_%'], 0)}% a mais** no log-log "
          f"({reais(coef_lin.loc['lancamento', 'coef'])} "
          "no linear). **Ressalva:** o coeficiente mistura o ágio do imóvel novo com um possível erro de medida. Nos "
          "lançamentos, o cartão mostra o preço \"a partir de\" da construtora (só "
          f"{br(redondo[1] * 100)}% são múltiplos de R$ 10 mil, contra {br(redondo[0] * 100)}% nos demais) junto com a "
          "área e os quartos de uma unidade específica, que nem sempre é a mais barata; um lançamento chegou a ter "
          "área em faixa nos dados brutos (107 a 218 m²). Não dá para separar as duas coisas com estes dados.")
    w("")

    w("### Coeficientes não significativos e por quê\n")
    reg_ns = {m: [d[4:] for d in DUMMIES if t.loc[d, "p_valor"] >= 0.05]
              for m, t in (("linear", coef_lin), ("log-log", coef_log))}
    w(f"- **Quartos** (p = {pv(coef_lin.loc['quartos', 'p_valor'])} no linear e "
      f"{pv(coef_log.loc['quartos', 'p_valor'])} no log-log): colinear com a área (correlação {br(corr_q_area, 2)} no "
      "treino). Com a área fixa, um quarto a mais só divide o mesmo espaço em cômodos menores; o efeito próprio é "
      "pequeno e impreciso.")
    ns_area = [v for v in ("area_m2", "area_x_casa") if coef_lin.loc[v, "p_valor"] >= 0.05]
    if ns_area:
        lista_ns = ", ".join(f"`{v}` p = {pv(coef_lin.loc[v, 'p_valor'])}" for v in ns_area)
        w(f"- **Área no linear** ({lista_ns}): entre os apartamentos a área tem correlação de {br(corr_apto.min(), 2)} "
          f"a {br(corr_apto.max(), 2)} com quartos, banheiros e vagas, e a heterocedasticidade em R$ infla os "
          "erros-padrão robustos.")
    if casa_vs_apto[2] >= 0.05:
        w(f"- **Casa × apartamento no linear** ({reais(casa_vs_apto[0])} com {br(area_casa_tip)} m², p = "
          f"{pv(casa_vs_apto[2])}; o nível isolado de `casa` tem p = {pv(coef_lin.loc['casa', 'p_valor'])} e só tem "
          f"leitura junto da área): poucas casas no treino (n = {int(treino['casa'].sum())}) e muita dispersão de "
          f"preço em R$. No log-log a diferença é significativa ({br(pct(casa_vs_apto_log[0]), 1)}%, p "
          f"{p_txt(casa_vs_apto_log[2])}).")
    w(f"- **Área da casa no log-log** (elasticidade {br(el_casa[0], 2)}, p = {pv(el_casa[2])}): o preço da casa quase "
      "não varia com a área informada, coerente com a área de terreno (seção 6).")
    w(f"- **Regionais** sem diferença significativa da {REF_REGIONAL}: {', '.join(reg_ns['linear'])} no linear; "
      f"{', '.join(reg_ns['log-log'])} no log-log. São regionais com poucos imóveis no treino ("
      + ", ".join(f"{r} n = {n_reg[r]}" for r in sorted(set(reg_ns['linear']) | set(reg_ns['log-log']),
                                                         key=lambda r: n_reg[r]))
      + ") ou com preço próximo ao da referência.\n")

    w("## 4. Diagnósticos\n")
    w("| Modelo | Breusch-Pagan LM | p-valor | Conclusão |")
    w("|---|---|---|---|")
    for _, r in bp.iterrows():
        w(f"| {r.modelo} | {br(r.LM, 1)} | {pv(r.p_valor_LM)} | {r.conclusao} |")
    w("")
    w("VIF (treino), com e sem centrar a área:\n")
    w("| Variável | VIF linear | linear, área centrada | VIF log-log | log-log, área centrada |")
    w("|---|---|---|---|---|")
    for v, r in vif_tab.iterrows():
        w(f"| {v} | {br(r.VIF_linear, 2)} | {br(r.VIF_linear_area_centrada, 2)} | {br(r.VIF_loglog, 2)} | "
          f"{br(r.VIF_loglog_area_centrada, 2)} |")
    w(f"\nO VIF de `area_m2` no linear ({br(vif_tab.loc['area_m2', 'VIF_linear'], 2)}) não cai ao centrar a área, mas "
      "também **não vem de quartos, banheiros e vagas**: sem a interação ele é "
      f"{br(vif_sem_inter['area_m2'], 2)}, e na forma equivalente com a área separada por tipo (`area_apto`, "
      f"`area_casa`, mesmo ajuste) o VIF da área do apartamento é {br(vif_reparam['area_apto'], 2)}. O valor alto é "
      "estrutural: as casas concentram as maiores áreas, então `area_m2` e `area_x_casa` andam juntas. Entre os "
      f"apartamentos, a área tem colinearidade moderada com quartos, banheiros e vagas (correlação de "
      f"{br(corr_apto.min(), 2)} a {br(corr_apto.max(), 2)}). Regionais, banheiros, vagas, quartos"
      + (" e `lancamento`" if ADOTA_LANC else "") + " ficam abaixo de 3.")
    w("\nGráficos: `graficos/5_teste_real_x_previsto.png`, `graficos/6_residuos_x_previsto.png`, "
      "`graficos/7_qqplot_residuos.png`.\n")

    COOK_LIN = ["area_m2", "area_x_casa", "banheiros", "vagas", "reg_Centro-Sul"]
    var_cook = pd.Series({v: (fit_sem.params[v] / fit_lin.params[v] - 1) * 100 for v in COOK_LIN})
    mais_sensivel = var_cook.abs().idxmax()
    mesmo_sinal = all(np.sign(fit_sem.params[v]) == np.sign(fit_lin.params[v]) for v in COOK_LIN) and \
        all(np.sign(fit_log_sem.params[v]) == np.sign(fit_log.params[v])
            for v in ("log_area", "log_area_x_casa", "banheiros", "vagas", "reg_Centro-Sul"))
    assert mesmo_sinal, "algum sinal mudou sem os influentes: revisar o texto da seção 5"
    w("## 5. Robustez: pontos influentes\n")
    w(f"{infl.sum()} imóveis do treino têm distância de Cook > 4/n ({br(lim_cook, 4)}) no modelo linear. Eles foram "
      "**mantidos**. Reajustando sem eles:\n")
    w("| Coeficiente | Com todos | Sem influentes |")
    w("|---|---|---|")
    for v in ("area_m2", "area_x_casa", "banheiros", "vagas", "reg_Centro-Sul"):
        w(f"| {v} (linear) | {reais(fit_lin.params[v])} | {reais(fit_sem.params[v])} |")
    for v in ("log_area", "log_area_x_casa", "banheiros", "vagas", "reg_Centro-Sul"):
        w(f"| {v} (log-log) | {br(fit_log.params[v], 3)} | {br(fit_log_sem.params[v], 3)} |")
    w(f"\nNa CV do modelo linear (avaliada em todos os imóveis do treino), treinar sem os influentes leva o R² de "
      f"{br(r2_cook['com todos'][0], 3)} para {br(r2_cook['sem influentes'][0], 3)} e o MAE de "
      f"{reais(r2_cook['com todos'][1])} para {reais(r2_cook['sem influentes'][1])}. "
      f"O coeficiente mais sensível a esses pontos é `{mais_sensivel}` no linear ({br(var_cook[mais_sensivel], 0)}%); "
      "os sinais dos coeficientes da tabela não mudam. Os influentes ficam na base (decisão registrada) e a análise "
      "entra como robustez.\n")
    w(f"**Banheiros no linear depende de poucos imóveis caros.** Os {infl.sum()} influentes têm preço mediano de "
      f"{reais(infl_preco_med)} (contra {reais(demais_preco_med)} nos demais; {infl_acima_p75} deles acima do P75 do "
      f"treino) e {br(infl_banh, 1)} banheiros em média (contra {br(demais_banh, 1)}). Sem eles, o efeito de "
      f"+1 banheiro no linear cai {br(abs(var_banh_lin), 0)}% ({reais(fit_lin.params['banheiros'])} → "
      f"{reais(fit_sem.params['banheiros'])}, p {p_txt(fit_sem.pvalues['banheiros'])}), enquanto no log-log a variação "
      f"é menor ({br(abs(var_banh_log), 0)}% no coeficiente: {br(pct(fit_log.params['banheiros']), 1)}% → "
      f"{br(pct(fit_log_sem.params['banheiros']), 1)}% por banheiro, p {p_txt(fit_log_sem.pvalues['banheiros'])}). Em "
      "R$, os poucos imóveis caros com muitos banheiros pesam muito mais na reta do que na escala log; o efeito em % "
      "do log-log é a leitura mais estável.\n")

    w("## 6. Limitações\n")
    w(f"- **Área de terreno nas casas:** casas com área ≥ 300 m² e R$/m² abaixo do P10 das casas provavelmente "
      "informam a área do lote (padrão de BH = 360 m²), não a área construída. A interação área × casa absorve parte "
      f"disso (o m² da casa vale {reais(m2_casa[0])} contra {reais(m2_apto[0])} no apartamento), mas a variável "
      "continua medindo coisas diferentes em cada tipo.")
    w(f"- **Viés do Buritis:** {br(buritis, 1)}% da amostra é de um único bairro (Oeste), o que pesa no efeito da "
      "regional de referência e reflete a ordenação do site, não o mercado.")
    w(f"- **Mapeamento de regionais aproximado:** feito manualmente; {len(duvidosos)} bairros de divisa ou com nome "
      f"repetido ({n_duv} imóveis) estão marcados como duvidosos. Trocá-los para a regional alternativa quase não muda "
      "as métricas (R² log 0,637 → 0,651 na CV, teste feito antes da inclusão de `lancamento`).")
    if ADOTA_LANC:
        w(f"- **`lancamento` incompleta e posterior ao teste:** {int(df['link'].isna().sum())} imóveis vêm de cartões "
          "agrupados (\"Ver os N anúncios deste imóvel\"), sem link, e ficaram como não lançamento por falta de "
          f"informação (mediana de {reais(df.loc[df.link.isna(), 'preco_m2'].median())}/m², perto dos não "
          "lançamentos). A variável foi encontrada depois da primeira avaliação no teste, então a 2ª métrica de teste "
          "não é independente.")
    w(f"- **Previsões abaixo do mínimo:** o modelo linear não respeita a positividade do preço. No teste, "
      f"{abaixo_teste} de {len(teste)} previsões ficaram abaixo do menor preço do treino ({reais(piso)}); na CV, "
      f"{br(cv_lin.abaixo_min_por_rep, 1)} por repetição. O log-log não tem esse problema.")
    w(f"- **1–2 casas grandes derrubam o log-log em R$:** o `exp()` amplifica erros nos imóveis grandes, sobretudo "
      f"{maior_erro_log} (maior erro em {n_maior_erro} de {REPETICOES} repetições). Na CV, sem os 2 maiores erros de "
      f"cada repetição, o R² em R$ do log-log passaria de {br(r2_log_com, 2)} para {br(r2_log_sem, 2)}. Por isso o "
      "log-log é usado para interpretar efeitos em %, e o linear para prever em R$.")
    w(f"- **Amostra pequena e de anúncios:** {len(df)} imóveis, preço pedido (não de venda), coletados das 10 "
      "primeiras páginas do VivaReal em uma data; não é uma amostra aleatória do mercado de BH.")

    (RELATORIO / "resumo_etapa4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n-> relatorio/resumo_etapa4.md | resultados/coeficientes_final_linear.csv, coeficientes_final_loglog.csv")


if __name__ == "__main__":
    raise SystemExit("Rode `python src/modelo_final.py`, que chama os diagnósticos e o resumo.")
