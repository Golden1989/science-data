"""Etapa 4 – itens 3 e 4 (+ simplificação da interação área × tipo).

  0. Interação: completa (log_area × casa e × cobertura) vs. simplificada (só × casa; cobertura = apartamento)
  3. Smearing de Duan no modelo log (fator calculado dentro de cada fold); R² em escala log e em R$,
     com e sem smearing; comparação justa log × linear
  4. Codificação do bairro: limiar 5, limiar 3, regional administrativa, target encoding (mediana de
     log(preco_m2) do bairro, suavizada, calculada dentro do fold e com cross-fitting no treino)

Protocolo: mesma separação treino/teste (80/20, seed 42). Tudo é CV 5-fold × 3 repetições SÓ no treino;
o teste não é usado. Encoders e fatores de smearing são ajustados apenas nas linhas de treino de cada fold.

Entradas: dados/processados/imoveis_limpo.csv, dados/regionais_bh.csv
Saídas:   resultados/exploratorio/itens3_4_resultados.csv, resultados/exploratorio/item4_estabilidade.csv
Caminhos relativos à raiz do projeto; o script roda de qualquer pasta.

Uso: python src/exploratorio/etapa4_itens3_4.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

SEED = 42
REPETICOES = 3
SUAVIZACAO_TE = 5     # peso (em nº de anúncios) da mediana global no target encoding
REF_REGIONAL = "Oeste"  # regional de referência (dummy omitida)

# dados
RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "resultados" / "exploratorio"
SAIDA.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(RAIZ / "dados" / "processados" / "imoveis_limpo.csv", encoding="utf-8-sig")
df["tipo"] = df["tipo"].replace({"sobrado": "casa"})
df["log_area"] = np.log(df["area_m2"])
df["log_preco"] = np.log(df["preco"])
df["casa"] = (df["tipo"] == "casa").astype(float)
df["cobertura"] = (df["tipo"] == "cobertura").astype(float)

reg = pd.read_csv(RAIZ / "dados" / "regionais_bh.csv", encoding="utf-8").fillna("")
sem_regional = sorted(set(df["bairro"]) - set(reg["bairro"]))
if sem_regional:
    raise SystemExit(f"Bairros sem regional em regionais_bh.csv: {sem_regional}")
df = df.merge(reg[["bairro", "regional"]], on="bairro", how="left")

treino, _teste = train_test_split(df, test_size=0.2, random_state=SEED)
treino = treino.reset_index(drop=True)
print(f"Treino: {len(treino)} (CV 5×{REPETICOES} só aqui) | Teste: {len(_teste)} (intocado)")

print("\n=== Mapeamento bairro -> regional (regionais_bh.csv) ===")
cont = treino.groupby("regional").agg(imoveis=("bairro", "size"), bairros=("bairro", "nunique"))
print(cont.sort_values("imoveis", ascending=False).to_string())
duv = reg[reg["duvidoso"] == "sim"]
duv = duv.assign(imoveis_treino=duv["bairro"].map(treino["bairro"].value_counts()).fillna(0).astype(int))
print(f"\nCasos duvidosos ({len(duv)}):")
print(duv[["bairro", "regional", "imoveis_treino", "obs"]].to_string(index=False))


# matriz de desenho
def numericas(d, log, interacao):
    area = "log_area" if log else "area_m2"
    X = pd.DataFrame({area: d[area], "quartos": d["quartos"], "banheiros": d["banheiros"],
                      "vagas": d["vagas"], "casa": d["casa"]}, index=d.index)
    X[f"{area}_x_casa"] = d[area] * d["casa"]
    if interacao == "completa":
        X["cobertura"] = d["cobertura"]
        X[f"{area}_x_cobertura"] = d[area] * d["cobertura"]
    return X.astype(float)


def mapa_te(d):
    """Mediana suavizada de log(preco_m2) por bairro, a partir das linhas d."""
    alvo = np.log(d["preco_m2"])
    glob = alvo.median()
    g = alvo.groupby(d["bairro"]).agg(["median", "size"])
    return (g["size"] * g["median"] + SUAVIZACAO_TE * glob) / (g["size"] + SUAVIZACAO_TE), glob


def localizacao(tr, te, metodo, seed):
    """Variáveis de localização ajustadas SÓ em tr e aplicadas em tr e te."""
    if metodo.startswith("limiar"):
        k = int(metodo[-1])
        cont = tr["bairro"].value_counts()
        manter = sorted(cont[cont >= k].index)  # demais bairros = referência "Outros"
        f = lambda d: pd.DataFrame({f"bairro_{b}": (d["bairro"] == b).astype(float) for b in manter}, index=d.index)
        return f(tr), f(te)
    if metodo == "regional":
        cats = sorted(set(tr["regional"]) - {REF_REGIONAL})
        f = lambda d: pd.DataFrame({f"reg_{r}": (d["regional"] == r).astype(float) for r in cats}, index=d.index)
        return f(tr), f(te)
    if metodo == "target_enc":
        # Treino: cross-fitting (cada linha codificada por um mapa que não a viu) para não vazar o próprio alvo
        enc_tr = pd.Series(np.nan, index=tr.index)
        for i_in, i_out in KFold(5, shuffle=True, random_state=seed).split(tr):
            m, glob = mapa_te(tr.iloc[i_in])
            enc_tr.iloc[i_out] = tr["bairro"].iloc[i_out].map(m).fillna(glob).values
        m, glob = mapa_te(tr)
        enc_te = te["bairro"].map(m).fillna(glob)
        return pd.DataFrame({"te_log_pm2": enc_tr}), pd.DataFrame({"te_log_pm2": enc_te})
    raise ValueError(metodo)


def cv(dados, log, interacao, metodo):
    """OOF por repetição + coeficientes de cada fold."""
    reps, coefs = [], []
    for r in range(REPETICOES):
        seed = SEED + r
        oof = {k: np.full(len(dados), np.nan) for k in ("log", "rs_sem", "rs_duan", "smear")}
        for tr_i, te_i in KFold(5, shuffle=True, random_state=seed).split(dados):
            tr, te = dados.iloc[tr_i], dados.iloc[te_i]
            loc_tr, loc_te = localizacao(tr, te, metodo, seed)
            X_tr = numericas(tr, log, interacao).join(loc_tr)
            X_te = numericas(te, log, interacao).join(loc_te)
            X_tr = X_tr.loc[:, X_tr.std() > 0]  # dummy ausente no fold de treino
            X_te = X_te[X_tr.columns]
            y_tr = tr["log_preco"] if log else tr["preco"]
            m = LinearRegression().fit(X_tr, y_tr)
            p = m.predict(X_te)
            coefs.append(pd.Series(m.coef_, index=X_tr.columns))
            if log:
                smear = np.mean(np.exp(y_tr - m.predict(X_tr)))  # Duan: só resíduos do treino do fold
                oof["log"][te_i] = p
                oof["rs_sem"][te_i] = np.exp(p)
                oof["rs_duan"][te_i] = np.exp(p) * smear
                oof["smear"][te_i] = smear
            else:
                piso = tr["preco"].min()  # previsão <= 0 não tem log: trunca no menor preço do treino
                oof["rs_sem"][te_i] = oof["rs_duan"][te_i] = p
                oof["log"][te_i] = np.log(np.maximum(p, piso))
                oof["smear"][te_i] = (p < piso)  # aqui: marca previsões truncadas
        reps.append(oof)
    return reps, pd.DataFrame(coefs)


def resumo(dados, reps, log):
    y, ylog = dados["preco"].values, dados["log_preco"].values
    f = lambda fn: np.mean([fn(o) for o in reps])
    out = {
        "R2_log": f(lambda o: r2_score(ylog, o["log"])),
        "R2_R$_sem": f(lambda o: r2_score(y, o["rs_sem"])),
        "R2_R$": f(lambda o: r2_score(y, o["rs_duan"])),
        "R2_R$_dp": np.std([r2_score(y, o["rs_duan"]) for o in reps]),
        "RMSE": f(lambda o: np.sqrt(mean_squared_error(y, o["rs_duan"]))),
        "MAE": f(lambda o: mean_absolute_error(y, o["rs_duan"])),
        "MedAE": f(lambda o: np.median(np.abs(y - o["rs_duan"]))),
    }
    if log:
        out["smear"] = f(lambda o: o["smear"].mean())
    else:
        out["R2_R$_sem"] = np.nan  # sem transformação: "com/sem smearing" não se aplica
        out["truncadas"] = f(lambda o: o["smear"].sum())
    return out


def mostrar(tab, titulo):
    t = tab.copy()
    for c in ("R2_log", "R2_R$_sem"):
        if c in t:
            t[c] = t[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    t["R2_R$"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(tab["R2_R$"], tab["R2_R$_dp"])]
    for c in ("RMSE", "MAE", "MedAE"):
        t[c] = tab[c].map(lambda v: f"{v/1e3:,.0f} mil")
    for c in ("smear", "truncadas"):
        if c in t:
            t[c] = tab[c].map(lambda v: "—" if pd.isna(v) else (f"{v:.3f}" if c == "smear" else f"{v:.1f}"))
    print(f"\n=== {titulo} ===")
    print(t.drop(columns="R2_R$_dp").to_string(index=False))


ALVOS = {"log-log": True, "linear": False}

# 0. interação completa × simplificada
linhas = []
for nome_alvo, log in ALVOS.items():
    for inter in ("completa", "simplificada"):
        reps, coefs = cv(treino, log, inter, "limiar5")
        linhas.append({"alvo": nome_alvo, "interacao": inter, "params": coefs.shape[1] + 1, **resumo(treino, reps, log)})
t0 = pd.DataFrame(linhas)
mostrar(t0, "0. Interação área × tipo: completa vs. simplificada (bairro limiar 5; R$ do log-log com Duan)")

# 3. Duan e log × linear
INTER = "simplificada"
linhas = []
for nome_alvo, log in ALVOS.items():
    reps, _ = cv(treino, log, INTER, "limiar5")
    linhas.append({"alvo": nome_alvo, **resumo(treino, reps, log)})
t3 = pd.DataFrame(linhas)
mostrar(t3, f"3. Log × linear com smearing de Duan (interação {INTER}, bairro limiar 5)")
print("R2_log: R² em escala log | R2_R$_sem: exp(previsão) sem correção | R2_R$, RMSE, MAE, MedAE: com Duan\n"
      "smear: fator médio de Duan nos folds | truncadas: previsões <= piso no modelo linear (por repetição)")

# 4. codificação do bairro
METODOS = {"limiar5": "(a) bairro, limiar 5", "limiar3": "(b) bairro, limiar 3",
           "regional": "(c) regional administrativa", "target_enc": "(d) target encoding"}
CHAVE = ["log_area", "area_m2", "quartos", "banheiros", "vagas", "casa", "log_area_x_casa", "area_x_casa"]
linhas, estab, todos_coefs = [], [], {}
for nome_alvo, log in ALVOS.items():
    for met, desc in METODOS.items():
        reps, coefs = cv(treino, log, INTER, met)
        todos_coefs[(nome_alvo, met)] = coefs
        loc = [c for c in coefs.columns if c not in CHAVE]
        # estabilidade dos coeficientes de localização: presentes em >= 50% dos folds, sinal trocando
        presentes = [c for c in loc if coefs[c].notna().mean() >= 0.5]
        troca = [c for c in presentes if 0 < (coefs[c].dropna() > 0).mean() < 1]
        principais = [c for c in coefs.columns if c in CHAVE and c != "casa"]
        cv_princ = {c: coefs[c].std() / abs(coefs[c].mean()) for c in principais}
        linhas.append({"alvo": nome_alvo, "bairro": desc, "params": int(coefs.notna().mean().round().sum()) + 1,
                       **resumo(treino, reps, log)})
        estab.append({"alvo": nome_alvo, "bairro": desc, "coef_localizacao": len(presentes),
                      "trocam_sinal": len(troca), "quais_trocam": ", ".join(t.split("_", 1)[1] for t in troca)[:90],
                      **{f"cv_{c}": round(v, 3) for c, v in cv_princ.items()}})
t4 = pd.DataFrame(linhas)
for nome_alvo in ALVOS:
    mostrar(t4[t4["alvo"] == nome_alvo].drop(columns="alvo"), f"4. Codificação do bairro – alvo {nome_alvo}")

e4 = pd.DataFrame(estab)
print("\n=== 4. Estabilidade dos coeficientes nos 15 folds ===")
print("coef_localizacao: dummies/variável presentes em >= 50% dos folds | trocam_sinal: quantos mudam de sinal entre folds")
print("cv_*: coeficiente de variação (dp/|média|) dos efeitos principais entre folds (menor = mais estável)")
print(e4.to_string(index=False))

print("\nEfeito de localização no log-log (média ± dp nos folds):")
for met in METODOS:
    c = todos_coefs[("log-log", met)]
    loc = [x for x in c.columns if x not in CHAVE and c[x].notna().mean() >= 0.5]
    txt = ", ".join(f"{x.split('_', 1)[1]} {c[x].mean():+.2f}±{c[x].std():.2f}" for x in loc)
    print(f"  {METODOS[met]}: {txt}")

pd.concat([t0.assign(tabela="0_interacao"), t3.assign(tabela="3_duan"), t4.assign(tabela="4_bairro")]) \
    .to_csv(SAIDA / "itens3_4_resultados.csv", index=False, encoding="utf-8-sig")
e4.to_csv(SAIDA / "item4_estabilidade.csv", index=False, encoding="utf-8-sig")
print("\n-> resultados/exploratorio/itens3_4_resultados.csv, item4_estabilidade.csv")
