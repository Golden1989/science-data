"""Etapa 3 – Análise exploratória (EDA) dos imóveis limpos.

Entrada: dados/processados/imoveis_limpo.csv (gerado por src/limpeza.py)
Saída:   graficos/1_*.png a graficos/4_*.png e estatísticas no terminal
Caminhos relativos à raiz do projeto; o script roda de qualquer pasta.

Uso: python src/eda.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, ScalarFormatter

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "dados" / "processados" / "imoveis_limpo.csv"
PASTA = RAIZ / "graficos"
PASTA.mkdir(exist_ok=True)

# Paleta
AZUL = "#2a78d6"
AZUL_ESCURO = "#184f95"
LARANJA = "#eb6834"
NEUTRO = "#f0efec"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
FUNDO = "#fcfcfb"
GRADE = "#e4e3df"

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
    "axes.edgecolor": GRADE, "grid.color": GRADE, "grid.linewidth": 0.8,
    "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "text.color": TINTA, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlelocation": "left", "font.size": 10, "figure.dpi": 110,
})

reais = FuncFormatter(lambda v, _: f"R$ {v/1e6:.1f} mi".replace(".", ",") if abs(v) >= 1e6 else f"R$ {v/1e3:.0f} mil")
reais_m2 = FuncFormatter(lambda v, _: f"R$ {v/1e3:.0f} mil")
NUM = ["preco", "preco_m2", "area_m2", "quartos", "banheiros", "vagas"]
ROTULOS = {"preco": "Preço", "preco_m2": "Preço/m²", "area_m2": "Área (m²)",
           "quartos": "Quartos", "banheiros": "Banheiros", "vagas": "Vagas"}

df = pd.read_csv(ENTRADA, encoding="utf-8-sig")
print(f"Imóveis: {len(df)} | bairros: {df['bairro'].nunique()}")
print("\nDistribuição por tipo:\n" + df["tipo"].value_counts().to_string())
with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 160, "display.max_columns", None):
    print("\nEstatísticas descritivas:")
    print(df[NUM].describe().T)
    print(f"\nAssimetria (skew): preco={df['preco'].skew():.2f} | preco_m2={df['preco_m2'].skew():.2f} "
          f"| log(preco)={np.log(df['preco']).skew():.2f}")


def virgula_eixos(fig):
    """Troca o ponto decimal por vírgula nos eixos que usam o formatador numérico padrão."""
    for ax in fig.axes:
        for eixo in (ax.xaxis, ax.yaxis):
            if isinstance(eixo.get_major_formatter(), ScalarFormatter):
                eixo.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def salvar(fig, nome):
    virgula_eixos(fig)
    fig.tight_layout()
    fig.savefig(PASTA / nome, dpi=150)
    plt.close(fig)
    print(f"-> {PASTA / nome}")


print()
# 1. Histogramas de preco e preco_m2
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col, fmt, titulo in [
    (axes[0], "preco", reais, "Distribuição do preço"),
    (axes[1], "preco_m2", reais_m2, "Distribuição do preço por m²"),
]:
    sns.histplot(df[col], bins=25, color=AZUL, edgecolor=FUNDO, linewidth=1.5, alpha=1, ax=ax)
    med = df[col].median()
    ax.axvline(med, color=TINTA, lw=1.5, ls="--")
    ax.text(med, ax.get_ylim()[1] * 0.97, f" mediana: R$ {med:,.0f}".replace(",", "."), va="top", fontsize=9,
            color=TINTA, bbox={"facecolor": FUNDO, "edgecolor": "none", "pad": 2})
    ax.xaxis.set_major_formatter(fmt)
    ax.set(title=titulo, xlabel=ROTULOS[col], ylabel="Nº de imóveis")
    ax.grid(axis="x", visible=False)
salvar(fig, "1_histogramas.png")

# 2. Matriz de correlação (Pearson)
corr = df[NUM].corr()
divergente = LinearSegmentedColormap.from_list("div", [LARANJA, NEUTRO, AZUL])
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr.rename(index=ROTULOS, columns=ROTULOS), annot=corr.map(lambda v: f"{v:.2f}".replace(".", ",")).values, fmt="", cmap=divergente,
            vmin=-1, vmax=1, center=0, square=True, linewidths=2, linecolor=FUNDO,
            annot_kws={"color": TINTA, "fontsize": 10}, cbar_kws={"shrink": 0.8, "label": "Correlação de Pearson"}, ax=ax)
ax.set_title("Matriz de correlação")
ax.grid(False)
salvar(fig, "2_correlacao.png")

# 3. Scatter area_m2 × preco com reta de tendência (escala linear e log-log)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
kw = dict(scatter_kws={"s": 28, "color": AZUL, "alpha": 0.6, "edgecolor": FUNDO, "linewidths": 0.8},
          line_kws={"color": TINTA, "lw": 2})
sns.regplot(data=df, x="area_m2", y="preco", ax=axes[0], **kw)
r = df["area_m2"].corr(df["preco"])
axes[0].yaxis.set_major_formatter(reais)
axes[0].set(title="Área × preço", xlabel="Área (m²)", ylabel="Preço")
axes[0].text(0.98, 0.04, f"r = {r:.2f}".replace(".", ","), transform=axes[0].transAxes, ha="right", color=TINTA_2)

log = pd.DataFrame({"log_area": np.log(df["area_m2"]), "log_preco": np.log(df["preco"])})
sns.regplot(data=log, x="log_area", y="log_preco", ax=axes[1], **kw)
r_log = log.corr().iloc[0, 1]
ticks_a = [30, 50, 100, 200, 400, 800]
ticks_p = [3e5, 5e5, 1e6, 2e6, 3e6]
axes[1].set_xticks(np.log(ticks_a), [str(t) for t in ticks_a])
axes[1].set_yticks(np.log(ticks_p), [reais(t, None) for t in ticks_p])
axes[1].set(title="Área × preço (escala logarítmica)", xlabel="Área (m², escala log)", ylabel="Preço (escala log)")
axes[1].text(0.98, 0.04, f"r = {r_log:.2f}".replace(".", ","), transform=axes[1].transAxes, ha="right", color=TINTA_2)
salvar(fig, "3_area_x_preco.png")

# 4. Boxplot de preço pelos 10 bairros com mais anúncios (ordenados pela mediana)
top = df["bairro"].value_counts().head(10)
sub = df[df["bairro"].isin(top.index)]
ordem = sub.groupby("bairro")["preco"].median().sort_values(ascending=False).index
rotulos = [f"{b} (n={top[b]})" for b in ordem]
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=sub, y="bairro", x="preco", order=ordem, color=AZUL, width=0.6, linewidth=1.2,
            boxprops={"alpha": 0.35, "edgecolor": AZUL_ESCURO}, medianprops={"color": TINTA, "lw": 2},
            whiskerprops={"color": AZUL_ESCURO}, capprops={"color": AZUL_ESCURO}, showfliers=False, ax=ax)
sns.stripplot(data=sub, y="bairro", x="preco", order=ordem, color=AZUL, size=4.5, alpha=0.8,
              jitter=0.18, edgecolor=FUNDO, linewidth=0.6, ax=ax)
ax.set_yticks(range(len(ordem)), rotulos)
ax.xaxis.set_major_formatter(reais)
ax.set(title="Preço nos 10 bairros com mais anúncios", xlabel="Preço", ylabel="")
ax.grid(axis="y", visible=False)
salvar(fig, "4_boxplot_bairros.png")

# Reflexão: correlação com o preço
print("\nCorrelação com o preço:")
tab = pd.DataFrame({
    "pearson": df[NUM].corr()["preco"],
    "spearman": df[NUM].corr(method="spearman")["preco"],
}).drop("preco").sort_values("pearson", ascending=False)
print(tab.round(3).to_string())
print(f"\nÁrea × preço: r linear = {r:.3f} | r log-log = {r_log:.3f}")
print("\nMediana de preço e R$/m² nos top 10 bairros:")
print(sub.groupby("bairro")[["preco", "preco_m2"]].median().loc[ordem].round(0).astype(int).to_string())
