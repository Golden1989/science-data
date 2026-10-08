"""Etapa 2 – Limpeza e tratamento dos imóveis coletados pelo scraper.js.

Entrada:  dados/brutos/imoveis.csv            (dados brutos, ex.: preco="R$ 1.200.000", area_m2="75 m²")
Saída:    dados/processados/imoveis_limpo.csv (dados numéricos, sem ausentes nem outliers, com preco_m2)
Caminhos relativos à raiz do projeto; o script roda de qualquer pasta.

Uso: python src/limpeza.py [entrada.csv] [saida.csv]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "dados" / "brutos" / "imoveis.csv"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else RAIZ / "dados" / "processados" / "imoveis_limpo.csv"
SAIDA.parent.mkdir(parents=True, exist_ok=True)


def converter_preco(valor):
    """'R$ 1.200.000' -> 1200000.0 ; textos sem número (ex.: 'Sob consulta') -> NaN."""
    if pd.isna(valor):
        return np.nan
    m = re.search(r"\d[\d.]*(?:,\d+)?", str(valor))
    if not m:
        return np.nan
    return float(m.group().replace(".", "").replace(",", "."))


def converter_area(valor):
    """'75 m²' -> 75.0 ; faixas de lançamentos ('120 - 246 m²') usam o primeiro valor."""
    if pd.isna(valor):
        return np.nan
    m = re.search(r"\d+(?:[.,]\d+)?", str(valor))
    return float(m.group().replace(",", ".")) if m else np.nan


def limites_iqr(serie, k=3.0):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


df = pd.read_csv(ENTRADA, encoding="utf-8-sig")
print(f"Registros brutos: {len(df)}")

# 1 e 2. Conversão de tipos
df["preco"] = df["preco"].apply(converter_preco)
df["area_m2"] = df["area_m2"].apply(converter_area)
for col in ["quartos", "banheiros", "vagas"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["bairro"] = df["bairro"].str.strip()
df["tipo"] = df["tipo"].str.strip().str.lower()

# 3. Valores ausentes
print("\nAusentes antes do tratamento:")
print(df[["preco", "area_m2", "quartos", "banheiros", "vagas", "bairro", "tipo"]].isna().sum().to_string())

# Sem preço ou área não há como usar o imóvel (preco é a variável alvo) -> dropna
df = df.dropna(subset=["preco", "area_m2"])
# O site omite o campo de vagas quando o imóvel não tem garagem -> imputação com 0
df["vagas"] = df["vagas"].fillna(0)
# Quartos/banheiros ausentes (se houver) -> mediana do mesmo tipo de imóvel
for col in ["quartos", "banheiros"]:
    df[col] = df[col].fillna(df.groupby("tipo")[col].transform("median"))
df = df.dropna(subset=["quartos", "banheiros", "bairro", "tipo"])
for col in ["quartos", "banheiros", "vagas"]:
    df[col] = df[col].astype(int)
print(f"Após tratar ausentes: {len(df)}")

# 5. Preço por m² (criado antes dos outliers para também filtrar valores absurdos de R$/m²)
df["preco_m2"] = df["preco"] / df["area_m2"]

# 4. Outliers: fora de Q1 - 3×IQR ou Q3 + 3×IQR (outliers extremos) em preco e preco_m2
for col in ["preco", "preco_m2"]:
    lo, hi = limites_iqr(df[col])
    fora = ~df[col].between(lo, hi)
    print(f"Outliers em {col} (fora de [{lo:,.0f}; {hi:,.0f}]): {fora.sum()}")
    df = df[~fora]

df["preco_m2"] = df["preco_m2"].round(2)

# Lançamento (imóvel novo na planta/em obras): identificado pelo link do anúncio. Cartões agrupados
# ("Ver os N anúncios deste imóvel") não têm link e ficam como 0 por falta de informação.
df["lancamento"] = df["link"].fillna("").str.contains("/imoveis-lancamentos/").astype(int)
print(f"Lançamentos: {df['lancamento'].sum()} | sem link (lancamento=0 por falta de informação): "
      f"{df['link'].isna().sum()}")

df = df.reset_index(drop=True)
df.to_csv(SAIDA, index=False, encoding="utf-8-sig")

print(f"\nRegistros finais: {len(df)} -> {SAIDA}")
print("\nTipos de dados:")
print(df.dtypes.to_string())
print("\nResumo:")
with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 160, "display.max_columns", None):
    print(df[["preco", "area_m2", "preco_m2", "quartos", "banheiros", "vagas"]].describe().T)
