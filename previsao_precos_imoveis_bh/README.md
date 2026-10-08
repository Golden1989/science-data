# Avaliação 1 – Web scraping e regressão linear para preços de imóveis (Belo Horizonte)

Trabalho da disciplina de Ciência de Dados: coleta de anúncios de venda do VivaReal (apartamentos, casas e
coberturas em BH), limpeza, análise exploratória e regressão linear para prever o preço.

> **Versão pública.** Em relação ao trabalho entregue, esta versão **não inclui os dados coletados** e teve
> **um trecho do scraper removido**. Os motivos estão em [O que foi retirado desta versão e por quê](#o-que-foi-retirado-desta-versão-e-por-quê).

---

## Comece aqui

- **Ler o trabalho** -> [`relatorio/relatorio.pdf`](relatorio/relatorio.pdf) (4 páginas, as 5 etapas do enunciado)
- Ver os gráficos -> pasta [`graficos/`](graficos/) (figuras 1 a 7 do relatório)
- Ver os números do modelo em detalhe -> [`relatorio/resumo_etapa4.md`](relatorio/resumo_etapa4.md) e pasta [`resultados/`](resultados/)
- Ver o código -> pasta [`src/`](src/) e [`scraper.js`](scraper.js)

---

## O modelo final em uma frase

Regressão linear (OLS, erros-padrão robustos HC3) do preço em função de área, quartos, banheiros, vagas, tipo
(casa × apartamento, com interação com a área), regional administrativa de BH e lançamento. Uma versão log-log
complementar lê os efeitos em %. Todas as escolhas foram feitas por validação cruzada só no treino.

Principais resultados no conjunto de teste:

- Linear OLS -> R² = 0,658 · MAE = R$ 238.265
- Log-log (complementar) -> R² = 0,782 · MAE = R$ 197.789
- Cada m² adicional eleva o preço do apartamento em cerca de R$ 4.475

Detalhes e justificativas estão nas seções "Especificação do modelo e métricas" e "Interpretação dos coeficientes"
do relatório.

---

## Onde está cada etapa

- 1 – Coleta (web scraping) -> `scraper.js` (Node.js + Playwright) -> `dados/brutos/imoveis.csv` (não publicado)
- 2 – Limpeza e tratamento -> `src/limpeza.py` -> `dados/processados/imoveis_limpo.csv` (não publicado)
- 3 – Análise exploratória -> `src/eda.py` -> `graficos/1_` a `4_*.png`
- 4 – Modelagem (OLS, Ridge, Lasso, log, VIF) -> `src/modelo_final.py` (chama `diagnosticos.py` e `gerar_resumo.py`) -> `graficos/5_` a `7_*.png`, `resultados/*.csv`, `relatorio/resumo_etapa4.md`
- 5 – Relatório -> `relatorio/relatorio.md` -> `gerar_relatorio.js` -> `relatorio/relatorio.docx` -> `relatorio.pdf`

Os scripts em `src/exploratorio/` são as análises intermediárias que justificaram as escolhas do modelo final.

---

## O que foi retirado desta versão e por quê

### 1. Os dados coletados (`dados/brutos/` e `dados/processados/`)

A base tem o endereço e o link de cada anúncio, que podem identificar indiretamente os anunciantes (LGPD). Além
disso, a coleta **não respeitou o robots.txt nem os Termos de Uso do Grupo OLX** (dono do VivaReal), cuja
cláusula 9.1 proíbe usar robôs para reproduzir ou distribuir o conteúdo do site. Publicar a base seria
redistribuir esse conteúdo. Por isso só estão aqui os **resultados agregados** (coeficientes, métricas e
gráficos), que não expõem anúncios individuais.

### 2. O trecho do scraper que contornava a proteção do site

O site usa o Cloudflare, que bloqueia navegadores automatizados. Na versão entregue, o `scraper.js` trocava o
user-agent por um de Chrome comum e escondia a marca `navigator.webdriver`, para não ser identificado como
automação. Esse trecho foi **removido** desta versão: publicar um código feito para burlar a proteção de um site
específico não é adequado. Sem ele, o scraper provavelmente será bloqueado (erro 403).

### 3. O robots.txt continua sendo um problema

A URL de busca usada no `scraper.js` contém os parâmetros `onde=` e `tipos=`, que o robots.txt do VivaReal
bloqueia para robôs (`Disallow: *onde=*` e `*tipos=*`). **Não use este scraper para coletar dados do VivaReal.**
Ele está aqui para documentar o método do trabalho. Para reproduzir a análise, o caminho adequado é pedir
autorização ao site, usar uma API ou base oficial, ou escolher uma fonte que permita a coleta.

A seção "Considerações éticas sobre scraping" do relatório discute esses pontos.

---

## Como rodar a análise com os seus próprios dados

Os scripts Python funcionam com qualquer base no mesmo formato. Coloque em `dados/brutos/imoveis.csv` um CSV
(UTF-8) com as colunas:

`preco, area_m2, quartos, banheiros, vagas, bairro, tipo, endereco, link, pagina`

O preço e a área podem vir como texto ("R$ 1.200.000", "75 m²"), porque a limpeza converte. Os bairros precisam
estar em `dados/regionais_bh.csv` (mapeamento bairro -> regional administrativa de BH); se não estiverem, o
script avisa quais faltam.

### Instalar (uma vez só)

Precisa de **Python 3.11 ou mais novo** (testado no 3.14):

**Windows:**
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**macOS / Linux:**
```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### Rodar, nesta ordem

**Windows:**
```
.venv\Scripts\python src\limpeza.py
.venv\Scripts\python src\eda.py
.venv\Scripts\python src\modelo_final.py
```

**macOS / Linux:**
```
.venv/bin/python src/limpeza.py
.venv/bin/python src/eda.py
.venv/bin/python src/modelo_final.py
```

Com outra base, os números serão diferentes dos do relatório.

### Gerar o relatório (opcional, Node.js 18+)

```
npm install
node gerar_relatorio.js
```

Gera `relatorio/relatorio.docx` a partir de `relatorio/relatorio.md` e das figuras. O PDF foi exportado pelo Word.

---

## Estrutura de pastas

```
previsao_precos_imoveis_bh/
├── README.md                este arquivo
├── requirements.txt         pacotes Python (versões fixas)
├── package.json             pacotes Node.js (playwright, docx)
├── scraper.js               Etapa 1: coleta (sem o trecho de contorno; ver acima)
├── gerar_relatorio.js       relatorio.md -> relatorio.docx
├── src/
│   ├── limpeza.py           Etapa 2
│   ├── eda.py               Etapa 3
│   ├── modelo_final.py      Etapa 4: dados, validação cruzada, ajuste final e teste
│   ├── diagnosticos.py      Breusch-Pagan, VIF, Cook, figuras 5 a 7 (chamado pelo modelo_final.py)
│   ├── gerar_resumo.py      escreve relatorio/resumo_etapa4.md (chamado pelo modelo_final.py)
│   └── exploratorio/        análises intermediárias
├── dados/
│   └── regionais_bh.csv     bairro -> regional administrativa de BH (os CSVs da coleta não são publicados)
├── resultados/              coeficientes do modelo final, VIF, Breusch-Pagan
│   └── exploratorio/        tabelas das análises intermediárias
├── graficos/                figuras 1 a 7 do relatório
└── relatorio/               relatorio.pdf, relatorio.docx, relatorio.md, resumo_etapa4.md
```

---

## Análises intermediárias (`src/exploratorio/`)

- `modelagem.py` -> primeira versão da Etapa 4 (bairro com one-hot, OLS/Ridge/Lasso, log, VIF)
- `analise_casas_cook.py` -> interação área × tipo (nas casas, a área informada parece ser a do terreno) e distância de Cook
- `etapa4_itens3_4.py` -> smearing de Duan e codificação do bairro (one-hot com limiar 5 e 3, regional, target encoding)
