# science-data

🇺🇸 [Read in English](README.en.md)

Repositório de estudos de Ciência de Dados: web scraping, regressão linear, regressão logística e um projeto
completo de coleta e previsão de preços de imóveis. Cada pasta de projeto tem o próprio README com os detalhes,
os resultados e as instruções para rodar.

---

## Projetos

### [`previsao_precos_imoveis_bh/`](previsao_precos_imoveis_bh/): previsão de preços de imóveis em Belo Horizonte

Projeto completo, do zero ao relatório: coleta de anúncios do VivaReal com Playwright, limpeza, análise
exploratória e regressão linear múltipla, comparada com Ridge, Lasso, um modelo log-log e um Random Forest.

O que você vai encontrar:
- o relatório técnico em PDF (4 páginas);
- o código de cada etapa (`scraper.js` e `src/`);
- os gráficos, as métricas e os coeficientes do modelo;
- uma discussão sobre a ética do scraping. Os dados coletados **não** são publicados, e o README da pasta explica por quê.

Resultado principal: o modelo explica cerca de 2/3 da variação dos preços (R² 0,66 no teste).

### [`modelos_preditivos/`](modelos_preditivos/): aluguel e câncer de mama

Um script (`models_pipeline.py`) com duas tarefas de aprendizado de máquina em dados tabulares:
- **Regressão linear** para prever o valor do aluguel (`houses_to_rent.csv`). Compara a regressão simples, log nas
  explicativas, Huber e polinomial de grau 2 com Ridge, e mostra como identificar e remover vazamento de dados.
- **Regressão logística** para classificar tumores de mama como benignos ou malignos (`BreastCancer.csv`), com
  acurácia, precisão, recall, F1, AUC-ROC e matriz de confusão.

O que você vai encontrar: o script, os dados, os modelos treinados (`.joblib`) e as métricas em CSV.

Resultados principais: R² 0,57 no aluguel e AUC-ROC 0,995 no câncer de mama.

### [`regressao_linear/`](regressao_linear/): preço de imóveis (dataset sem sinal)

Pipeline de regressão linear com limpeza, novas variáveis, One-Hot Encoding e comparação entre LinearRegression,
Ridge, Lasso e ElasticNet com `GridSearchCV`.

O que você vai encontrar: o script (`train_model.py`), os dados, as métricas, os coeficientes e os gráficos.

Resultado principal: **nenhum modelo supera o baseline**, porque nesse dataset o preço não tem relação com as
variáveis. O README da pasta mostra como a comparação com o baseline revela isso.

### [`src/scraping/`](src/scraping/): exemplos e exercícios de web scraping

- `bs4_example.py` e `selenium_example.py` -> exemplos básicos com BeautifulSoup e Selenium;
- `exercises/` -> 5 exercícios guiados: livros (`books.toscrape.com`), busca com Selenium e autores
  (`quotes.toscrape.com`), laptops (`webscraper.io`) e produtos (`scrapingcourse.com`).

Todos os sites usados são públicos e feitos justamente para praticar scraping.

### [`notebooks/`](notebooks/)

- `01_webscraping_intro.ipynb` -> primeiros testes de scraping com BeautifulSoup.

---

## Estrutura

```
science-data/
├── previsao_precos_imoveis_bh/   projeto completo: scraping + regressão linear (imóveis em BH)
├── modelos_preditivos/           regressão linear (aluguel) e regressão logística (câncer de mama)
├── regressao_linear/             regressão linear em um dataset sem sinal
├── src/scraping/                 exemplos e exercícios de web scraping
├── notebooks/                    notebooks de estudo
├── data/                         raw/ e processed/ para dados dos exercícios (o conteúdo é ignorado no git)
├── environment.yml               ambiente conda para os exemplos de scraping
└── requirements.txt              dependências dos exemplos de scraping (pip)
```

Os projetos de modelagem têm dependências próprias: veja o `requirements.txt` ou o README de cada pasta.

---

## Ambiente para os exemplos de scraping

Com conda (Python 3.11):

```
conda env create -f environment.yml
conda activate science_data
```

Ou com pip, dentro de um ambiente já existente:

```
pip install -r requirements.txt
```

Para rodar um exemplo:

```
python src/scraping/bs4_example.py
python src/scraping/selenium_example.py
```

No VS Code, escolha o interpretador `science_data` em `Ctrl+Shift+P` → "Python: Select Interpreter". Nos
notebooks, selecione o kernel **Python (science_data)**.

---

## Próximos passos

- [x] Web scraping (BeautifulSoup, Selenium, Playwright)
- [x] Machine Learning (scikit-learn): regressão linear e logística
- [ ] NLP (nltk / spaCy)
