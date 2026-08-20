# science-data

Ambiente de estudo de Ciência de Dados: webscraping (BeautifulSoup4, Selenium), com espaço reservado para Machine Learning e NLP mais adiante.

## Estrutura

```
science_data/
├── data/
│   ├── raw/          # dados brutos coletados (ignorado no git)
│   └── processed/    # dados tratados (ignorado no git)
├── notebooks/         # notebooks Jupyter de estudo/experimentação
├── src/
│   └── scraping/       # scripts de webscraping (bs4 e selenium)
├── environment.yml     # definição do ambiente conda
└── requirements.txt    # dependências (pip)
```

## Ambiente (conda)

O ambiente `science_data` já foi criado localmente com Python 3.11. Para recriá-lo em outra máquina:

```powershell
conda env create -f environment.yml
conda activate science_data
```

Ou, se preferir pip dentro de um env existente:

```powershell
pip install -r requirements.txt
```

## Uso no VS Code

1. Abra a pasta `C:\science_data` no VS Code.
2. `Ctrl+Shift+P` → "Python: Select Interpreter" → escolha `science_data` (env do conda).
3. Para notebooks: crie um `.ipynb` em `notebooks/` e selecione o kernel **Python (science_data)** no canto superior direito.

## Scripts de exemplo

- `src/scraping/bs4_example.py` — requests + BeautifulSoup em uma página estática.
- `src/scraping/selenium_example.py` — Selenium (headless Chrome) em uma página renderizada via JS.

Ambos usam [quotes.toscrape.com](https://quotes.toscrape.com/), um site público feito para prática de scraping.

Rodar:

```powershell
conda activate science_data
python src/scraping/bs4_example.py
python src/scraping/selenium_example.py
```

## Próximos passos

- [ ] Machine Learning (scikit-learn)
- [ ] NLP (nltk / spaCy)
