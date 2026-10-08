# science-data

🇧🇷 [Leia em português](README.md)

A Data Science study repository: web scraping, linear regression, logistic regression and a complete project that
collects real estate listings and predicts their prices. Each project folder has its own README with details,
results and instructions to run it. The code, comments and reports are written in Portuguese.

---

## Projects

### [`previsao_precos_imoveis_bh/`](previsao_precos_imoveis_bh/): real estate price prediction in Belo Horizonte, Brazil

A complete end-to-end project: scraping listings from VivaReal (a Brazilian real estate portal) with Playwright,
data cleaning, exploratory analysis and multiple linear regression, compared with Ridge, Lasso, a log-log model and a
Random Forest.

What you will find:
- the technical report as a PDF (4 pages, in Portuguese);
- the code for each step (`scraper.js` and `src/`);
- the charts, metrics and model coefficients;
- a discussion of the ethics of web scraping. The scraped data is **not** published, and the folder's README explains why.

Main result: the model explains about 2/3 of the price variation (R² 0.66 on the test set).

### [`modelos_preditivos/`](modelos_preditivos/): rent prices and breast cancer ("predictive models")

A single script (`models_pipeline.py`) with two machine learning tasks on tabular data:
- **Linear regression** to predict rent prices (`houses_to_rent.csv`). It compares plain regression, log-transformed
  features, Huber regression and a degree-2 polynomial with Ridge, and shows how to detect and remove data leakage.
- **Logistic regression** to classify breast tumors as benign or malignant (`BreastCancer.csv`), with accuracy,
  precision, recall, F1, ROC AUC and a confusion matrix.

What you will find: the script, the datasets, the trained models (`.joblib`) and the metrics as CSV files.

Main results: R² 0.57 for rent prices and ROC AUC 0.995 for breast cancer.

### [`regressao_linear/`](regressao_linear/): house prices on a dataset with no signal ("linear regression")

A linear regression pipeline with data cleaning, feature engineering, one-hot encoding and a comparison of
LinearRegression, Ridge, Lasso and ElasticNet tuned with `GridSearchCV`.

What you will find: the script (`train_model.py`), the dataset, the metrics, the coefficients and the charts.

Main result: **no model beats the baseline**, because in this dataset the price is not related to any of the
features. The folder's README shows how comparing against a baseline reveals this.

### [`src/scraping/`](src/scraping/): web scraping examples and exercises

- `bs4_example.py` and `selenium_example.py` -> basic examples with BeautifulSoup and Selenium;
- `exercises/` -> 5 guided exercises: books (`books.toscrape.com`), Selenium search and authors
  (`quotes.toscrape.com`), laptops (`webscraper.io`) and products (`scrapingcourse.com`).

All the websites used are public sandboxes built specifically for scraping practice.

### [`notebooks/`](notebooks/)

- `01_webscraping_intro.ipynb` -> first scraping tests with BeautifulSoup.

---

## Structure

```
science-data/
├── previsao_precos_imoveis_bh/   end-to-end project: scraping + linear regression (real estate in Belo Horizonte)
├── modelos_preditivos/           linear regression (rent) and logistic regression (breast cancer)
├── regressao_linear/             linear regression on a dataset with no signal
├── src/scraping/                 web scraping examples and exercises
├── notebooks/                    study notebooks
├── data/                         raw/ and processed/ for exercise data (contents are git-ignored)
├── environment.yml               conda environment for the scraping examples
└── requirements.txt              dependencies for the scraping examples (pip)
```

The modeling projects have their own dependencies: see the `requirements.txt` or the README in each folder.

---

## Environment for the scraping examples

With conda (Python 3.11):

```
conda env create -f environment.yml
conda activate science_data
```

Or with pip, inside an existing environment:

```
pip install -r requirements.txt
```

To run an example:

```
python src/scraping/bs4_example.py
python src/scraping/selenium_example.py
```

In VS Code, select the `science_data` interpreter with `Ctrl+Shift+P` → "Python: Select Interpreter". For the
notebooks, select the **Python (science_data)** kernel.

---

## Roadmap

- [x] Web scraping (BeautifulSoup, Selenium, Playwright)
- [x] Machine Learning (scikit-learn): linear and logistic regression
- [ ] NLP (nltk / spaCy)
