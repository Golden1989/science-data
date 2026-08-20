"""
EXERCICIO 1 - Webscraping SEM Selenium (requests + BeautifulSoup)
Site: https://books.toscrape.com/  (site publico, feito para pratica de scraping)

OBJETIVO
--------
Coletar dados de livros das 3 primeiras paginas do catalogo e salvar em CSV.

Para cada livro, extrair:
  - titulo
  - preco (como float, sem o simbolo de moeda)
  - rating (como numero inteiro de 1 a 5, nao como texto "Three")
  - disponibilidade (texto, ex: "In stock")

Depois:
  - Juntar tudo em um pandas.DataFrame
  - Salvar em data/processed/livros.csv (index=False)
  - Imprimir quantos livros foram coletados e o preco medio

DICAS
-----
- Cada livro esta dentro de <article class="product_pod">.
- O titulo completo (nao truncado) esta no atributo `title` do <a> dentro do <h3>.
- O preco esta em <p class="price_color"> como texto "£51.77" -> precisa limpar
  o simbolo de moeda e converter para float.
- ATENCAO: esse site declara o encoding errado (ISO-8859-1), entao o simbolo
  £ pode vir corrompido (ex: "Â£51.77"). Corrija com
  `response.encoding = response.apparent_encoding` logo depois do
  `requests.get(...)`, antes de passar `response.text` pro BeautifulSoup.
- O rating vem como classe CSS, ex: <p class="star-rating Three">.
  A segunda classe (Three, Four, One...) e o rating por extenso em ingles.
  Voce vai precisar de um dicionario para converter texto -> numero.
- Para paginar: o link "next" fica em <li class="next"><a href="...">next</a></li>.
  As paginas seguem o padrao catalogue/page-2.html, catalogue/page-3.html...
  Quando nao existir mais <li class="next">, acabou o catalogo.
- Use BASE_URL para montar a URL completa a partir do href relativo.

Rode com:
    conda activate science_data
    python src/scraping/exercises/exercicio1_bs4_livros.py
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://books.toscrape.com/"
START_URL = "https://books.toscrape.com/index.html"
NUM_PAGES = 3  # colete so as 3 primeiras paginas

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def parse_book(article) -> dict:
    """Recebe uma tag <article class="product_pod"> e retorna um dict com
    title, price, rating e availability.

    TODO: implementar.
    """
    raise NotImplementedError


def get_next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    """Retorna a URL absoluta da proxima pagina, ou None se nao houver.

    TODO: implementar (procurar <li class="next"><a href="...">, usar
    requests.compat.urljoin(current_url, href) para resolver o caminho relativo).
    """
    raise NotImplementedError


def scrape_books(start_url: str = START_URL, num_pages: int = NUM_PAGES) -> list[dict]:
    """Percorre `num_pages` paginas a partir de start_url e retorna a lista
    de dicts de todos os livros encontrados.

    TODO: implementar o loop de paginacao usando parse_book() e
    get_next_page_url().
    """
    raise NotImplementedError


if __name__ == "__main__":
    books = scrape_books()
    df = pd.DataFrame(books)

    df.to_csv("data/processed/livros.csv", index=False)

    print(f"Livros coletados: {len(df)}")
    print(f"Preco medio: £{df['price'].mean():.2f}")
