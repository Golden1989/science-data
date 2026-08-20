"""
EXERCICIO 4 - Webscraping SEM Selenium (requests + BeautifulSoup)
Site: https://webscraper.io/test-sites/e-commerce/static/computers/laptops
(site publico, feito por uma empresa de scraping justamente para pratica)

Mesma formula do exercicio 1: parse de um item, paginacao, DataFrame, CSV.
Nao tem nada novo aqui, e so pra repetir a sintaxe em outro site.

OBJETIVO
--------
Coletar dados das 3 primeiras paginas de laptops.

Para cada produto, extrair:
  - name     (nome completo do produto, sem cortar)
  - price    (float, sem o simbolo $)
  - rating   (numero de 1 a 5)
  - reviews  (numero inteiro de avaliacoes)

Depois:
  - Montar um pandas.DataFrame
  - Salvar em data/processed/laptops.csv (index=False)
  - Imprimir quantos produtos foram coletados e o preco medio

DICAS
-----
- Cada produto fica dentro de <div class="thumbnail">.
- O nome completo (nao cortado) esta no atributo `title` de um
  <a class="title" title="...">, igual ao esquema do exercicio 1 com
  os livros.
- O preco esta em <span itemprop="price">$295.99</span> -- aqui o simbolo
  e "$" (dolar), nao "£", e nao tem o bug de encoding do exercicio 1.
- O rating NAO vem em classe CSS dessa vez -- vem direto como atributo:
  <p data-rating="3">...</p>. Pra pegar um atributo de uma tag, e so fazer
  tag["data-rating"] (lembra do exercicio 1 pegando o atributo "title"?
  e a mesma ideia, so muda o nome do atributo). Repare que o valor volta
  como string ("3"), entao converte pra int.
- O numero de reviews esta em <span itemprop="reviewCount">14</span>.
- Paginacao: a URL usa query string, tipo
  ...static/computers/laptops?page=2 -- procure por <a class="next"
  rel="next" href="...">. Se nao existir esse link, acabaram as paginas.
- Limite a 3 paginas (NUM_PAGES ja esta definido abaixo).

Rode com:
    conda activate science_data
    python src/scraping/exercises/exercicio4_bs4_laptops.py
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://webscraper.io/"
START_URL = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops"
NUM_PAGES = 3


def parse_item(card) -> dict:
    """Recebe uma tag <div class="thumbnail"> e retorna um dict com
    name, price, rating e reviews.

    TODO: implementar.
    """
    raise NotImplementedError


def get_next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    """Retorna a URL absoluta da proxima pagina, ou None se nao houver.

    TODO: implementar (igual ao exercicio 1: procurar o link, usar
    requests.compat.urljoin caso o href seja relativo).
    """
    raise NotImplementedError


def scrape_laptops(start_url: str = START_URL, num_pages: int = NUM_PAGES) -> list[dict]:
    """Percorre `num_pages` paginas a partir de start_url e retorna a lista
    de dicts de todos os produtos encontrados.

    TODO: implementar o loop de paginacao (igual ao exercicio 1).
    """
    raise NotImplementedError


if __name__ == "__main__":
    laptops = scrape_laptops()
    df = pd.DataFrame(laptops)

    df.to_csv("data/processed/laptops.csv", index=False)

    print(f"Produtos coletados: {len(df)}")
    print(f"Preco medio: ${df['price'].mean():.2f}")
