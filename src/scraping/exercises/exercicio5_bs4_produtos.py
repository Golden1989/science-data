"""
EXERCICIO 5 - Webscraping SEM Selenium (requests + BeautifulSoup)
Site: https://www.scrapingcourse.com/ecommerce/
(site publico, feito por uma empresa de scraping justamente para pratica)

Mesma formula de novo: parse de um item, paginacao, DataFrame, CSV.
Ultimo de repeticao antes de seguir pra outra coisa -- capricha nesse.

OBJETIVO
--------
Coletar dados das 3 primeiras paginas de produtos.

Para cada produto, extrair:
  - name         (nome do produto)
  - price        (float, sem o simbolo $)
  - product_url  (link da pagina do produto)

Depois:
  - Montar um pandas.DataFrame
  - Salvar em data/processed/produtos.csv (index=False)
  - Imprimir quantos produtos foram coletados e o preco medio

DICAS
-----
- Cada produto fica dentro de <li class="product">.
- O nome esta dentro de um <h2> (texto direto, sem precisar de atributo
  dessa vez -- so soup.select_one("h2").get_text(strip=True)).
- O preco esta em <span class="price">$69.00</span>.
- O link do produto esta no atributo href do primeiro <a> dentro do
  <li class="product">.
- Paginacao: procure por <a class="next" href="...">. Repare que aqui o
  href da proxima pagina JA VEM COMO URL COMPLETA (comeca com "https://"),
  entao dessa vez voce nem precisa usar urljoin -- pode usar o href direto.
  (No exercicio 1 e no 4 o href era relativo e precisava de urljoin; aqui
  nao precisa. Bom pra perceber que cada site se comporta diferente, e
  sempre vale conferir olhando o href de verdade antes de decidir.)
- Limite a 3 paginas (NUM_PAGES ja esta definido abaixo).

Rode com:
    conda activate science_data
    python src/scraping/exercises/exercicio5_bs4_produtos.py
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd

START_URL = "https://www.scrapingcourse.com/ecommerce/"
NUM_PAGES = 3


def parse_item(card) -> dict:
    """Recebe uma tag <li class="product"> e retorna um dict com
    name, price e product_url.

    TODO: implementar.
    """
    raise NotImplementedError


def get_next_page_url(soup: BeautifulSoup) -> str | None:
    """Retorna a URL da proxima pagina, ou None se nao houver.

    TODO: implementar (aqui NAO precisa de urljoin, o href ja vem completo).
    """
    raise NotImplementedError


def scrape_products(start_url: str = START_URL, num_pages: int = NUM_PAGES) -> list[dict]:
    """Percorre `num_pages` paginas a partir de start_url e retorna a lista
    de dicts de todos os produtos encontrados.

    TODO: implementar o loop de paginacao (igual aos exercicios anteriores).
    """
    raise NotImplementedError


if __name__ == "__main__":
    products = scrape_products()
    df = pd.DataFrame(products)

    df.to_csv("data/processed/produtos.csv", index=False)

    print(f"Produtos coletados: {len(df)}")
    print(f"Preco medio: ${df['price'].mean():.2f}")
