"""
EXERCICIO 3 - Webscraping SEM Selenium (requests + BeautifulSoup)
Site: https://quotes.toscrape.com/  (o mesmo do exemplo bs4_example.py)

Esse exercicio segue a MESMA estrutura do exercicio 1 (parse de um item,
paginacao, DataFrame final), mas com uma habilidade nova: seguir um link
para uma PAGINA DE DETALHE para completar a informacao. Isso e muito comum
em scraping de verdade (ex: lista de produtos que linka pra pagina de
detalhe de cada um).

OBJETIVO
--------
Coletar dados das 2 primeiras paginas de citacoes e, para cada citacao,
completar com a data de nascimento do autor (que so aparece na pagina do
autor, nao na listagem).

Para cada citacao, o resultado final deve ter:
  - quote_text     (o texto da citacao)
  - author         (nome do autor)
  - tags           (lista de tags, como lista de strings -- ex: ["change", "world"])
  - author_born    (data de nascimento do autor, ex: "March 14, 1879")

Depois:
  - Montar um pandas.DataFrame
  - Salvar em data/processed/citacoes.csv (index=False)
  - Imprimir quantas citacoes foram coletadas e quantos autores UNICOS existem
    (dica: df['author'].nunique())

DICAS
-----
- Cada citacao fica num <div class="quote">.
- O texto esta em <span class="text" itemprop="text">.
- O nome do autor esta em <small class="author" itemprop="author">.
- As tags ficam em varias <a class="tag">dentro de <div class="tags"></a>
  -- use select() (nao select_one) pra pegar todas, e uma list comprehension
  pra virar lista de strings.
- O link "(about)" do autor e um <a href="/author/Albert-Einstein"> --
  repare que o href e RELATIVO (comeca com /), entao precisa juntar com o
  dominio base (https://quotes.toscrape.com) antes de dar requests.get nele.
  Dica: requests.compat.urljoin("https://quotes.toscrape.com/", href)
- ATENCAO: como cada citacao tem um link de autor, se voce nao tomar cuidado
  vai fazer uma requisicao NOVA pra pagina do mesmo autor toda vez que ele
  aparecer (Einstein aparece varias vezes na mesma pagina!). Isso funciona,
  mas e MUITO ineficiente e sobrecarrega o servidor a toa.
  Sugestao: guarde um dict cache = {} onde a chave e a URL do autor e o
  valor e a data de nascimento ja buscada. Antes de baixar a pagina do
  autor, confira se ela ja esta no cache.
- Na pagina do autor, a data de nascimento esta em
  <span class="author-born-date">.
- Paginacao: igual ao exercicio 1, procure <li class="next"><a href="...">.
  Aqui o href tambem e relativo (ex: "/page/2/").
- Limite a 2 paginas (NUM_PAGES ja esta definido abaixo).

Rode com:
    conda activate science_data
    python src/scraping/exercises/exercicio3_bs4_autores.py
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://quotes.toscrape.com/"
START_URL = "https://quotes.toscrape.com/"
NUM_PAGES = 2

# cache global: url do autor -> data de nascimento, pra nao repetir requests
_author_cache: dict[str, str] = {}


def get_author_born_date(author_url: str) -> str:
    """Recebe a URL absoluta da pagina de um autor e retorna a data de
    nascimento. Usa _author_cache pra nao buscar a mesma pagina duas vezes.
    """
    if author_url in _author_cache:
        return _author_cache[author_url]

    response = requests.get(author_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    born_date = soup.select_one("span.author-born-date").text.strip()

    _author_cache[author_url] = born_date
    return born_date
    
    

  


def parse_quote(quote_div, base_url: str = BASE_URL) -> dict:
    """Recebe uma tag <div class="quote"> e retorna um dict com
    quote_text, author, tags (lista) e author_born.

    TODO: implementar (vai chamar get_author_born_date por dentro).
    """
    quote_text = quote_div.select_one(".text").get_text(strip=True)
    author = quote_div.select_one(".author").get_text(strip=True)
    tags = [tag.get_text(strip=True) for tag in quote_div.select(".tag")]
    author_relative_url = quote_div.select_one("a[href^='/author/']")["href"]
    author_url = requests.compat.urljoin(base_url, author_relative_url)
    author_born = get_author_born_date(author_url)

    return {
        "quote_text": quote_text,
        "author": author,
        "tags": tags,
        "author_born": author_born,
    }


def get_next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    """Retorna a URL absoluta da proxima pagina, ou None se nao houver.

    TODO: implementar (igual ao exercicio 1).
    """
    next_link = soup.select_one("li.next a")
    if next_link is None:
        return None  # ultima pagina do catalogo, nao tem mais "next"

    href = next_link["href"]  # algo tipo "/page/2/" (relativo)
    # urljoin junta a URL atual com o href relativo pra formar uma URL absoluta,
    next_page_url = requests.compat.urljoin(current_url, href)
    return next_page_url


def scrape_quotes(start_url: str = START_URL, num_pages: int = NUM_PAGES) -> list[dict]:
    """Percorre `num_pages` paginas a partir de start_url e retorna a lista
    de dicts de todas as citacoes encontradas.

    TODO: implementar o loop de paginacao (igual ao exercicio 1).
    """
    quotes = []
    url = start_url

    for _ in range(num_pages):
        response = requests.get(url, timeout=10)
        response.encoding = response.apparent_encoding  # corrige o encoding bugado do site
        soup = BeautifulSoup(response.text, "html.parser")

        for quote_div in soup.select(".quote"):
            quotes.append(parse_quote(quote_div))

        url = get_next_page_url(soup, url)
        if url is None:
            break  # nao tem mais paginas

    return quotes


if __name__ == "__main__":
    quotes = scrape_quotes()
    df = pd.DataFrame(quotes)

    df.to_csv("data/processed/citacoes.csv", index=False)

    print(f"Citacoes coletadas: {len(df)}")
    print(f"Autores unicos: {df['author'].nunique()}")
