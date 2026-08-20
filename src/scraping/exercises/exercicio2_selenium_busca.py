"""
EXERCICIO 2 - Webscraping COM Selenium
Site: https://quotes.toscrape.com/search.aspx  (site publico, feito para pratica)

Por que precisa de Selenium aqui?
----------------------------------
Essa pagina tem dois <select>: "author" e "tag". O <select> de "tag" comeca
VAZIO e so e preenchido via AJAX (JavaScript) depois que voce escolhe um
autor no primeiro <select>. Um requests.get() simples nao executa JS, entao
ele nunca veria as opcoes de tag. Por isso precisamos de um navegador de
verdade (Selenium) para: selecionar o autor -> esperar o JS popular o
segundo <select> -> selecionar uma tag -> clicar em "Search" -> ler o
resultado.

OBJETIVO
--------
1. Abrir https://quotes.toscrape.com/search.aspx
2. Selecionar o autor "Albert Einstein" no <select id="author">
3. Esperar o <select id="tag"> ser populado com opcoes (ele comeca so com
   a opcao "----------")
4. Selecionar a primeira tag disponivel (que nao seja "----------")
5. Clicar no botao de busca (<input type="submit" name="submit_button">)
6. Ler as citacoes que aparecem no resultado (elas ficam em <span class="content">
   dentro da area de resultado) e imprimir cada uma
7. Fechar o navegador

DICAS
-----
- Use `from selenium.webdriver.support.ui import Select` para manipular
  <select> por value/texto: Select(elemento).select_by_value("Albert Einstein")
- Depois de selecionar o autor, o <select id="tag"> demora um instante para
  ser atualizado (chamada AJAX). NAO use time.sleep() fixo - use
  WebDriverWait junto com uma expected_condition, por exemplo esperando que
  o <select id="tag"> tenha mais de 1 <option>.
- `from selenium.webdriver.support.ui import WebDriverWait`
  `from selenium.webdriver.support import expected_conditions as EC`
- Para pegar a segunda opcao do <select id="tag"> depois de populado:
  `Select(tag_select).options[1]` (o indice 0 e sempre "----------").
- O botao de busca pode ser encontrado com
  `driver.find_element(By.NAME, "submit_button")`.
- Depois do clique, o resultado aparece na mesma pagina (ou muda de URL,
  confira). As citacoes ficam em elementos com a classe "quote", parecido
  com o que voce ja viu no exemplo `selenium_example.py`.

Rode com:
    conda activate science_data
    python src/scraping/exercises/exercicio2_selenium_busca.py
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://quotes.toscrape.com/search.aspx"
AUTHOR = "Albert Einstein"


def build_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def search_quotes_by_author_and_tag(driver: webdriver.Chrome, author: str = AUTHOR) -> list[str]:
    """Executa os passos 1 a 6 do objetivo e retorna a lista de citacoes
    encontradas (texto de cada uma).

    TODO: implementar.
    """
    raise NotImplementedError


if __name__ == "__main__":
    driver = build_driver()
    try:
        quotes = search_quotes_by_author_and_tag(driver)
        for q in quotes:
            print(q)
    finally:
        driver.quit()
