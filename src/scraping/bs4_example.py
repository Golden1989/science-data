"""Exemplo simples de webscraping com requests + BeautifulSoup."""
import requests
from bs4 import BeautifulSoup

URL = "https://quotes.toscrape.com/"


def get_quotes(url: str = URL) -> list[dict]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    quotes = []
    for card in soup.select(".quote"):
        text = card.select_one(".text").get_text(strip=True)
        author = card.select_one(".author").get_text(strip=True)
        tags = [tag.get_text(strip=True) for tag in card.select(".tags .tag")]
        quotes.append({"text": text, "author": author, "tags": tags})
    return quotes


if __name__ == "__main__":
    for quote in get_quotes():
        print(f"{quote['author']}: {quote['text']}")
        
    """
    1. select_one funciona igual pra QUALQUER tag

Essa é a parte importante: não importa se é <span>, <p>, <div>, <a> — o select_one() usa "seletor CSS", que é a mesma linguagem que você usa quando inspeciona a página no Chrome. A tag em si quase nunca importa, o que importa é a classe ou o atributo.

Exemplo do exercício 1 (já resolvido, olha o parse_book):
price_text = article.select_one(".price_color").get_text(strip=True)
Aqui .price_color é uma classe. O bs4 procura qualquer tag que tenha class="price_color". Não importa se é <p class="price_color"> ou <span class="price_color"> — o . na frente diz "procure por classe", e funciona igual.

Agora no exercício 5, a dica diz que o preço está em:
<span class="price">$69.00</span>
Então é exatamente a mesma sintaxe:
price_text = card.select_one(".price").get_text(strip=True)
Só troca o nome da classe (.price_color → .price). O fato de ser <span> em vez de <p> não muda nada no código.

Se quiser ser mais específico (opcional, não obrigatório), dá pra escrever "span.price" — tag colada com a classe, sem espaço — mas normalmente só .price já resolve, porque não tem outra coisa com essa classe na página.

2. Pegar texto vs pegar atributo

Duas operações diferentes que se confundem:

- Texto visível (o que aparece escrito na te
- Atributo (algo escrito dentro da tag, tipo title="..." ou href="..."): ["nome_do_atributo"]

No exercício 1:
title = article.h3.a["title"]          # ATRIBUTO title="..."
availability = article.select_one(".availability").get_text(strip=True)  # TEXTO

No exercício 4, a dica fala que o rating vem como atributo:
<p data-rating="3">...</p>
Isso é atributo, não texto. Então:
rating = card.select_one("p")["data-rating"]
Repara: primeiro eu acho a tag com select_one, e SÓ DEPOIS pego o atributo com [...]. São dois passos separados, sempre nessa ordem: acha a tag → pega o que quer dela (texto ou atributo).

3. Os "fors" — na verdade tem só DOIS tipos, em lugares diferentes

Isso é o que mais confunde, então presta ate

For tipo A — dentro de parse_item, só quando tem MÚLTIPLOS valores pra um mesmo item. Exemplo real, do exercício 3,pegando várias tags de uma citação:
tags = [tag.get_text(strip=True) for tag in quote_div.select(".tag")]
Isso só existe porque uma citação pode ter várias tags. No exercício 4 e 5, nenhum campo seu precisa disso — name, price, rating, reviews, product_url são todoão dentro do seu parse_item você não vai usar nenhum for, só uma sequência de select_one + ["attr"] ou .get_text(), um por linha, igual o parse_book do exercício 1. Isso deve tirar uma das suas travas.

For tipo B — fica em scrape_laptops/scrape_products, NÃO em parse_item. Esse for percorre a lista de cards da página echama parse_item uma vez pra cada card:
for card in soup.select("div.thumbnail"):
    laptops.append(parse_item(card))
Isso é exatamente o que já está pronto no scrape_quotes do exercício 3 (linha for quote_div in soup.select(".quote"): quotes.append(parse_quote(quote_div))) e no scrape_books do exercício 1. Você pode copiar essa estrutura quase igual, só trocando o seletor do card (.thumbnail no da função.
    Resumo do padrão que talvez ainda não tinha ficado claro: atributo tipo itemprop="x" ou data-x="y" sempre vira [nome="valor"] no seletor, nunca .nome. O . é só pra classe CSS de verdade (class="...").
    Pra parse_item em qualquer um dos dois exercícios, o "molde" é sempre:
def parse_item(card) -> dict:
    campo1 = card.select_one("SELETOR_CSS").get_text(strip=True)   # se for texto
    campo2 = card.select_one("SELETOR_CSS")["ATRIBUTO"]            # se for atributo
    ...
    return {"campo1": campo1, "campo2": campo2, ...}
Sem for nenhum aí dentro nos exercícios 4 e 5. O for só aparece em scrape_laptops/scrape_products, percorrendo os cards e (no fim do loop) trocando de página.
    """
