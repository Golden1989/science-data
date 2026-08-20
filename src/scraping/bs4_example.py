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
