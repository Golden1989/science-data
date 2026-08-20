"""Exemplo de webscraping com Selenium (util para paginas renderizadas via JS)."""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://quotes.toscrape.com/js/"


def get_quotes(url: str = URL, headless: bool = True) -> list[dict]:
    options = Options()
    if headless:
        options.add_argument("--headless=new")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        quotes = []
        for card in driver.find_elements(By.CLASS_NAME, "quote"):
            text = card.find_element(By.CLASS_NAME, "text").text
            author = card.find_element(By.CLASS_NAME, "author").text
            tags = [t.text for t in card.find_elements(By.CLASS_NAME, "tag")]
            quotes.append({"text": text, "author": author, "tags": tags})
        return quotes
    finally:
        driver.quit()


if __name__ == "__main__":
    for quote in get_quotes():
        print(f"{quote['author']}: {quote['text']}")
