import os
import re
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from playwright.sync_api import sync_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Загружаем переменные окружения
load_dotenv()

# Конфигурация
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "nav2-docs"
BASE_URL = "https://docs.nav2.org/"
EMBEDDING_DIMENSION = 1536  # DeepSeek R1 7B typically uses 1536 dimensions
OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"  # Endpoint for embeddings

# Инициализация Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

class Scraper:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Проверяем, существует ли индекс в Pinecone
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            # Если индекс не существует, создаем его
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Подключаемся к индексу
        self.pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    def clean_text(self, text: str) -> str:
        """Очистка текста от лишних символов и форматирования."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """Получение векторных embeddings с помощью Ollama API."""
        data = {
            "model": "deepseek-r1",  # Используемая модель
            "prompt": text,
            "temperature": 0.7,
            "max_tokens": 100
        }

        try:
            response = requests.post(OLLAMA_URL, json=data, timeout=10)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("embedding", [])  # Извлекаем embeddings
        except requests.exceptions.RequestException as e:
            print(f"Ошибка Ollama API: {e}")
            return None

    def get_links(self, page) -> List[str]:
        """Извлечение всех ссылок на документацию со страницы."""
        selector = "a"
        links = page.locator(selector).evaluate_all("elements => elements.map(e => e.href)")
        print(f"Найдено ссылок: {len(links)}")
        return [link for link in links if link.startswith(BASE_URL)]

    def scrape_page(self, page, url: str) -> Optional[Dict[str, Any]]:
        """Сбор структурированных данных с одной страницы."""
        print(f"Обработка страницы: {url}")
        try:
            page.goto(url, timeout=90000)
            page.wait_for_selector("body", timeout=120000)

            # Извлечение всех заголовков <h1>
            h1_elements = page.locator("h1").all()
            titles = [h1.inner_text() for h1 in h1_elements] if h1_elements else ["Без заголовка"]

            # Используем первый заголовок (или "Без заголовка", если их нет)
            title = titles[0]

            # Извлечение основного текста
            content = page.locator("body").inner_text()
            cleaned_content = self.clean_text(content)

            # Возвращаем структурированные данные
            return {
                "url": url,
                "title": title,
                "content": cleaned_content
            }
        except Exception as e:
            print(f"Ошибка при обработке {url}: {e}")
            return None
    

    def process_documentation(self):
        """Основной процесс сбора данных и загрузки в Pinecone."""
        with sync_playwright() as p:
            # Запуск браузера
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(BASE_URL, timeout=90000)  # Переход на главную страницу
            
            links = self.get_links(page)

            # Обрабатываем каждую страницу
            for url in links:
                print(f"Обработка: {url}")
                page_data = self.scrape_page(page, url)
                if not page_data:
                    continue

                # Разделение текста на фрагменты
                chunks = self.text_splitter.split_text(page_data["content"])

                # Создание векторов и метаданных
                vectors = []
                for i, chunk in enumerate(chunks):
                    embedding = self.get_embeddings(chunk)
                    
                    if not embedding:
                        print(f"Пропуск фрагмента {i} из-за пустого embedding")
                        continue
                    
                    vectors.append((
                        f"{url}-{i}",  # Уникальный ID для фрагмента
                        embedding,  # Векторное представление текста
                        {
                            "text": chunk,
                            "url": page_data["url"],
                            "title": page_data["title"],
                            "source": PINECONE_INDEX_NAME
                        }
                    ))
                
                if vectors:
                    self.pinecone_index.upsert(vectors=vectors)
                    print(f"Загружено {len(vectors)} фрагментов с {url}")
            
            # Закрытие браузера после обработки всех страниц
            browser.close()

if __name__ == "__main__":
    scraper = Scraper()
    scraper.process_documentation()