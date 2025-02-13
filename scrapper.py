import os
import re
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from playwright.async_api import async_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

# Загружаем переменные окружения
load_dotenv()

# Конфигурация
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "nav-2"
BASE_URL = "https://docs.nav2.org/"  # Замените на URL вашего сайта
EMBEDDING_DIMENSION = 1536  # Размерность embeddings (зависит от модели)
OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"  # URL для Ollama API

# Инициализация Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

class Scraper:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Размер фрагмента текста
            chunk_overlap=200,  # Перекрытие между фрагментами
            length_function=len,
        )
        self.embeddings = OllamaEmbeddings(model="deepseek-r1")  # Инициализация модели для embeddings

        # Проверяем, существует ли индекс в Pinecone
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            # Если индекс не существует, выводим в логи
            print ("Идекс не существует в Pinecone")
            )

        # Подключаемся к индексу
        self.pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    def clean_text(self, text: str) -> str:
        """Очистка текста от лишних символов и форматирования."""
        text = re.sub(r'\s+', ' ', text)  # Удаляем лишние пробелы
        text = re.sub(r'\n+', '\n', text)  # Удаляем лишние переносы строк
        return text.strip()

    async def get_embeddings(self, text: str) -> List[float]:
        """Получение векторных embeddings с помощью Ollama API."""
        data = {
            "model": "deepseek-r1",  # Используемая модель
            "prompt": text,
            "temperature": 0.7,
            "max_tokens": 100
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(OLLAMA_URL, json=data, timeout=30.0)
                response.raise_for_status()
                return response.json().get("embedding", [])
        except Exception as e:
            print(f"Ошибка Ollama API: {e}")
            return []

    async def get_links(self, page) -> List[str]:
        """Извлечение всех ссылок на документацию со страницы."""
        selector = "a"
        links = await page.locator(selector).evaluate_all("elements => elements.map(e => e.href)")
        print(f"Найдено ссылок: {len(links)}")
        return [link for link in links if link.startswith(BASE_URL)]

    async def scrape_page(self, page, url: str) -> str:
        """Сбор текста с одной страницы."""
        print(f"Обработка страницы: {url}")
        await page.goto(url, timeout=90000)
        await page.wait_for_selector("body", timeout=120000)
        content = await page.locator("body").inner_text()
        return self.clean_text(content)

    async def process_page(self, page, url: str):
        """Обработка одной страницы и загрузка данных в Pinecone."""
        try:
            # Сбор текста с текущей страницы
            content = await self.scrape_page(page, url)

            # Разделение текста на фрагменты
            chunks = self.text_splitter.split_text(content)

            # Создание векторов и метаданных
            vectors = []
            for i, chunk in enumerate(chunks):
                embedding = await self.get_embeddings(chunk)

                if not embedding:
                    print(f"Пропуск фрагмента {i} из-за пустого embedding")
                    continue

                vectors.append((
                    f"{url}-{i}",  # Уникальный ID для фрагмента
                    embedding,  # Векторное представление текста
                    {
                        "text": chunk,  # Оригинальный текст
                        "url": url,  # URL страницы
                        "source": "tech-docs"  # Источник данных
                    }
                ))

            # Загрузка данных в Pinecone
            if vectors:
                self.pinecone_index.upsert(vectors=vectors)
                print(f"Загружено {len(vectors)} фрагментов с {url}")

        except Exception as e:
            print(f"Ошибка при обработке {url}: {str(e)}")

    async def process_documentation(self):
        """Основной процесс сбора данных и загрузки в Pinecone."""
        async with async_playwright() as p:
            # Запуск браузера
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(BASE_URL, timeout=90000)  # Переход на главную страницу

            # Получаем все ссылки на документацию
            links = await self.get_links(page)

            # Обрабатываем каждую страницу асинхронно
            tasks = [self.process_page(page, url) for url in links]
            await asyncio.gather(*tasks)

            # Закрытие браузера
            await browser.close()

if __name__ == "__main__":
    scraper = Scraper()
    asyncio.run(scraper.process_documentation())