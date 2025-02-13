import os
import re
import asyncio
import httpx
import random
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from playwright.async_api import async_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Загружаем переменные окружения
load_dotenv()

# Конфигурация
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "nav2-docs"
BASE_URL = "https://docs.nav2.org/"
EMBEDDING_DIMENSION = 1536
OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"

# Инициализация Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)


class Scraper:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.embeddings = OllamaEmbeddings(model="deepseek-r1")

        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print("Индекс не существует в Pinecone")
            return

        self.pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    async def get_embeddings(self, text: str) -> List[float]:
        data = {
            "model": "deepseek-r1", "prompt": text, "temperature": 0.7, "max_tokens": 100
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(OLLAMA_URL, json=data, timeout=60.0)
                response.raise_for_status()
                return response.json().get("embedding", [])
        except Exception as e:
            print(f"Ошибка Ollama API: {e}")
            return []

    async def get_links(self, page) -> List[str]:
        links = await page.locator("a").evaluate_all("elements => elements.map(e => e.href)")
        return [
            link for link in links
            if link.startswith(BASE_URL) and not link.endswith(('.png', '.jpg', '.jpeg', '.gif', '#'))
        ]

    async def scrape_page(self, page, url: str) -> str:
        print(f"Обработка страницы: {url}")
        clean_url = url.split("#")[0]  # Удаляем хэш из URL

        for attempt in range(3):  # Повторяем попытку 3 раза
            try:
                # Устанавливаем User-Agent для имитации реального браузера
                await page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                })

                response = await page.goto(clean_url, timeout=900000, wait_until="networkidle")

                # Проверка статуса ответа
                if response and response.status != 200:
                    print(f"Ошибка: статус {response.status} для {clean_url}")
                    continue

                # Эмуляция поведения пользователя: случайная прокрутка
                await self.emulate_user_behavior(page)

                content = await page.content()
                if content:
                    return self.clean_text(content)
            except Exception as e:
                print(f"Ошибка при загрузке {clean_url} (попытка {attempt + 1}): {e}")
                await asyncio.sleep(5)  # Пауза перед повторной попыткой
        return ""

    async def emulate_user_behavior(self, page):
        """Эмуляция поведения пользователя: случайная прокрутка и задержки."""
        # Случайная прокрутка страницы
        scroll_steps = random.randint(1, 5)
        for _ in range(scroll_steps):
            await page.mouse.wheel(0, random.randint(200, 800))  # Прокрутка вниз
            await asyncio.sleep(random.uniform(0.5, 2.0))  # Случайная задержка

    async def process_page(self, page, url: str):
        content = await self.scrape_page(page, url)
        if not content:
            print(f"Пропуск страницы {url} из-за отсутствия контента")
            return

        chunks = self.text_splitter.split_text(content)
        vectors = []

        for i, chunk in enumerate(chunks):
            embedding = await self.get_embeddings(chunk)
            if embedding:
                vectors.append((
                    f"{url}-{i}", embedding, {"text": chunk, "url": url, "source": PINECONE_INDEX_NAME}
                ))

        if vectors:
            self.pinecone_index.upsert(vectors=vectors)
            print(f"Загружено {len(vectors)} фрагментов с {url}")

    async def process_documentation(self):
        async with async_playwright() as p:
            # Запуск браузера с настройками для имитации реального пользователя
            browser = await p.chromium.launch(headless=False)  # headless=False для визуального отображения
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = await context.new_page()

            await page.goto(BASE_URL, timeout=900000, wait_until="networkidle")
            links = await self.get_links(page)

            tasks = [self.process_page(page, url) for url in links]
            await asyncio.gather(*tasks)

            await browser.close()


if __name__ == "__main__":
    scraper = Scraper()
    asyncio.run(scraper.process_documentation())