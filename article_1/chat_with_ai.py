from typing import Any, Dict, List, Literal, Optional

import torch
from config import settings
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from loguru import logger


class ChatWithAI:
    def __init__(self, provider: Literal["deepseek", "openai"] = "deepseek"):
        self.provider = provider
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.LM_MODEL_NAME,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if provider == "deepseek":
            self.llm = ChatDeepSeek(
                api_key=settings.DEEPSEEK_API_KEY,
                model=settings.DEEPSEEK_MODEL_NAME,
                temperature=0.7,
            )
        elif provider == "openai":
            self.llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL_NAME,
                temperature=0.7,
            )
        else:
            raise ValueError(f"Неподдерживаемый провайдер: {provider}")

        self.chroma_db = Chroma(
            persist_directory=settings.AMVERA_CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name=settings.AMVERA_COLLECTION_NAME,
        )

    def get_relevant_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Получение релевантного контекста из базы данных."""
        try:
            results = self.chroma_db.similarity_search(query, k=k)
            return [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Ошибка при получении контекста: {e}")
            return []

    def format_context(self, context: List[Dict[str, Any]]) -> str:
        """Форматирование контекста для промпта."""
        formatted_context = []
        for item in context:
            metadata_str = "\n".join(f"{k}: {v}" for k, v in item["metadata"].items())
            formatted_context.append(
                f"Текст: {item['text']}\nМетаданные:\n{metadata_str}\n"
            )
        return "\n---\n".join(formatted_context)

    def generate_response(self, query: str) -> Optional[str]:
        """Генерация ответа на основе запроса и контекста."""
        try:
            context = self.get_relevant_context(query)
            if not context:
                return "Извините, не удалось найти релевантный контекст для ответа."

            formatted_context = self.format_context(context)

            messages = [
                {
                    "role": "system",
                    "content": """Ты — внутренний менеджер компании Amvera Cloud. Отвечаешь по делу без лишних вступлений.

Правила:
1. Сразу переходи к сути, без фраз типа "На основе контекста"
2. Используй только факты. Если точных данных нет — отвечай общими фразами об Amvera Cloud, но не придумывай конкретику
3. Используй обычный текст без форматирования
4. Включай ссылки только если они есть в контексте
5. Говори от первого лица множественного числа: "Мы предоставляем", "У нас есть"
6. При упоминании файлов делай это естественно, например: "Я прикреплю инструкцию, где подробно описаны шаги"
7. На приветствия отвечай доброжелательно, на негатив — с легким юмором
8. Можешь при ответах использовать общую информацию из открытых источников по Amvera Cloud, но опирайся на контекст
9. Если пользователь спрашивает о ценах, планах или технических характеристиках — давай конкретные ответы из контекста
10. При технических вопросах предлагай практические решения

Персонализируй ответы, упоминая имя клиента если оно есть в контексте. Будь краток, информативен и полезен.""",
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {query}\nКонтекст: {formatted_context}",
                },
            ]
            response = self.llm.invoke(messages)
            if hasattr(response, "content"):
                return str(response.content)
            return str(response).strip()
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return "Произошла ошибка при генерации ответа."


if __name__ == "__main__":
    chat = ChatWithAI(provider="deepseek")
    print("\n=== Чат с ИИ ===\n")

    while True:
        query = input("Вы: ")
        if query.lower() == "выход":
            print("\nДо свидания!")
            break

        print("\nИИ печатает...", end="\r")
        response = chat.generate_response(query)
        print(" " * 20, end="\r")  # Очищаем "ИИ печатает..."
        print(f"ИИ: {response}\n")
