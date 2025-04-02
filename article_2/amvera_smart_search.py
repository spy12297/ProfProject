import asyncio

from article_2.chroma_db_manager import ChromaConnectionManager
from loguru import logger

chroma_manager = ChromaConnectionManager()


async def example_one():
    await chroma_manager.initialize()
    client = chroma_manager.get_sync_client()
    results = client.similarity_search_with_score(
        query="как у вас запустить Flask приложение?", k=3
    )
    for doc, score in results:
        logger.info(f"Схожесть: {score}")
        logger.info(f"Текст: {doc.page_content}")
        logger.info(f"Метаданные: {doc.metadata}")
        logger.info("-" * 50)


async def example_two():
    await chroma_manager.initialize()
    client = chroma_manager.get_sync_client()

    logger.info("Соединение с ChromaDB установлено. Введите 'exit' для выхода.")

    while True:
        user_query = input("Введите ваш запрос: ")

        if user_query.lower() == "exit":
            logger.info("Закрытие соединения...")
            await chroma_manager.close()
            break

        results = client.similarity_search_with_score(query=user_query, k=3)
        logger.info("Результаты поиска:")
        for doc, score in results:
            data = {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
            }
            logger.info(data)
            logger.info("-" * 50)


async def chat_with_deepseek():
    pass
