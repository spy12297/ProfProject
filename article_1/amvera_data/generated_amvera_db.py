import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

# Добавляем путь к родительской директории для импорта config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


def load_json_files(directory: str) -> List[Dict[str, Any]]:
    """Загрузка всех JSON файлов из указанной директории."""
    documents = []

    try:
        if not os.path.exists(directory):
            logger.error(f"Директория {directory} не существует")
            return documents

        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        documents.append(
                            {"text": data["text"], "metadata": data["metadata"]}
                        )
                        logger.info(f"Загружен файл: {filename}")
                except Exception as e:
                    logger.error(f"Ошибка при чтении файла {filename}: {e}")

        logger.success(f"Загружено {len(documents)} JSON файлов")
        return documents
    except Exception as e:
        logger.error(f"Ошибка при загрузке JSON файлов: {e}")
        return documents


def split_text_into_chunks(text: str, metadata: Dict[str, Any]) -> List[Any]:
    """Разделение текста на чанки с сохранением метаданных."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.MAX_CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.create_documents(texts=[text], metadatas=[metadata])
    return chunks


def generate_chroma_db() -> Optional[Chroma]:
    """Инициализация ChromaDB с данными из JSON файлов."""
    try:
        # Создаем директорию для хранения базы данных, если она не существует
        os.makedirs(settings.AMVERA_CHROMA_PATH, exist_ok=True)

        # Загружаем JSON файлы
        documents = load_json_files(settings.PARSED_JSON_PATH)

        if not documents:
            logger.warning("Нет документов для добавления в базу данных")
            return None

        # Инициализируем модель эмбеддингов
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.LM_MODEL_NAME,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Подготавливаем данные для Chroma
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = split_text_into_chunks(doc["text"], doc["metadata"])
            all_chunks.extend(chunks)
            logger.info(
                f"Документ {i+1}/{len(documents)} разбит на {len(chunks)} чанков"
            )

        # Создаем векторное хранилище
        texts = [chunk.page_content for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        ids = [f"doc_{i}" for i in range(len(all_chunks))]

        chroma_db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            ids=ids,
            metadatas=metadatas,
            persist_directory=settings.AMVERA_CHROMA_PATH,
            collection_name=settings.AMVERA_COLLECTION_NAME,
            collection_metadata={
                "hnsw:space": "cosine",
            },
        )

        logger.success(
            f"База Chroma инициализирована, добавлено {len(all_chunks)} чанков из {len(documents)} документов"
        )
        return chroma_db
    except Exception as e:
        logger.error(f"Ошибка инициализации Chroma: {e}")
        raise


if __name__ == "__main__":
    generate_chroma_db()
