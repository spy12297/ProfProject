from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import torch
from config import settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


class ChromaConnectionManager:
    def __init__(self):
        self.CHROMA_PATH = settings.AMVERA_CHROMA_PATH
        self.COLLECTION_NAME = settings.AMVERA_COLLECTION_NAME
        self.MODEL_NAME = settings.LM_MODEL_NAME
        self._chroma_db: Optional[Chroma] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._is_initialized = False

    async def initialize(self):
        """Инициализация подключения (вызывается при старте приложения)"""
        if not self._is_initialized:
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.MODEL_NAME,
                    model_kwargs={
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    },
                    encode_kwargs={"normalize_embeddings": True},
                )

                self._chroma_db = Chroma(
                    persist_directory=self.CHROMA_PATH,
                    embedding_function=self._embeddings,
                    collection_name=self.COLLECTION_NAME,
                )
                self._is_initialized = True
                logger.success("Chroma connection initialized")
            except Exception as e:
                logger.error(f"Chroma initialization failed: {e}")
                raise

    async def close(self):
        """Закрытие подключения (вызывается при остановке приложения)"""
        if self._is_initialized:
            try:
                # Chroma не требует явного закрытия, но очищаем ресурсы
                self._chroma_db = None
                self._embeddings = None
                self._is_initialized = False
                logger.info("Chroma connection closed")
            except Exception as e:
                logger.error(f"Error closing Chroma connection: {e}")
                raise

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[Optional[Chroma], Any]:
        """Async контекстный менеджер для получения клиента"""
        if not self._is_initialized:
            await self.initialize()

        try:
            yield self._chroma_db
        except Exception as e:
            logger.error(f"Error during Chroma operation: {e}")
            raise

    def get_sync_client(self) -> Optional[Chroma]:
        """Синхронный метод для получения клиента (для использования в синхронных контекстах)"""
        if not self._is_initialized:
            raise RuntimeError("Chroma connection not initialized")
        return self._chroma_db
