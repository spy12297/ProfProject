import time
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from loguru import logger
from langchain_chroma import Chroma


CHROMA_PATH = "./shop_chroma_db"
COLLECTION_NAME = "shop_data"

SHOP_DATA = [
    {
        "text": 'Ноутбук Lenovo IdeaPad 5: 16 ГБ RAM, SSD 512 ГБ, экран 15.6", цена 55000 руб.',
        "metadata": {
            "id": "1",
            "type": "product",
            "category": "laptops",
            "price": 55000,
            "stock": 3,
        },
    },
    {
        "text": "Смартфон Xiaomi Redmi Note 12: 128 ГБ, камера 108 МП, цена 18000 руб.",
        "metadata": {
            "id": "2",
            "type": "product",
            "category": "phones",
            "price": 18000,
            "stock": 10,
        },
    },
    {
        "text": "Планшет Samsung Galaxy Tab A8: 10.5 дюймов, 64 ГБ, цена 22000 руб.",
        "metadata": {
            "id": "3",
            "type": "product",
            "category": "tablets",
            "price": 22000,
            "stock": 5,
        },
    },
    {
        "text": "Наушники Sony WH-1000XM4: шумоподавление, 30 часов работы, цена 28000 руб.",
        "metadata": {
            "id": "4",
            "type": "product",
            "category": "audio",
            "price": 28000,
            "stock": 7,
        },
    },
    {
        "text": "Умные часы Apple Watch Series 8: GPS, пульсометр, цена 35000 руб.",
        "metadata": {
            "id": "5",
            "type": "product",
            "category": "wearables",
            "price": 35000,
            "stock": 4,
        },
    },
    {
        "text": "Игровая консоль PlayStation 5: 825 ГБ SSD, 4K HDR, цена 49990 руб.",
        "metadata": {
            "id": "6",
            "type": "product",
            "category": "gaming",
            "price": 49990,
            "stock": 2,
        },
    },
    {
        "text": "Фотоаппарат Canon EOS 250D: 24.1 МП, 4K видео, цена 42000 руб.",
        "metadata": {
            "id": "7",
            "type": "product",
            "category": "cameras",
            "price": 42000,
            "stock": 3,
        },
    },
    {
        "text": "Электросамокат Xiaomi Mi Electric Scooter Pro 2: 25 км/ч, запас хода 45 км, цена 39990 руб.",
        "metadata": {
            "id": "8",
            "type": "product",
            "category": "transport",
            "price": 39990,
            "stock": 6,
        },
    },
    {
        "text": "Робот-пылесос Roborock S7: влажная и сухая уборка, цена 32000 руб.",
        "metadata": {
            "id": "9",
            "type": "product",
            "category": "appliances",
            "price": 32000,
            "stock": 8,
        },
    },
    {
        "text": "Кофемашина DeLonghi Magnifica S: автоматическая, 15 бар, цена 29990 руб.",
        "metadata": {
            "id": "10",
            "type": "product",
            "category": "kitchen",
            "price": 29990,
            "stock": 4,
        },
    },
]


def generate_chroma_db():
    try:
        start_time = time.time()

        logger.info("Загрузка модели эмбеддингов...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Модель загружена за {time.time() - start_time:.2f} сек")

        logger.info("Создание Chroma DB...")
        chroma_db = Chroma.from_texts(
            texts=[item["text"] for item in SHOP_DATA],
            embedding=embeddings,
            ids=[str(item["metadata"]["id"]) for item in SHOP_DATA],
            metadatas=[item["metadata"] for item in SHOP_DATA],
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
        )
        logger.info(f"Chroma DB создана за {time.time() - start_time:.2f} сек")

        return chroma_db
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise


generate_chroma_db()
