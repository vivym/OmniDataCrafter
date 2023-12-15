from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from .models import gather_documents
from .settings import settings


async def init_db() -> None:
    client = AsyncIOMotorClient(settings.mongo_uri)

    await init_beanie(
        database=getattr(client, settings.mongo_db),
        document_models=gather_documents(),
    )
