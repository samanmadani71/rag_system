from motor.motor_asyncio import AsyncIOMotorClient
import asyncio


MONGO_URI = "mongodb://localhost:27017/chatbot"
DATABASE_NAME = 'rag'

client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]

async def check_connection():
    try:
        collections = await db.list_collection_names()
        print(f'successfully connected to {DATABASE_NAME} database. Collections: {collections}')
    except Exception as e:
        print(f'error connecting to {DATABASE_NAME} database {e}')

    
asyncio.run(check_connection())