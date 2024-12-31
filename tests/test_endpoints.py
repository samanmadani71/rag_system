import os
import pytest

from fastapi.testclient import TestClient
from main import app,get_database
from motor.motor_asyncio import AsyncIOMotorClient

client = TestClient(app)



TEST_DB_NAME = "rag_test"
TEST_COLLECTION_NAME = "test_file_metadata"
TEST_CHAT_COLLECTION_NAME = "test_chats"



@pytest.fixture(scope='function')
async def test_db():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["rag_test"]


    # Provide the test collections
    yield db
    await client.drop_database(TEST_DB_NAME)

@pytest.mark.asyncio
async def test_upload_and_query(test_db):
    """Test upload and query endpoints with automatic cleanup."""

    app.dependency_overrides[get_database] = lambda: test_db
    # Simulate file upload
    test_file_path = os.path.join('tests','assets','test.pdf')
    with open(test_file_path, "rb") as file:
        file_data = {"files": ("test.pdf", file, "application/pdf")}
        upload_response = client.post("/upload", files=file_data)
        assert upload_response.status_code == 200
        assert "uploaded_files" in upload_response.json()

        # Verify document insertion in MongoDB
        collection = test_db['test_file_metadata']
        uploaded_file = await collection.find_one({"filenames": ["test.pdf"]})
        assert uploaded_file is not None

