from typing import List
from fastapi import FastAPI,File,UploadFile,Depends,HTTPException
from fastapi.requests import Request
from fastapi.responses import Response
import shutil
from pydantic import BaseModel
import datetime
from datetime import timezone
from motor.motor_asyncio import AsyncIOMotorClient
from uuid import uuid4
import redis
import os
import time
from .lib.utils import parse_pdf,chunk_text
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import faiss
import numpy as np
import getpass



embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

app = FastAPI()


REDIS_HOST = '45.41.206.68'
REDIS_PORT = 6379
REDIS_PASSWORD = 'samanmadani'


redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True  # Decodes byte responses to strings
)

client = AsyncIOMotorClient('mongodb://localhost:27017')
db = client['rag']
collection = db['file_metadata']
chat_collection = db['chats']


sessions = {}

class Query(BaseModel):
    query:str

def get_or_create_session_id(request:Request,response:Response):
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid4())
        response.set_cookie(key='session_id',value=session_id,httponly=True,secure=True,expires=60*60)
    return session_id


@app.get('/')
def read_root(request:Request,response:Response):
    session_id = get_or_create_session_id(request,response)
    if not session_id:
        raise HTTPException(status_code=403,detail='Unauthenticated')

    return {'message':'welcome user'}






@app.post('/upload')
async def upload_documents(request:Request,response:Response,files:List[UploadFile] = File(...)):
    session_id = get_or_create_session_id(request,response)

    allowed_extensions = ['.pdf','.doc']



    uploaded_files = []
    for file in files:
        filename = file.filename
        content_type = file.content_type
        file_path = f'./uploaded_files/{filename}'
        _,file_extension = os.path.splitext(filename)
        if file_extension.lower() not in allowed_extensions:
            raise HTTPException(status_code=400,detail=f"Invalid file type: {file_extension}. Only .pdf and .doc files are allowed.")

        if os.path.exists(file_path):
            unique_id = str(int(time.time()*10000))
            filename = f'{unique_id}_{filename}'
            file_path = f'./uploaded_files/{filename}'


        with open(file_path,'wb') as buffer:
            shutil.copyfileobj(file.file,buffer)

        pdf_text = parse_pdf(file_path)
        chunks = chunk_text(pdf_text)
        filename_without_extension,_ = os.path.splitext(filename)

        chunk_embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
        embedding_dim = len(chunk_embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)
        embeddings_np = np.array(chunk_embeddings).astype('float32')
        index.add(embeddings_np)
        index_path = f'd:/technicalTest/index/{filename_without_extension}.bin'
        faiss.write_index(index,index_path)

        file_metadata = {'session_id':session_id,'filename':filename,'index_path':index_path,'content_type':content_type,'file_path':file_path,'uploaded_at':datetime.datetime.now(timezone.utc)}
        result = await collection.insert_one(file_metadata)
        uploaded_files.append({'file_id':str(result.inserted_id),'filename':filename,'content_type':content_type,'file_path':file_path,'index_path':index_path})
    return {'uploaded_files':uploaded_files}

@app.post('/query')
def query_answering(request:Request,response:Response,query:Query):
    
    session_id = get_or_create_session_id(request,response)

    

    query = query.query



    return {'answer':query.query}

@app.get('/logs')
def get_logs():
    return {'logs':'a history of logs'}