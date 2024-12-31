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
from bson.objectid import ObjectId  # For handling the _id field
import json
from langchain_core.prompts import PromptTemplate



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


llm = ChatOpenAI(model='gpt-4o-mini')

prompt = PromptTemplate.from_template('context: {context}\nquery: {query}\n')

chain  = prompt | llm 



sessions = {}

class Query(BaseModel):
    query:str

class FileId(BaseModel):
    file_id:str

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
    all_chunks = []
    filenames = []
    files_path = []
    uniqe_id_timestamp = str(int(time.time()*10000))
    for file in files:
        filename = file.filename
        content_type = file.content_type
        file_path = f'd:/technicalTest/uploaded_files/{filename}'
        _,file_extension = os.path.splitext(filename)
        if file_extension.lower() not in allowed_extensions:
            raise HTTPException(status_code=400,detail=f"Invalid file type: {file_extension}. Only .pdf and .doc files are allowed.")

        if os.path.exists(file_path):
            unique_id = uniqe_id_timestamp
            filename = f'{unique_id}_{filename}'
            file_path = f'd:/technicalTest/uploaded_files/{filename}'

        files_path.append(file_path)
        filenames.append(filename)
        with open(file_path,'wb') as buffer:
            shutil.copyfileobj(file.file,buffer)

        pdf_text = parse_pdf(file_path)
        chunks = chunk_text(pdf_text)
        all_chunks.extend(chunks)
    chunks_path = f'd:/technicalTest/uploaded_files/{uniqe_id_timestamp}_chunks.json'
    with open(chunks_path, 'w') as chunk_file:
        json.dump(all_chunks, chunk_file)

    chunk_embeddings = [embedding_model.embed_query(chunk) for chunk in all_chunks]
    embedding_dim = len(chunk_embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    embeddings_np = np.array(chunk_embeddings).astype('float32')
    index.add(embeddings_np)
    index_path = f'd:/technicalTest/index/{uniqe_id_timestamp}.bin'
    faiss.write_index(index,index_path)

    file_metadata = {'session_id':session_id,'filenames':filenames,'index_path':index_path,'chunks_path':chunks_path,'content_type':content_type,'files_path':files_path,'uploaded_at':datetime.datetime.now(timezone.utc)}
    result = await collection.insert_one(file_metadata)
    uploaded_files.append({'file_id':str(result.inserted_id),'filenames':filenames,'content_type':content_type,'files_path':files_path,'index_path':index_path})
    return {'uploaded_files':uploaded_files}

@app.delete('/upload')
async def delete_uploaded_file(request:Request,response:Response,file_id:FileId):
    session_id = get_or_create_session_id(request,response)
    
    try:
        file_id_obj = ObjectId(file_id.file_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file_id format")

    record = await collection.find_one(
        {"session_id": session_id, "_id": file_id_obj}
    )

    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    if files_path:=record.get('files_path'):
        for file_path in files_path:
            if os.path.exists(file_path):
                os.remove(file_path)
    if index_path:=record.get('index_path'):
        if os.path.exists(index_path):
            os.remove(index_path)
    if chunks_path:=record.get('chunks_path'):
        if os.path.exists(chunks_path):
            os.remove(chunks_path)

    delete_result  = await collection.delete_one({'_id':file_id_obj})
    if delete_result.deleted_count == 1:
        return {
            "message": "File and metadata deleted successfully",
            "file_id": file_id,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to delete metadata from the database")


@app.post('/query')
async def query_answering(request:Request,response:Response,query:Query):
    start_time = time.time()
    
    session_id = get_or_create_session_id(request,response)

    cursor = collection.find({'session_id':session_id},{'_id':0,'index_path':1,'chunks_path':1}).sort('uploaded_at',-1).limit(1)

    record = await cursor.to_list(length=1)

    main_context=''
    if record:
        record = record[0]
        index_path = record.get('index_path')
        chunks_path = record.get('chunks_path')


        all_chunks = []
        with open(chunks_path, 'r') as chunk_file:
            all_chunks.extend(json.load(chunk_file))

        query = query.query
        query_embedding = embedding_model.embed_query(query) 
        
        index = faiss.read_index(index_path)
        k=2
        distances,indices = index.search(np.array([query_embedding]).astype('float32'),k)
        relevant_chunks = [all_chunks[i] for i in indices[0]]
        context = '\n'.join(relevant_chunks)
        main_context+=context

    response = chain.invoke({'context':main_context,'query':query})

    await chat_collection.insert_one({'session_id':session_id,'query':query,'response':response.content,'context':main_context,'timestamp':datetime.datetime.now(timezone.utc),'duration':time.time()-start_time})
    return {'answer':response}

@app.get('/logs')
def get_logs():
    return {'logs':'a history of logs'}