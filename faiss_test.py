from lib.utils import parse_pdf,chunk_text
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import faiss
import numpy as np
import os, getpass

from langchain_core.prompts import PromptTemplate

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')


pdf_text = parse_pdf(r'd:/technicalTest/uploaded_files/bigbook.pdf')
chunks = chunk_text(pdf_text)

print(f'extracted {len(chunks)} chunks from the pdf')

# chunk_embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]


# embedding_dim = len(chunk_embeddings[0])
# index = faiss.IndexFlatL2(embedding_dim)
# embeddings_np = np.array(chunk_embeddings).astype('float32')
# index.add(embeddings_np)

# print(f'Added {index.ntotal} vectors to the FAISS index.')

# faiss.write_index(index,'d:/technicalTest/index/faiss_index.bin')


# query = 'explain if the resume of Saman Madani is good for senior full-stack developer position or not?'
# query_embedding = embedding_model.embed_query(query)    

# k =3
# distances,indices = index.search(np.array([query_embedding]).astype('float32'),k)

# relevant_chunks = [chunks[i] for i in indices[0]]
# context = '\n'.join(relevant_chunks)

# print(f'Context for LLM:\n {context}')

# if not os.environ.get('OPENAI_API_KEY'):
#     os.environ['OPENAI_API_KEY'] = getpass.getpass('Enter your OpenAI API key: ')


# # llm = OpenAI(model='gpt-4o-mini')
# llm = ChatOpenAI(model='gpt-4o-mini')

# prompt = PromptTemplate.from_template('context: {context}\nquery: {query}\n')

# chain  = prompt | llm 


# response = chain.invoke({'context':context,'query':query})

# print(response)