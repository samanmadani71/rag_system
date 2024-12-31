# FastAPI Application with MongoDB, LangChain, and FAISS

This is a Retrieval-Augmented Generation (RAG) API that uses a Large Language Model (LLM) to answer questions based on a set of documents. The system is scalable, API-driven, and containerized.

---

## Features

- **File Upload**: Upload and index PDF documents.
- **Query Answering**: Retrieve and generate responses based on indexed data.
- **Chat Logs**: Maintain session-specific chat history.
- **FAISS Integration**: Efficient similarity search using LangChain and FAISS.
- **MongoDB**: Robust and scalable database for metadata storage.

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Docker** and **Docker Compose**
- **MongoDB** (if not using Docker)
- **OpenAI API Key** (required for LangChain)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/samanmadani71/rag_system.git
   cd rag_system
   ```

2. Create a virtual environment and activate it:
   ```python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Create an .env file and add your environment variables:

   ```
    OPENAI_API_KEY=your_openai_api_key_here
    MONGO_URI=mongodb://localhost:27017/rag
   ```

5. Run the application:

   ```
   uvicorn main:app --reload
   ```

6. Open your browser and navigate to:

   ```
   http://127.0.0.1:8000/docs
   ```
