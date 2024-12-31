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

You can either install the app manually or use Docker to quickly set it up and run.

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

#### Using Docker

1. Build and run the application using Docker Compose:

   ```
   docker-compose build
   docker-compose up
   ```

2. Access the application at:
   ```
   http://localhost:8000/docs
   ```

---

## Endpoints

### **File Upload**

- **URL**: `/upload`
- **Method**: `POST`
- **Description**: Upload and index a PDF document.
- **Request Body**:
  - `files`: List of PDF files.
- **Response**:
  ```json
  {
    "uploaded_files": [
      {
        "file_id": "63e8b2f1",
        "filenames": ["example.pdf"],
        "content_type": "application/pdf",
        "files_path": ["uploaded_files/example.pdf"],
        "index_path": "index/example.bin"
      }
    ]
  }
  ```

### **Query Answering**

- **URL**: `/query`
- **Method**: `POST`
- **Description**: Retrieve answers based on indexed data.
- **Request Body**:
  - `query`: The user query.
- **Response**:
  ```json
  {
    "answer": "AI-generated response based on the context."
  }
  ```

### **Chat Logs**

- **URL**: `/logs`
- **Method**: `GET`
- **Description**: Retrieve chat logs for the current session.
- **Response**:
  ```json
  [
    {
      "query": "What is AI?",
      "response": "AI is artificial intelligence.",
      "context": "Context retrieved from documents.",
      "timestamp": "2023-01-01T12:00:00Z",
      "duration": 0.2
    }
  ]
  ```

---

## Project Structure

```plaintext
.
├── main.py                # Application entry point
├── lib/
│   ├── utils.py           # Helper functions for PDF parsing and text processing
│   └── __init__.py        # Package initialization
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker setup
├── docker-compose.yml     # Docker Compose configuration
├── uploaded_files/        # Directory for uploaded files (Docker volume)
└── README.md              # Project documentation
```

---

## Technologies Used

- **FastAPI**: Backend framework.
- **MongoDB**: Database for storing metadata and chat logs.
- **LangChain**: Framework for managing and querying indexed embeddings.
- **FAISS**: Library for efficient similarity search.
- **Docker**: Containerization for consistent deployment.

---

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact:

- **Name**: Saman Madani
- **Email**: SamanMadani.dev@gmail.com
- **Website**: [My Personal Website](https://samanmadani.ir)
