# FastAPI Application with MongoDB, LangChain, and FAISS

This is a FastAPI-based application that processes and indexes user-uploaded PDF documents using MongoDB, LangChain, and FAISS. The app provides endpoints for uploading files, querying indexed data, and managing chat logs.

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
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
