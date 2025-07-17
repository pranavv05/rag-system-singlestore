# 🧠 RAG System with SingleStore Vector DB

This project is a basic implementation of a **Retrieval-Augmented Generation (RAG)** system that:

- Retrieves content from **Wikipedia**
- Generates embeddings using **HuggingFace Transformers**
- Stores embeddings in **SingleStore** (vector database)
- Allows users to query the system via a LangChain-powered agent
- Uses **GPT-3.5-Turbo** as the language model

---

## 💻 Tech Stack

- **Python**
- **LangChain**
- **OpenAI GPT-3.5**
- **HuggingFace Embeddings**
- **SingleStoreDB** (for storing and retrieving vectors)
- **Wikipedia API**

---

## 🚀 Features

- Fetches and processes Wikipedia pages on selected topics
- Splits documents into manageable chunks
- Embeds and stores the chunks in SingleStore
- Allows natural language queries against stored content
- RAG agent powered by LangChain for interactive Q&A

---

## ⚙️ How it works

1. Fetches content for pre-defined topics from Wikipedia  
2. Splits the content using `RecursiveCharacterTextSplitter`  
3. Generates embeddings using `all-MiniLM-L6-v2` model  
4. Stores content + embeddings in SingleStore (as a vector DB)  
5. Answers user queries by retrieving relevant chunks and generating context-aware responses using GPT-3.5

---

## 📦 Setup Instructions

1. Clone the repository  
2. Install dependencies  
   pip install -r requirements.txt

3. Create a .env file with your credentials:
         OPENAI_API_KEY=your-openai-key
         SINGLESTORE_PASSWORD=your-password
 4. Create a vector DB in Singlestore 
 5. Run the script:
        python agentic_rag_system.py
