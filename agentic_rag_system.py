

import os
import warnings
import numpy as np
import singlestoredb as s2
import wikipediaapi
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI  # ✅ Keep ChatOpenAI (you still need the LLM)
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ Use HuggingFace for embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import BaseRetriever

os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')
SINGLESTORE_HOST = 'svc-17465ce5-7a0e-402a-a101-afdea9f2ebfa-dml.aws-virginia-8.svc.singlestore.com'
SINGLESTORE_PORT = 3306
SINGLESTORE_USER = 'admin'
SINGLESTORE_PASSWORD = 'ufq((8mk-8nA;e{brznxPkm?M)'  # Replace if different
SINGLESTORE_DATABASE = os.getenv('SINGLESTORE_PASSWORD')

warnings.filterwarnings('ignore')

def create_singlestore_connection():
    try:
        connection = s2.connect(
            host=SINGLESTORE_HOST,
            port=SINGLESTORE_PORT,
            user=SINGLESTORE_USER,
            password=SINGLESTORE_PASSWORD,
            database=SINGLESTORE_DATABASE
        )
        print(" Successfully connected to SingleStore!")
        return connection
    except Exception as e:
        print(f" Error connecting to SingleStore: {e}")
        return None

conn = create_singlestore_connection()

def create_vector_table(connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS wikipedia_vectors (
        id INT AUTO_INCREMENT PRIMARY KEY,
        content TEXT,
        title VARCHAR(500),
        url VARCHAR(1000),
        chunk_index INT,
        embedding BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX(title), INDEX(chunk_index)
    );
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(create_table_query)
            connection.commit()
        print(" Vector table created successfully!")
    except Exception as e:
        print(f" Error creating table: {e}")

if conn:
    create_vector_table(conn)
class WikipediaFetcher:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='LangChain-Tutorial/1.0'
        )

    def fetch_page_content(self, page_title: str) -> dict:
        page = self.wiki.page(page_title)
        if not page.exists():
            return None
        return {
            'title': page.title,
            'content': page.text,
            'url': page.fullurl,
            'summary': page.summary
        }

    def fetch_multiple_pages(self, page_titles: List[str]) -> List[dict]:
        return [p for title in page_titles if (p := self.fetch_page_content(title))]

wiki_fetcher = WikipediaFetcher()
topics = ["Retrieval-augmented generation", "Machine Learning", "NLP", "Computer Vision", "Database"]
wikipedia_pages = wiki_fetcher.fetch_multiple_pages(topics)

class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def process_documents(self, pages: List[dict]) -> List[Document]:
        docs = []
        for page in pages:
            doc = Document(
                page_content=page['content'],
                metadata={'title': page['title'], 'url': page['url'], 'summary': page['summary']}
            )
            chunks = self.splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = i
                docs.append(chunk)
        return docs

    def create_embeddings(self, docs: List[Document]) -> List[List[float]]:
        return self.embeddings.embed_documents([doc.page_content for doc in docs])

processor = DocumentProcessor()
documents = processor.process_documents(wikipedia_pages)
embeddings = processor.create_embeddings(documents)

def store_vectors_in_singlestore(connection, documents, embeddings):
    insert_query = """
    INSERT INTO wikipedia_vectors (content, title, url, chunk_index, embedding)
    VALUES (%s, %s, %s, %s, %s)
    """
    try:
        with connection.cursor() as cursor:
            for doc, embed in zip(documents, embeddings):
                embed_bytes = np.array(embed, dtype=np.float32).tobytes()
                cursor.execute(insert_query, (
                    doc.page_content,
                    doc.metadata['title'],
                    doc.metadata['url'],
                    doc.metadata['chunk_index'],
                    embed_bytes
                ))
            connection.commit()
        print(f" Stored {len(documents)} vectors.")
    except Exception as e:
        print(f" Error storing vectors: {e}")

store_vectors_in_singlestore(conn, documents, embeddings)

class SingleStoreRetriever:
    def __init__(self, connection, embeddings_model):
        self.connection = connection
        self.embeddings = embeddings_model

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        query_embed = self.embeddings.embed_query(query)
        query_bytes = np.array(query_embed, dtype=np.float32).tobytes()
        sql = """
        SELECT content, title, url, chunk_index, embedding,
               DOT_PRODUCT(embedding, %s) /
               (SQRT(DOT_PRODUCT(embedding, embedding)) * SQRT(DOT_PRODUCT(%s, %s))) as similarity
        FROM wikipedia_vectors
        ORDER BY similarity DESC
        LIMIT %s
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, (query_bytes, query_bytes, query_bytes, k))
                results = cursor.fetchall()
                return [Document(page_content=row[0], metadata={'title': row[1], 'url': row[2], 'chunk_index': row[3], 'similarity': float(row[5])}) for row in results]
        except Exception as e:
            print(f"❌ Similarity search error: {e}")
            return []

retriever = SingleStoreRetriever(conn, processor.embeddings)

class WikipediaRAGAgent:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def search_knowledge_base(self, query: str) -> str:
        docs = self.retriever.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found."
        return "\n\n".join([f"From '{doc.metadata['title']}':\n{doc.page_content}" for doc in docs])

    def answer_with_context(self, query: str) -> str:
        context = self.search_knowledge_base(query)
        if "No relevant" in context:
            return context
        prompt = f"""
        Based on the context below, answer the question:

        {context}

        Question: {query}
        Answer:
        """
        try:
            res = self.llm.invoke(prompt)
            return res.content if hasattr(res, 'content') else str(res)
        except Exception as e:
            return f"Error: {e}"

    def create_agent(self):
        tools = [
            Tool(name="Search_Wikipedia", func=self.search_knowledge_base, description="Search KB"),
            Tool(name="Answer_With_Context", func=self.answer_with_context, description="Answer from KB")
        ]
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
rag_agent = WikipediaRAGAgent(retriever, llm)
agent = rag_agent.create_agent()

def ask_direct(question: str):
    return rag_agent.answer_with_context(question)

if __name__ == '__main__':
    questions = [
        "What is retrieval augmented generation?",
        "Difference between machine learning and RAG?",
        "Applications of a database?"
    ]
    for q in questions:
        print(f"\n❓ {q}\n{ask_direct(q)}\n{'-'*40}")
