import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_embeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

class DocumentRetriever:
    def __init__(self):
        self.embeddings = get_embeddings()
        if os.path.exists(DB_DIR):
            self.vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        else:
            self.vectorstore = None

    def ingest_document(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents)
        
        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = Chroma.from_documents(chunks, self.embeddings, persist_directory=DB_DIR)
        print("向量化入库完成！")

    def retrieve(self, query: str, top_k: int = 2) -> list:
        if not self.vectorstore:
            raise ValueError(
                "请先调用 ingest_document 入库文档。"
                "如果你在 notebook 中运行，请先执行入库单元。"
            )
        return self.vectorstore.similarity_search(query, k=top_k)