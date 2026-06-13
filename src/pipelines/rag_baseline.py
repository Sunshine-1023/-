import time
from src.config import get_llm
from src.retriever import DocumentRetriever
from langchain_core.prompts import PromptTemplate

class RAGPipeline:
    def __init__(self):
        self.llm = get_llm()
        self.retriever = DocumentRetriever()
        template = """请严格基于以下上下文回答问题。如果上下文中没有包含足够的信息，请直接回复“抱歉，提供的信息不足以回答该问题”。
        上下文：\n{context}\n\n问题：{question}"""
        self.prompt = PromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm

    def run(self, question: str) -> dict:
        start_time = time.time()
        docs = self.retriever.retrieve(question)
        context_text = "\n".join([doc.page_content for doc in docs])
        try:
            response = self.chain.invoke({"context": context_text, "question": question})
            answer = response.content
        except Exception as e:
            answer = f"【降级回答】模型调用失败：{type(e).__name__}"
        
        return {
            "answer": answer,
            "latency": time.time() - start_time,
            "context": [doc.page_content for doc in docs]
        }