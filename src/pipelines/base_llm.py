import time
from src.config import get_llm
from langchain_core.prompts import PromptTemplate

class BasePipeline:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = PromptTemplate.from_template("请直接回答以下问题：\n{question}")
        self.chain = self.prompt | self.llm

    def run(self, question: str) -> dict:
        start_time = time.time()
        try:
            response = self.chain.invoke({"question": question})
            answer = response.content
        except Exception as e:
            answer = f"【降级回答】模型调用失败：{type(e).__name__}"
        return {
            "answer": answer,
            "latency": time.time() - start_time,
            "context": []
        }