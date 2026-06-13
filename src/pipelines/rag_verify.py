import time
from src.pipelines.rag_baseline import RAGPipeline
from langchain_core.prompts import PromptTemplate

class RAGVerifyPipeline(RAGPipeline):
    def __init__(self):
        super().__init__()
        self.verify_prompt = PromptTemplate.from_template(
            "已知上下文：\n{context}\n目标答案：{answer}\n请严格判断目标答案是否完全由上下文支持。如果有任何捏造或上下文中未提及的信息，请仅输出'False'，否则输出'True'。"
        )
        self.verify_chain = self.verify_prompt | self.llm

    def run(self, question: str) -> dict:
        start_time = time.time()
        docs = self.retriever.retrieve(question)
        context_text = "\n".join([doc.page_content for doc in docs])
        try:
            # 1. 生成初稿
            draft = self.chain.invoke({"context": context_text, "question": question}).content
            # 2. 验证初稿
            verification = self.verify_chain.invoke({"context": context_text, "answer": draft}).content
            # 3. 熔断判定
            if "False" in verification:
                final_answer = "【拒答】抱歉，由于缺乏确凿证据，我拒绝回答该问题。"
            else:
                final_answer = draft
        except Exception as e:
            final_answer = f"【降级回答】模型调用失败：{type(e).__name__}"
            
        return {
            "answer": final_answer,
            "latency": time.time() - start_time,
            "context": [doc.page_content for doc in docs]
        }