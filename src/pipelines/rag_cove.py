import time
from src.pipelines.rag_baseline import RAGPipeline
from langchain_core.prompts import PromptTemplate

class RAGCovePipeline(RAGPipeline):
    def __init__(self):
        super().__init__()
        self.plan_chain = PromptTemplate.from_template("基于以下草稿，提出2个简短的事实核查问题以验证其准确性。用换行符分隔。\n草稿：{draft}") | self.llm
        self.exec_chain = PromptTemplate.from_template("基于上下文：{context}\n简短回答核查问题：{question}") | self.llm
        self.final_chain = PromptTemplate.from_template("上下文：{context}\n原草稿：{draft}\n核查结果：{qa_results}\n请基于核查结果修正原草稿，输出最终准确的答案。") | self.llm

    def run(self, question: str) -> dict:
        start_time = time.time()
        docs = self.retriever.retrieve(question)
        context_text = "\n".join([doc.page_content for doc in docs])
        try:
            # Step 1: 初始草稿
            draft = self.chain.invoke({"context": context_text, "question": question}).content
            # Step 2: 规划核查问题
            check_questions = self.plan_chain.invoke({"draft": draft}).content.split('\n')
            # Step 3: 独立回答核查问题
            qa_results = ""
            for q in check_questions:
                if q.strip():
                    ans = self.exec_chain.invoke({"context": context_text, "question": q}).content
                    qa_results += f"Q: {q}\nA: {ans}\n"
            # Step 4: 最终生成
            final_answer = self.final_chain.invoke({
                "context": context_text,
                "draft": draft,
                "qa_results": qa_results
            }).content
        except Exception as e:
            final_answer = f"【降级回答】模型调用失败：{type(e).__name__}"
            
        return {
            "answer": final_answer,
            "latency": time.time() - start_time,
            "context": [doc.page_content for doc in docs]
        }