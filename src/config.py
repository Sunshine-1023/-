import os
import hashlib
from typing import List
from dotenv import load_dotenv
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings as HuggingFaceEmbeddings

from src.proxy import setup_proxy

load_dotenv()
setup_proxy()

# 如果配置了 HF 镜像（如 https://hf-mirror.com），在下载模型前生效
_hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

class LocalHashEmbeddings(Embeddings):
    """离线可用的确定性向量，作为 BGE 不可用时的兜底方案。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vector = []
        for i in range(self.dim):
            digest = hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
            value = int.from_bytes(digest[:4], byteorder="big", signed=False)
            vector.append((value / 2**32) * 2 - 1)
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if (not api_key) or ("在这里填入" in api_key) or ("your" in api_key.lower()):
        raise ValueError(
            "DEEPSEEK_API_KEY 未正确配置，请在 .env 中填写真实密钥（例如 sk-xxxx）。"
        )
    timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
    # 忽略系统代理环境变量，避免 TUN/fake-ip 下 requests/httpx 走错通道
    trust_env = os.getenv("LLM_TRUST_ENV_PROXY", "0").lower() in {"1", "true", "yes"}
    http_client = httpx.Client(trust_env=trust_env, timeout=timeout_s)
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        max_tokens=1024,
        temperature=0.0,
        timeout=timeout_s,
        max_retries=max_retries,
        http_client=http_client,
    )

def get_embeddings():
    device = os.getenv("BGE_DEVICE", "mps")
    local_only = os.getenv("BGE_LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes"}
    try:
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={"device": device, "local_files_only": local_only},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        # 网络受限或模型下载失败时，保证流程仍可运行
        print("⚠️ BGE 模型加载失败，已切换到离线 LocalHashEmbeddings 备援。")
        return LocalHashEmbeddings(dim=256)