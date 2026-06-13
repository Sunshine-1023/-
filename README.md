# RAG 原型项目

这是一个基于检索增强生成（RAG）的实验性项目，包含向量化入库、检索与多种生成管道（基线、核验、CoVe）。适用于小规模知识库的原型验证与算法对比实验。

**快速概览**
- Embeddings: `BAAI/bge-large-zh-v1.5`（优先通过 `langchain-huggingface`，失败时降级到离线方案）
- 向量库: `Chroma`（优先通过 `langchain-chroma`，本地持久化目录 `./chroma_db`）
- LLM: DeepSeek（通过 `DEEPSEEK_API_KEY` 访问）
- 主要模块: `src/retriever.py`, `src/pipelines/*`

---

## 准备与安装

1. 克隆仓库并进入目录。
2. 创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

3. 安装依赖（推荐使用 `requirements-frozen.txt`）：

```bash
pip install -r requirements-frozen.txt
```

4. 配置环境变量：复制 `.env.example` 为 `.env`，并填写必要项：

```bash
cp .env.example .env
```

`.env` 中至少需要：

```env
DEEPSEEK_API_KEY=你的真实密钥
BGE_LOCAL_FILES_ONLY=0
HF_ENDPOINT=https://hf-mirror.com
```

---

## 唯一官方入口（跑实验 + 出报告）

请使用以下命令完成**全部实验流程**（入库 → 四方法推理 → 指标计算 → 中文报告生成）：

```bash
source .venv/bin/activate
python scripts/run_and_report.py
```

该脚本会自动完成：
1. 知识库向量化入库
2. 运行 4 个方法（`Base_LLM` / `RAG` / `RAG_Verify` / `RAG_CoVe`）
3. 计算正确率、幻觉率、平均延迟
4. 生成可读的中文 Markdown 报告

### 输出文件（均在 `results/` 目录）

| 文件 | 说明 |
|---|---|
| `experiment_results.csv` / `.json` | 原始问答结果 |
| `experiment_metrics.csv` / `.json` | 各方法指标汇总 |
| `experiment_eval_detail.csv` | 逐题判定明细 |
| **`experiment_report.md`** | **推荐查看：中文实验报告** |

---

## 运行前预检（可选，非实验入口）

在首次运行或网络异常时，可先执行连通性检查：

```bash
python scripts/network_check.py
```

该脚本仅用于诊断 DeepSeek / HuggingFace 是否可达，**不替代** `run_and_report.py`。

---

## 注意与建议

- 请锁定并验证依赖版本（已提供 `requirements-frozen.txt`）。
- 若在 macOS 上遇到 `grpcio ... not supported on this platform`，请使用项目中的依赖约束（`grpcio>=1.75,<1.81`）。
- 若 DeepSeek 或 HuggingFace 连接受限，程序会降级运行；请先通过 `network_check.py` 排查网络与代理配置。
- `notebooks/evaluation.ipynb` 仅作交互式调试参考，**不作为官方运行入口**。
# -
