import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

import sys

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retriever import DocumentRetriever
from src.pipelines import BasePipeline, RAGPipeline, RAGVerifyPipeline, RAGCovePipeline


REFUSAL_KEYWORDS = [
    "信息不足",
    "未提及",
    "无法回答",
    "不能回答",
    "拒绝回答",
    "抱歉",
]

METHOD_NAME_CN = {
    "Base_LLM": "基础大模型",
    "RAG": "RAG 基线",
    "RAG_Verify": "RAG + 验证熔断",
    "RAG_CoVe": "RAG + CoVe 自检",
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", "", text.lower())


def is_refusal(answer: str) -> bool:
    norm = normalize_text(answer)
    return any(k in norm for k in REFUSAL_KEYWORDS)


def contains_truth(answer: str, ground_truth: str) -> bool:
    return normalize_text(ground_truth) in normalize_text(answer)


def evaluate_answer(answer: str, ground_truth: str) -> Dict[str, int]:
    # “上下文中未提及”这类题，非拒答视为幻觉
    if ground_truth == "上下文中未提及":
        correct = 1 if is_refusal(answer) else 0
        hallucination = 0 if correct else 1
        return {"correct": correct, "hallucination": hallucination}

    correct = 1 if contains_truth(answer, ground_truth) else 0
    hallucination = 0 if correct else 1
    return {"correct": correct, "hallucination": hallucination}


def build_markdown_report(
    metric_df: pd.DataFrame, detail_df: pd.DataFrame, run_seconds: float
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# 实验报告（自动生成）")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 项目路径：`{project_root}`")
    lines.append(f"- 总耗时：{run_seconds:.2f} 秒")
    lines.append("")

    lines.append("## 指标说明（中文）")
    lines.append("")
    lines.append("- 正确率：答案命中标准答案的比例。")
    lines.append("- 幻觉率：回答包含错误/捏造信息的比例（越低越好）。")
    lines.append("- 平均延迟：单次问答平均耗时（秒，越低越好）。")
    lines.append("")

    lines.append("## 方法对比总表")
    lines.append("")
    lines.append("| 方法 | 中文名 | 样本数 | 正确数 | 幻觉数 | 正确率 | 幻觉率 | 平均延迟(秒) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in metric_df.iterrows():
        method = row["method"]
        lines.append(
            f"| {method} | {METHOD_NAME_CN.get(method, method)} | "
            f"{int(row['total_samples'])} | {int(row['correct_count'])} | {int(row['hallucination_count'])} | "
            f"{row['accuracy']:.2%} | {row['hallucination_rate']:.2%} | {row['avg_latency']:.2f} |"
        )
    lines.append("")

    lines.append("## 逐题判定明细（前 20 行）")
    lines.append("")
    lines.append("| 题号 | 方法 | 是否正确 | 是否幻觉 | 问题 | 标准答案 |")
    lines.append("|---:|---|---:|---:|---|---|")
    preview = detail_df.head(20)
    for _, row in preview.iterrows():
        question_text = str(row["question"]).replace("|", "\\|")
        truth_text = str(row["ground_truth"]).replace("|", "\\|")
        lines.append(
            f"| {int(row['row_id']) + 1} | {row['method']} | {int(row['correct'])} | {int(row['hallucination'])} | "
            f"{question_text} | {truth_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def get_ground_truth(item: dict) -> str:
    return item.get("ground_truth") or item.get("standard_answer", "")


def main() -> None:
    run_start = datetime.now()
    print("[1/5] 初始化 retriever + 入库", flush=True)
    retriever = DocumentRetriever()
    retriever.ingest_document(str(project_root / "data" / "raw_docs" / "knowledge.txt"))

    print("[2/5] 实例化 pipelines", flush=True)
    pipelines = {
        "Base_LLM": BasePipeline(),
        "RAG": RAGPipeline(),
        "RAG_Verify": RAGVerifyPipeline(),
        "RAG_CoVe": RAGCovePipeline(),
    }

    with open(project_root / "data" / "test_dataset.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("[3/5] 执行实验", flush=True)
    rows = []
    for i, item in enumerate(test_data, 1):
        q = item["question"]
        gt = get_ground_truth(item)
        row = {
            "id": item.get("id", f"Q{i}"),
            "category": item.get("category", ""),
            "question": q,
            "ground_truth": gt,
        }
        for name, pipe in pipelines.items():
            out = pipe.run(q)
            row[f"{name}_answer"] = out["answer"]
            row[f"{name}_latency"] = round(out["latency"], 4)
        rows.append(row)
        print(f"   完成第 {i}/{len(test_data)} 题", flush=True)

    raw_df = pd.DataFrame(rows)
    raw_csv = results_dir / "experiment_results.csv"
    raw_json = results_dir / "experiment_results.json"
    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    raw_df.to_json(raw_json, orient="records", force_ascii=False, indent=2)

    print("[4/5] 计算指标", flush=True)
    methods: List[str] = [c[:-7] for c in raw_df.columns if c.endswith("_answer")]
    detail_records = []
    metric_records = []
    for method in methods:
        answer_col = f"{method}_answer"
        latency_col = f"{method}_latency"
        correct_count = 0
        hallucination_count = 0
        for i, row in raw_df.iterrows():
            res = evaluate_answer(str(row[answer_col]), str(row["ground_truth"]))
            correct_count += res["correct"]
            hallucination_count += res["hallucination"]
            detail_records.append(
                {
                    "row_id": int(i),
                    "method": method,
                    "method_cn": METHOD_NAME_CN.get(method, method),
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "correct": res["correct"],
                    "hallucination": res["hallucination"],
                }
            )
        total = len(raw_df)
        metric_records.append(
            {
                "method": method,
                "method_cn": METHOD_NAME_CN.get(method, method),
                "total_samples": total,
                "correct_count": correct_count,
                "hallucination_count": hallucination_count,
                "accuracy": round(correct_count / total, 4),
                "hallucination_rate": round(hallucination_count / total, 4),
                "avg_latency": round(float(raw_df[latency_col].mean()), 4),
            }
        )

    metric_df = pd.DataFrame(metric_records).sort_values(
        by=["hallucination_rate", "accuracy", "avg_latency"],
        ascending=[True, False, True],
    )
    detail_df = pd.DataFrame(detail_records)

    metric_csv = results_dir / "experiment_metrics.csv"
    metric_json = results_dir / "experiment_metrics.json"
    detail_csv = results_dir / "experiment_eval_detail.csv"
    metric_df.to_csv(metric_csv, index=False, encoding="utf-8-sig")
    metric_df.to_json(metric_json, orient="records", force_ascii=False, indent=2)
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    print("[5/5] 生成中文报告", flush=True)
    run_seconds = (datetime.now() - run_start).total_seconds()
    report_text = build_markdown_report(metric_df, detail_df, run_seconds)
    report_md = results_dir / "experiment_report.md"
    report_md.write_text(report_text, encoding="utf-8")

    print("=== 输出文件 ===", flush=True)
    print(f"实验原始 CSV : {raw_csv}", flush=True)
    print(f"实验原始 JSON: {raw_json}", flush=True)
    print(f"指标汇总 CSV : {metric_csv}", flush=True)
    print(f"指标汇总 JSON: {metric_json}", flush=True)
    print(f"逐题明细 CSV : {detail_csv}", flush=True)
    print(f"中文报告 MD  : {report_md}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

