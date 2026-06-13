import os
import sys
import socket
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx
import requests
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.proxy import setup_proxy


def check_dns(host: str) -> Tuple[bool, str]:
    try:
        ip = socket.gethostbyname(host)
        return True, ip
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def is_fake_ip(ip: str) -> bool:
    """Clash/Surge TUN fake-ip 常见网段。"""
    return ip.startswith("198.18.") or ip.startswith("198.19.")


def check_http_get(url: str, timeout: int = 12, retries: int = 3) -> Tuple[bool, str]:
    last_err = ""
    for i in range(max(1, retries)):
        try:
            resp = requests.get(url, timeout=timeout, allow_redirects=True)
            # 401/403 也说明 TLS 已建立，网络可达
            if resp.status_code < 500:
                return True, f"HTTP {resp.status_code}"
            last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if i < retries - 1:
            time.sleep(1.5 * (i + 1))
    return False, last_err


def check_deepseek_chat(api_key: str, timeout: int = 15, retries: int = 3) -> Tuple[bool, str]:
    """用 httpx 直连检测（与项目 LLM 客户端行为一致，忽略系统代理）。"""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    last_err = ""
    for i in range(max(1, retries)):
        try:
            with httpx.Client(trust_env=False, timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                return True, "HTTP 200（LLM 可用）"
            if resp.status_code in (401, 403):
                return True, f"HTTP {resp.status_code}（TLS 已连通）"
            last_err = f"HTTP {resp.status_code}: {resp.text[:80]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if i < retries - 1:
            time.sleep(2 * (i + 1))
    return False, last_err


def print_proxy_info(active_proxy: Optional[str]) -> None:
    print("=== Proxy 环境变量 ===")
    if active_proxy:
        print(f"已启用代理: {active_proxy}")
    keys = [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "NO_PROXY", "no_proxy",
    ]
    for k in keys:
        v = os.getenv(k)
        if v:
            print(f"{k}={v}")
    if not active_proxy and not os.getenv("HTTP_PROXY"):
        print("未配置代理（若 DeepSeek 失败，可在 .env 设置 HTTP_PROXY/HTTPS_PROXY）。")
    print()


def print_fix_hints(deepseek_dns_ip: str, llm_ok: bool) -> None:
    if llm_ok:
        return
    print("=== 修复建议 ===")
    print()
    if is_fake_ip(deepseek_dns_ip):
        print("检测到 fake-ip（198.18.x），说明 Clash/Surge 等 TUN 模式在接管 DNS。")
        print("请在代理工具中：")
        print("  1. 将 api.deepseek.com 设为「代理」而非 DIRECT")
        print("  2. 换一个可稳定 TLS 的节点后重试")
        print("  3. 关闭 HTTPS 解密/嗅探对该域名的拦截")
        print("  4. 或临时关闭 TUN，改用系统代理模式测试")
    else:
        print("DeepSeek API TLS 握手失败，常见原因：")
        print("  - 节点不稳定或被 reset")
        print("  - 防火墙/校园网拦截")
        print("  - 代理规则未放行 api.deepseek.com:443")
    print()
    print("验证命令（终端执行）：")
    print("  curl -I https://api.deepseek.com/v1/models")
    print("看到 HTTP 401/200 即表示网络已恢复。")
    print()


def main() -> None:
    load_dotenv()
    active_proxy = setup_proxy()
    print("=== RAG 网络预检 ===")
    print_proxy_info(active_proxy)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    key_ok = bool(api_key) and ("在这里填入" not in api_key) and ("your" not in api_key.lower())
    print("=== Key 检查 ===")
    print(f"DEEPSEEK_API_KEY: {'OK' if key_ok else '未配置或占位值'}")
    print()

    hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").strip().rstrip("/")
    hf_model_probe = f"{hf_endpoint}/BAAI/bge-large-zh-v1.5/resolve/main/config.json"

    print("=== DNS 与 HTTP 检查 ===")

    ds_host = "api.deepseek.com"
    ds_dns_ok, ds_dns_msg = check_dns(ds_host)
    print("[DeepSeek API]")
    print(f"  DNS : {'OK' if ds_dns_ok else 'FAIL'} | {ds_dns_msg}")

    llm_ok = False
    llm_msg = "跳过（未配置 Key）"
    if key_ok:
        llm_ok, llm_msg = check_deepseek_chat(api_key, retries=4)
        print(f"  Chat: {'OK' if llm_ok else 'FAIL'} | {llm_msg}")
    else:
        models_ok, models_msg = check_http_get("https://api.deepseek.com/v1/models", retries=3)
        llm_ok = models_ok
        llm_msg = models_msg
        print(f"  HTTP: {'OK' if models_ok else 'FAIL'} | {models_msg}")

    http_ok: Dict[str, bool] = {"DeepSeek": llm_ok}

    for name, url in {
        f"HuggingFace 入口 ({hf_endpoint})": hf_endpoint,
        "HF 模型文件探测": hf_model_probe,
    }.items():
        host = url.split("/")[2]
        dns_ok, dns_msg = check_dns(host)
        web_ok, web_msg = check_http_get(url)
        http_ok[name] = web_ok
        print(f"[{name}]")
        print(f"  DNS : {'OK' if dns_ok else 'FAIL'} | {dns_msg}")
        print(f"  HTTP: {'OK' if web_ok else 'FAIL'} | {web_msg}")
    print()

    print("=== 降级模式预测 ===")
    bge_local_only = os.getenv("BGE_LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes"}
    hf_reachable = http_ok.get("HF 模型文件探测", False)

    print(f"HF_ENDPOINT={hf_endpoint}")
    print(f"BGE_LOCAL_FILES_ONLY={os.getenv('BGE_LOCAL_FILES_ONLY', '1')}")
    print(f"LLM 连通: {'OK' if llm_ok and key_ok else 'FAIL'}")
    print(f"HF 连通 : {'OK' if hf_reachable else 'FAIL'}")

    if not (llm_ok and key_ok):
        print("结论: 运行时会触发 LLM 降级回答。")
        print_fix_hints(ds_dns_msg if ds_dns_ok else "", llm_ok and key_ok)
    elif bge_local_only or not hf_reachable:
        print("结论: LLM 可用，但 embedding 可能走本地备援。")
    else:
        print("结论: 预期可使用真实 LLM + 真实 embedding。")


if __name__ == "__main__":
    main()
