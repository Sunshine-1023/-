import os
import re
import subprocess
from typing import Optional


def _parse_scutil_proxy() -> Optional[str]:
    try:
        out = subprocess.check_output(["scutil", "--proxy"], text=True, timeout=5)
    except Exception:
        return None

    def pick(key: str) -> Optional[str]:
        m = re.search(rf"{key}\s*:\s*(\S+)", out)
        return m.group(1) if m else None

    https_enable = pick("HTTPSEnable") == "1"
    http_enable = pick("HTTPEnable") == "1"
    host = pick("HTTPSProxy") if https_enable else pick("HTTPProxy")
    port = pick("HTTPSPort") if https_enable else pick("HTTPPort")

    if (https_enable or http_enable) and host and port and port != "0":
        return f"http://{host}:{port}"
    return None


def setup_proxy() -> Optional[str]:
    """配置 HTTP(S)_PROXY。优先使用 .env，其次读取 macOS 系统代理。"""
    existing = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if existing:
        return existing

    auto = os.getenv("AUTO_USE_SYSTEM_PROXY", "0").lower() in {"1", "true", "yes"}
    if not auto:
        return None

    proxy = _parse_scutil_proxy()
    if proxy:
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)

    # HF 镜像/主站尽量直连，避免被不稳定代理影响
    no_proxy = os.getenv(
        "NO_PROXY",
        "localhost,127.0.0.1,hf-mirror.com,huggingface.co,cdn-lfs.huggingface.co",
    )
    os.environ.setdefault("NO_PROXY", no_proxy)
    os.environ.setdefault("no_proxy", no_proxy)
    return proxy
