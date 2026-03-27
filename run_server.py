#!/usr/bin/env python3
"""
new_vloet - Qwen 3.5 27B 내부망 서버
server.py 실행 시 vLLM + OpenAI 호환 API 자동 기동
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# 설정 (내부망 환경에 맞게 수정)
# ──────────────────────────────────────────────
CONFIG = {
    # 모델 설정
    "model_path": os.environ.get("MODEL_PATH", "/data/public/qwen3.5/Qwen3.5-27B-FP8"),
    "model_name": os.environ.get("MODEL_NAME", "Qwen3.5-27B-FP8"),
    "model_repo": os.environ.get("MODEL_REPO", "Qwen/Qwen3.5-27B-FP8"),  # HuggingFace repo ID

    # vLLM 서버 설정
    "vllm_host": os.environ.get("VLLM_HOST", "0.0.0.0"),
    "vllm_port": int(os.environ.get("VLLM_PORT", "8015")),
    "tensor_parallel_size": int(os.environ.get("TP_SIZE", "1")),
    "max_model_len": int(os.environ.get("MAX_MODEL_LEN", "32768")),
    "gpu_memory_utilization": float(os.environ.get("GPU_MEM_UTIL", "0.90")),

    # API 서버 설정
    "api_host": os.environ.get("API_HOST", "0.0.0.0"),
    "api_port": int(os.environ.get("API_PORT", "8790")),
    "api_key": os.environ.get("API_KEY", "63616e76"),  # Api-Key 헤더 인증값
    "timeout_sec": int(os.environ.get("TIMEOUT_SEC", "180")),
}

# ──────────────────────────────────────────────
# 내부 vLLM 연결 URL (자동 생성)
# ──────────────────────────────────────────────
VLLM_BASE_URL = f"http://127.0.0.1:{CONFIG['vllm_port']}/v1"


# ──────────────────────────────────────────────
# 자동 pip 설치
# ──────────────────────────────────────────────
def _auto_install():
    """첫 실행 시 필요한 패키지 자동 설치"""
    required = ["fastapi", "uvicorn", "pydantic", "huggingface_hub"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))
    if missing:
        print(f"[new_vloet] 누락 패키지 설치: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U"] + missing + ["--break-system-packages"],
            stdout=sys.stdout, stderr=sys.stderr,
        )


# ──────────────────────────────────────────────
# HTTP 유틸
# ──────────────────────────────────────────────
def _post_json(url: str, payload: dict, timeout_sec: int) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {CONFIG['api_key']}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_sec) as res:
        return json.loads(res.read().decode("utf-8"))


def _get_json(url: str, timeout_sec: int = 3) -> dict:
    req = request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with request.urlopen(req, timeout=timeout_sec) as res:
        return json.loads(res.read().decode("utf-8"))


def _probe_vllm() -> bool:
    try:
        _get_json(f"{VLLM_BASE_URL}/models", timeout_sec=2)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────
# vLLM 자동 기동
# ──────────────────────────────────────────────
def _build_vllm_cmd() -> str:
    return (
        f"nohup {sys.executable} -m vllm.entrypoints.openai.api_server "
        f"--model {CONFIG['model_path']} "
        f"--served-model-name {CONFIG['model_name']} "
        f"--host {CONFIG['vllm_host']} "
        f"--port {CONFIG['vllm_port']} "
        f"--tensor-parallel-size {CONFIG['tensor_parallel_size']} "
        f"--max-model-len {CONFIG['max_model_len']} "
        f"--gpu-memory-utilization {CONFIG['gpu_memory_utilization']} "
        f"--trust-remote-code "
        f"--dtype auto "
        f"--reasoning-parser qwen3 "
        f"--enforce-eager "
        f"--enable-auto-tool-choice "
        f"--tool-call-parser qwen3_coder "
        f">/tmp/new_vloet_vllm.log 2>&1 &"
    )


def _download_model() -> None:
    """모델 파일이 없으면 HuggingFace에서 자동 다운로드"""
    model_path = Path(CONFIG["model_path"])
    repo_id = CONFIG["model_repo"]

    if model_path.exists() and any(model_path.iterdir()):
        print(f"[new_vloet] 모델 이미 존재: {model_path}")
        return

    print(f"[new_vloet] 모델 파일이 없습니다. HuggingFace에서 다운로드합니다.")
    print(f"  repo:  {repo_id}")
    print(f"  경로:  {model_path}")

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"  HF_TOKEN: 설정됨")
    else:
        print(f"  HF_TOKEN: 미설정 (공개 모델이면 불필요)")

    try:
        from huggingface_hub import snapshot_download

        model_path.parent.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            token=hf_token or None,
            resume_download=True,
        )
        print(f"[new_vloet] 모델 다운로드 완료: {model_path}")

    except Exception as e:
        raise RuntimeError(
            f"[new_vloet] 모델 다운로드 실패: {e}\n"
            f"  → HF_TOKEN 환경변수를 확인하거나, 수동으로 모델을 배치하세요.\n"
            f"  → export HF_TOKEN=hf_xxxxx"
        ) from e


def _start_vllm() -> None:
    """vLLM 서버가 없으면 자동 시작하고 ready 될 때까지 대기"""
    if _probe_vllm():
        print(f"[new_vloet] vLLM 이미 실행 중: {VLLM_BASE_URL}")
        return

    # 모델 파일 없으면 자동 다운로드
    _download_model()

    model_path = Path(CONFIG["model_path"])
    if not model_path.exists():
        raise RuntimeError(f"[new_vloet] 모델 경로가 존재하지 않습니다: {model_path}")

    cmd = _build_vllm_cmd()
    print(f"[new_vloet] vLLM 서버 시작 중...")
    print(f"  모델: {CONFIG['model_name']}")
    print(f"  경로: {CONFIG['model_path']}")
    print(f"  포트: {CONFIG['vllm_port']}")
    print(f"  TP:   {CONFIG['tensor_parallel_size']}")
    print(f"  로그: /tmp/new_vloet_vllm.log")

    subprocess.Popen(cmd, shell=True)

    # 모델 로딩 대기 (27B는 시간 소요)
    deadline = time.time() + 600  # 10분 타임아웃
    dots = 0
    while time.time() < deadline:
        if _probe_vllm():
            print(f"\n[new_vloet] vLLM 준비 완료!")
            return
        dots += 1
        print(".", end="", flush=True)
        if dots % 60 == 0:
            print(f" ({dots * 2}s)")
        time.sleep(2)

    raise RuntimeError(
        "[new_vloet] vLLM 시작 시간 초과 (10분). 로그 확인: /tmp/new_vloet_vllm.log"
    )


# ──────────────────────────────────────────────
# FastAPI 앱
# ──────────────────────────────────────────────
app = FastAPI(title="new_vloet - Qwen 3.5 27B", version="1.0.0")

# ──────────────────────────────────────────────
# Api-Key 인증
# ──────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="Api-Key")


async def _verify_api_key(api_key: str = Depends(_api_key_header)) -> str:
    if api_key != CONFIG["api_key"]:
        raise HTTPException(status_code=401, detail="Invalid Api-Key")
    return api_key


class ChatRequest(BaseModel):
    q: str = Field(..., min_length=1, description="사용자 질의")
    image_data_url: str = Field(default="", description="data:image/... base64 (선택)")
    max_tokens: int = Field(default=2048, ge=1, le=16384)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: str = Field(default="", description="시스템 프롬프트 (선택)")


@app.on_event("startup")
async def startup_event() -> None:
    _start_vllm()


@app.get("/health")
async def health() -> dict[str, Any]:
    vllm_ok = _probe_vllm()
    return {
        "ok": vllm_ok,
        "model": CONFIG["model_name"],
        "model_path": CONFIG["model_path"],
        "vllm_url": VLLM_BASE_URL,
        "vllm_ok": vllm_ok,
    }


@app.post("/v1/models/t2i:predict", dependencies=[Depends(_verify_api_key)])
async def predict(req: ChatRequest) -> dict[str, Any]:
    """메인 엔드포인트 - 질의 → 응답"""
    t0 = time.time()
    messages: list[dict[str, Any]] = []

    # 시스템 프롬프트
    if req.system_prompt.strip():
        messages.append({"role": "system", "content": req.system_prompt.strip()})

    # 사용자 메시지 (이미지 + 텍스트)
    content: list[dict[str, Any]] = []
    image_used = False
    if req.image_data_url.strip():
        content.append({"type": "image_url", "image_url": {"url": req.image_data_url.strip()}})
        image_used = True
    content.append({"type": "text", "text": req.q})
    messages.append({"role": "user", "content": content})

    payload = {
        "model": CONFIG["model_name"],
        "messages": messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": 0.95,
    }

    try:
        data = _post_json(
            f"{VLLM_BASE_URL}/chat/completions",
            payload,
            CONFIG["timeout_sec"],
        )
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=500, detail=f"vLLM HTTP {e.code}: {detail}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM 요청 실패: {e}") from e

    choices = data.get("choices") or []
    answer = ""
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        answer = msg.get("content") or ""

    return {
        "query": req.q,
        "answer": answer,
        "usage": data.get("usage"),
        "meta": {
            "model": CONFIG["model_name"],
            "image_used": image_used,
            "latency_ms": int((time.time() - t0) * 1000),
        },
    }


# ──────────────────────────────────────────────
# TLS 자체서명 인증서 자동 생성
# ──────────────────────────────────────────────
CERT_DIR = Path(__file__).parent / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"


def _ensure_tls_cert() -> None:
    """자체서명 인증서가 없으면 자동 생성 (내부망용)"""
    if CERT_FILE.exists() and KEY_FILE.exists():
        return

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    print("[new_vloet] TLS 자체서명 인증서 생성 중...")
    subprocess.check_call([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
        "-days", "365", "-nodes",
        "-subj", "/CN=new-vloet-internal",
    ])
    print(f"[new_vloet] 인증서 생성 완료: {CERT_DIR}")


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    _auto_install()
    _ensure_tls_cert()
    import uvicorn

    print("=" * 60)
    print("  new_vloet - Qwen 3.5 27B Server (HTTPS)")
    print(f"  모델:    {CONFIG['model_name']}")
    print(f"  경로:    {CONFIG['model_path']}")
    print(f"  vLLM:    {CONFIG['vllm_host']}:{CONFIG['vllm_port']}")
    print(f"  API:     https://{CONFIG['api_host']}:{CONFIG['api_port']}")
    print(f"  TP:      {CONFIG['tensor_parallel_size']} GPU(s)")
    print(f"  Api-Key: {CONFIG['api_key']}")
    print("=" * 60)

    uvicorn.run(
        app,
        host=CONFIG["api_host"],
        port=CONFIG["api_port"],
        ssl_certfile=str(CERT_FILE),
        ssl_keyfile=str(KEY_FILE),
        reload=False,
    )
