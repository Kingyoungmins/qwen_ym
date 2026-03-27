#!/usr/bin/env python3
"""
new_vloet - Qwen 3.5 27B FP8 내부망 서버
run_server.py 실행 시 vLLM + OpenAI 호환 API 자동 기동
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, request
from functools import wraps

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
    "tensor_parallel_size": int(os.environ.get("TP_SIZE", "0")),  # 0 = 자동 감지
    "max_model_len": int(os.environ.get("MAX_MODEL_LEN", "262144")),
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
    required = ["flask", "huggingface_hub"]
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
def _detect_gpu_count() -> int:
    """사용 가능한 GPU 수 자동 감지"""
    try:
        import torch
        count = torch.cuda.device_count()
    except Exception:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            count = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 1
        except Exception:
            count = 1
    return max(count, 1)


def _resolve_tp_size() -> int:
    """TP_SIZE가 0이면 자동 감지, 아니면 설정값 사용"""
    tp = CONFIG["tensor_parallel_size"]
    if tp > 0:
        return tp
    detected = _detect_gpu_count()
    print(f"[new_vloet] GPU 자동 감지: {detected}개 → tensor_parallel_size={detected}")
    return detected


def _build_vllm_cmd() -> str:
    tp_size = _resolve_tp_size()
    return (
        f"nohup {sys.executable} -m vllm.entrypoints.openai.api_server "
        f"--model {CONFIG['model_path']} "
        f"--served-model-name {CONFIG['model_name']} "
        f"--host {CONFIG['vllm_host']} "
        f"--port {CONFIG['vllm_port']} "
        f"--tensor-parallel-size {tp_size} "
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
# Flask 앱
# ──────────────────────────────────────────────
from flask import Flask, request as flask_request, jsonify

app = Flask(__name__)


# ──────────────────────────────────────────────
# Api-Key 인증 데코레이터
# ──────────────────────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = flask_request.headers.get("Api-Key", "")
        if api_key != CONFIG["api_key"]:
            return jsonify({"error": "Invalid Api-Key"}), 401
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────
@app.route("/v1/models", methods=["GET"])
def models_health():
    return "OK"


@app.route("/health", methods=["GET"])
def health():
    vllm_ok = _probe_vllm()
    return jsonify({
        "ok": vllm_ok,
        "model": CONFIG["model_name"],
        "model_path": CONFIG["model_path"],
        "vllm_url": VLLM_BASE_URL,
        "vllm_ok": vllm_ok,
    })


@app.route("/v1/models/t2i:predict", methods=["POST"])
@require_api_key
def predict():
    """메인 엔드포인트 - 질의 → 응답"""
    t0 = time.time()
    req = flask_request.get_json(force=True)

    q = req.get("q", "").strip()
    if not q:
        return jsonify({"error": "q is required"}), 400

    image_data_url = req.get("image_data_url", "").strip()
    max_tokens = req.get("max_tokens", 2048)
    temperature = req.get("temperature", 0.7)
    system_prompt = req.get("system_prompt", "").strip()

    messages = []

    # 시스템 프롬프트
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 사용자 메시지 (이미지 + 텍스트)
    content = []
    image_used = False
    if image_data_url:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        image_used = True
    content.append({"type": "text", "text": q})
    messages.append({"role": "user", "content": content})

    payload = {
        "model": CONFIG["model_name"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
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
        return jsonify({"error": f"vLLM HTTP {e.code}: {detail}"}), 500
    except Exception as e:
        return jsonify({"error": f"vLLM 요청 실패: {e}"}), 500

    choices = data.get("choices") or []
    answer = ""
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        answer = msg.get("content") or ""

    return jsonify({
        "query": q,
        "answer": answer,
        "usage": data.get("usage"),
        "meta": {
            "model": CONFIG["model_name"],
            "image_used": image_used,
            "latency_ms": int((time.time() - t0) * 1000),
        },
    })


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
    _start_vllm()

    print("=" * 60)
    print("  new_vloet - Qwen 3.5 27B FP8 Server (HTTPS)")
    print(f"  모델:    {CONFIG['model_name']}")
    print(f"  경로:    {CONFIG['model_path']}")
    print(f"  vLLM:    {CONFIG['vllm_host']}:{CONFIG['vllm_port']}")
    print(f"  API:     https://{CONFIG['api_host']}:{CONFIG['api_port']}")
    print(f"  TP:      {CONFIG['tensor_parallel_size']} GPU(s)")
    print(f"  Api-Key: {CONFIG['api_key']}")
    print("=" * 60)

    app.run(
        host=CONFIG["api_host"],
        port=CONFIG["api_port"],
        ssl_context=(str(CERT_FILE), str(KEY_FILE)),
    )
