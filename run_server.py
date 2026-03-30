#!/usr/bin/env python3
"""
new_vloet - Qwen 3.5 27B FP8 내부망 서버
run_server.py 실행 → start_vllm.sh로 vLLM 기동 → Flask API 프록시
"""

import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from urllib import error, request
from functools import wraps

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
CONFIG = {
    "vllm_host": os.environ.get("VLLM_HOST", "127.0.0.1"),
    "vllm_port": int(os.environ.get("VLLM_PORT", "8015")),
    "api_host": os.environ.get("API_HOST", "0.0.0.0"),
    "api_port": int(os.environ.get("API_PORT", "8080")),
    "api_key": os.environ.get("API_KEY", "63616e76"),
    "vllm_api_key": os.environ.get("VLLM_API_KEY", "local-vllm"),
    "timeout_sec": int(os.environ.get("TIMEOUT_SEC", "180")),
    "model_name": os.environ.get("MODEL_NAME", "Qwen3.5-27B-FP8"),
}

VLLM_BASE_URL = f"http://{CONFIG['vllm_host']}:{CONFIG['vllm_port']}/v1"
SCRIPT_DIR = Path(__file__).parent
VLLM_SCRIPT = SCRIPT_DIR / "start_vllm.sh"


# ──────────────────────────────────────────────
# 자동 pip 설치
# ──────────────────────────────────────────────
def _auto_install():
    """Flask + vLLM 등 필요한 패키지 자동 설치"""
    # requirements.txt가 있으면 그걸로 설치
    req_file = SCRIPT_DIR / "requirements.txt"
    if req_file.exists():
        print(f"[new_vloet] requirements.txt로 패키지 설치...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--break-system-packages", "-q"],
            stdout=sys.stdout, stderr=sys.stderr,
        )
        return

    # requirements.txt 없으면 최소 패키지만 설치
    required = {"flask": "flask", "vllm": "vllm", "huggingface_hub": "huggingface-hub"}
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[new_vloet] 누락 패키지 설치: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U"] + missing + ["--break-system-packages"],
            stdout=sys.stdout, stderr=sys.stderr,
        )


# ──────────────────────────────────────────────
# HTTP 유틸
# ──────────────────────────────────────────────
def _post_json(url, payload, timeout_sec):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {CONFIG['vllm_api_key']}",
    }, method="POST")
    with request.urlopen(req, timeout=timeout_sec) as res:
        return json.loads(res.read().decode("utf-8"))


def _probe_vllm():
    try:
        req = request.Request(f"{VLLM_BASE_URL}/models", headers={"Accept": "application/json"})
        with request.urlopen(req, timeout=3) as res:
            return res.status == 200
    except error.HTTPError as e:
        # 401 Unauthorized = vLLM은 살아있지만 API_KEY 필요
        return e.code == 401
    except Exception:
        return False


# ──────────────────────────────────────────────
# vLLM 시작 (.sh 실행)
# ──────────────────────────────────────────────
def _start_vllm_sh():
    """start_vllm.sh를 백그라운드 스레드에서 실행"""
    print(f"[new_vloet] start_vllm.sh 실행 중...")
    print(f"  스크립트: {VLLM_SCRIPT}")
    print(f"  로그:     /tmp/new_vloet_vllm.log")

    # start_vllm.sh에 전달할 환경변수 (기존 동작 방식과 동일)
    vllm_env = {
        **os.environ,
        "HOST": os.environ.get("VLLM_HOST", "0.0.0.0"),
        "PORT": str(CONFIG["vllm_port"]),
        "VLLM_API_KEY": CONFIG["vllm_api_key"],
    }

    proc = subprocess.Popen(
        ["bash", str(VLLM_SCRIPT)],
        env=vllm_env,
    )
    proc.wait()
    print(f"[new_vloet] ⚠ vLLM 종료 (exit={proc.returncode})")


def _wait_for_vllm(timeout=600):
    print("[new_vloet] vLLM ready 대기 중...")
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        if _probe_vllm():
            print(f"\n[new_vloet] ✓ vLLM 준비 완료! ({VLLM_BASE_URL})")
            return True
        dots += 1
        print(".", end="", flush=True)
        if dots % 60 == 0:
            print(f" ({dots * 2}s)")
        time.sleep(2)
    print(f"\n[new_vloet] ⚠ vLLM 타임아웃 ({timeout}s). 로그: /tmp/new_vloet_vllm.log")
    return False


# ──────────────────────────────────────────────
# TLS 인증서
# ──────────────────────────────────────────────
CERT_DIR = SCRIPT_DIR / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"


def _ensure_tls_cert():
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    print("[new_vloet] TLS 인증서 생성...")
    subprocess.check_call([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
        "-days", "365", "-nodes", "-subj", "/CN=new-vloet-internal",
    ])


# ──────────────────────────────────────────────
# Flask 앱
# ──────────────────────────────────────────────
def _create_app():
    from flask import Flask, request as flask_request, jsonify

    app = Flask(__name__)

    def require_api_key(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            key = flask_request.headers.get("Api-Key", "")
            if key != CONFIG["api_key"]:
                return jsonify({"error": "Invalid Api-Key"}), 401
            return f(*args, **kwargs)
        return decorated

    @app.route("/v1/models", methods=["GET"])
    def models_health():
        return "OK"

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": _probe_vllm(), "model": CONFIG["model_name"], "vllm_url": VLLM_BASE_URL})

    @app.route("/v1/models/t2i:predict", methods=["POST"])
    @require_api_key
    def predict():
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
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

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
            data = _post_json(f"{VLLM_BASE_URL}/chat/completions", payload, CONFIG["timeout_sec"])
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            return jsonify({"error": f"vLLM HTTP {e.code}: {detail}"}), 500
        except Exception as e:
            return jsonify({"error": f"vLLM 요청 실패: {e}"}), 500

        choices = data.get("choices") or []
        answer = ""
        reasoning = ""
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            answer = msg.get("content") or ""
            reasoning = msg.get("reasoning") or ""
            # reasoning 모델: content가 비어있으면 reasoning을 answer로
            if not answer and reasoning:
                answer = reasoning

        return jsonify({
            "query": q,
            "answer": answer,
            "reasoning": reasoning,
            "usage": data.get("usage"),
            "meta": {
                "model": CONFIG["model_name"],
                "image_used": image_used,
                "latency_ms": int((time.time() - t0) * 1000),
            },
        })

    return app


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    _auto_install()
    _ensure_tls_cert()

    # vLLM이 이미 떠있으면 스킵, 아니면 .sh로 시작
    if _probe_vllm():
        print(f"[new_vloet] vLLM 이미 실행 중: {VLLM_BASE_URL}")
    else:
        if not VLLM_SCRIPT.exists():
            print(f"[new_vloet] ⚠ {VLLM_SCRIPT} 파일이 없습니다!")
            sys.exit(1)

        vllm_thread = threading.Thread(target=_start_vllm_sh, daemon=True)
        vllm_thread.start()

        if not _wait_for_vllm(timeout=600):
            print("[new_vloet] vLLM 시작 실패. 로그 확인: /tmp/new_vloet_vllm.log")
            sys.exit(1)

    app = _create_app()

    print("=" * 60)
    print("  new_vloet - Qwen 3.5 27B FP8")
    print(f"  vLLM:      {VLLM_BASE_URL}")
    print(f"  API:       https://{CONFIG['api_host']}:{CONFIG['api_port']}")
    print(f"  엔드포인트: POST /v1/models/t2i:predict")
    print(f"  Api-Key:   {CONFIG['api_key']}")
    print("=" * 60)

    app.run(
        host=CONFIG["api_host"],
        port=CONFIG["api_port"],
        ssl_context=(str(CERT_FILE), str(KEY_FILE)),
    )
