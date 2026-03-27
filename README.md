# qwen_ym - Qwen 3.5 27B FP8 내부망 서버

`run_server.py` 하나로 Qwen 3.5 27B FP8 모델을 자동 로딩하고 HTTPS API 서버를 띄웁니다.

## 구조

```
서버 (내부망 GPU 머신)              클라이언트 (OpenClaw)
┌─────────────────────┐            ┌──────────────────┐
│  run_server.py      │            │  OpenClaw (공식)  │
│    ├─ vLLM 자동기동  │  ◀──────  │    + 플러그인      │
│    ├─ 모델 자동다운로드│  HTTPS    │                  │
│    └─ API 대기       │            │                  │
└─────────────────────┘            └──────────────────┘
```

## 서버 실행

```bash
python run_server.py
```

실행 시 자동으로:
1. 누락 패키지 설치 (fastapi, uvicorn, huggingface-hub)
2. 모델 파일 없으면 HuggingFace에서 다운로드 (`Qwen/Qwen3.5-27B-FP8`)
3. TLS 자체서명 인증서 생성
4. vLLM 서버 기동 (reasoning parser, tool call parser 포함)
5. HTTPS API 서버 시작 (`0.0.0.0:8790`)

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_PATH` | `/data/public/qwen3.5/Qwen3.5-27B-FP8` | 모델 로컬 경로 |
| `MODEL_NAME` | `Qwen3.5-27B-FP8` | 서빙 모델 이름 |
| `MODEL_REPO` | `Qwen/Qwen3.5-27B-FP8` | HuggingFace repo (자동 다운로드용) |
| `VLLM_PORT` | `8015` | vLLM 내부 포트 |
| `API_PORT` | `8790` | API 서버 포트 |
| `API_KEY` | `63616e76` | Api-Key 인증값 |
| `TP_SIZE` | `1` | GPU 수 (tensor parallel) |
| `MAX_MODEL_LEN` | `262144` | 최대 컨텍스트 길이 (256K) |
| `GPU_MEM_UTIL` | `0.90` | GPU 메모리 사용률 |
| `HF_TOKEN` | (없음) | HuggingFace 토큰 (비공개 모델 시 필요) |

## API

### `POST /v1/models/t2i:predict`

```bash
curl -k -X POST https://localhost:8790/v1/models/t2i:predict \
  -H 'Api-Key: 63616e76' \
  -H 'Content-Type: application/json' \
  -d '{"q": "안녕하세요"}'
```

**요청:**
```json
{
  "q": "질의 내용",
  "image_data_url": "data:image/png;base64,...",
  "system_prompt": "시스템 프롬프트",
  "max_tokens": 2048,
  "temperature": 0.7
}
```

**응답:**
```json
{
  "query": "질의 내용",
  "answer": "모델 응답",
  "usage": { "prompt_tokens": 128, "completion_tokens": 256 },
  "meta": {
    "model": "Qwen3.5-27B-FP8",
    "image_used": false,
    "latency_ms": 3200
  }
}
```

### `GET /health`

서버 상태 확인

## vLLM 옵션

start 스크립트와 동일한 옵션 적용:
- `--reasoning-parser qwen3` (추론 파서)
- `--enforce-eager` (eager 모드)
- `--enable-auto-tool-choice` (자동 tool 선택)
- `--tool-call-parser qwen3_coder` (tool call 파서)
