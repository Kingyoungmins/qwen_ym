# qwen_ym - Qwen 3.5 27B 내부망 서버

`run_server.py` 하나로 Qwen 3.5 27B 모델을 자동 로딩하고 HTTPS API 서버를 띄웁니다.

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
2. 모델 파일 없으면 HuggingFace에서 다운로드
3. TLS 자체서명 인증서 생성
4. vLLM 서버 기동 (Qwen 3.5 27B 로딩)
5. HTTPS API 서버 시작 (`0.0.0.0:8790`)

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_PATH` | `/home/adminym/qwen3.5/Qwen3.5-27B` | 모델 로컬 경로 |
| `MODEL_NAME` | `Qwen3.5-27B` | 서빙 모델 이름 |
| `MODEL_REPO` | `Qwen/Qwen3.5-27B` | HuggingFace repo (자동 다운로드용) |
| `VLLM_PORT` | `8015` | vLLM 내부 포트 |
| `API_PORT` | `8790` | API 서버 포트 |
| `API_KEY` | `63616e76` | Api-Key 인증값 |
| `TP_SIZE` | `2` | GPU 수 (tensor parallel) |
| `MAX_MODEL_LEN` | `200000` | 최대 컨텍스트 길이 |
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
    "model": "Qwen3.5-27B",
    "image_used": false,
    "latency_ms": 3200
  }
}
```

### `GET /health`

서버 상태 확인

## OpenClaw 클라이언트 설정

공식 OpenClaw에서 이 서버를 사용하려면:

### 1. 플러그인 복사

`~/.openclaw/extensions/new-vloet/` 폴더에 플러그인 파일 배치:

```
~/.openclaw/extensions/new-vloet/
├── index.ts
├── src/predict-tool.ts
├── package.json
└── openclaw.plugin.json
```

### 2. tool 허용

`~/.openclaw/openclaw.json`에 추가:

```json
{
  "tools": {
    "allow": ["new_vloet_predict"]
  }
}
```

### 3. (선택) baseUrl/apiKey 변경

기본값과 다른 주소를 쓸 경우 `openclaw.json`에:

```json
{
  "plugins": {
    "new-vloet": {
      "baseUrl": "https://your-internal-server.co.kr",
      "apiKey": "your-api-key"
    }
  }
}
```
