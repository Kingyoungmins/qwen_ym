#!/usr/bin/env python3
"""
Hugging Face에서 Qwen3.5-27B-FP8 모델을 안정적으로 내려받는 스크립트.

- 저장 경로를 인자로 지정 가능
- huggingface_hub가 없으면 자동 설치
- 다운로드 중 실패해도 중단하지 않고 계속 재시도
- 기존에 받은 파일이 있으면 이어받기 동작 활용
"""

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


DEFAULT_REPO_ID = "Qwen/Qwen3.5-27B-FP8"
DEFAULT_RETRY_DELAY_SEC = 15
DEFAULT_MAX_RETRY_DELAY_SEC = 300


def _ensure_package(package_name: str, import_name: str | None = None) -> None:
    module_name = import_name or package_name
    try:
        __import__(module_name)
    except ImportError:
        print(f"[download] {package_name} 패키지 설치 중...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", package_name, "--break-system-packages"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-27B-FP8 모델을 Hugging Face에서 계속 재시도하며 다운로드합니다."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="모델을 저장할 로컬 디렉터리",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face 모델 repo id (기본값: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face 토큰. 미지정 시 HF_TOKEN 또는 HUGGINGFACE_HUB_TOKEN 사용",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="특정 브랜치/태그/커밋으로 고정할 때 사용",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="병렬 다운로드 워커 수 (기본값: 8)",
    )
    parser.add_argument(
        "--retry-delay-sec",
        type=int,
        default=DEFAULT_RETRY_DELAY_SEC,
        help=f"첫 재시도 대기 시간 초(기본값: {DEFAULT_RETRY_DELAY_SEC})",
    )
    parser.add_argument(
        "--max-retry-delay-sec",
        type=int,
        default=DEFAULT_MAX_RETRY_DELAY_SEC,
        help=f"최대 재시도 대기 시간 초(기본값: {DEFAULT_MAX_RETRY_DELAY_SEC})",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="필요한 파일 패턴만 내려받고 싶을 때 사용. 여러 번 지정 가능",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="제외할 파일 패턴. 여러 번 지정 가능",
    )
    return parser.parse_args()


def _prepare_env() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


def _download_forever(args: argparse.Namespace) -> Path:
    from huggingface_hub import snapshot_download

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    attempt = 0
    delay = max(1, args.retry_delay_sec)

    while True:
        attempt += 1
        print("=" * 72)
        print(f"[download] 시도 #{attempt}")
        print(f"[download] repo_id     : {args.repo_id}")
        print(f"[download] output_dir  : {output_dir}")
        print(f"[download] revision    : {args.revision or 'default'}")
        print(f"[download] max_workers : {args.max_workers}")
        print("=" * 72)

        try:
            local_path = snapshot_download(
                repo_id=args.repo_id,
                repo_type="model",
                local_dir=str(output_dir),
                token=args.token,
                revision=args.revision,
                max_workers=args.max_workers,
                allow_patterns=args.allow_pattern,
                ignore_patterns=args.ignore_pattern,
                local_files_only=False,
            )
            print(f"[download] 완료: {local_path}")
            return Path(local_path).resolve()
        except KeyboardInterrupt:
            print("\n[download] 사용자 중단으로 종료합니다.")
            raise
        except Exception as exc:
            print(f"[download] 실패: {exc.__class__.__name__}: {exc}")
            print("[download] 이어받기 가능한 상태로 계속 재시도합니다.")
            traceback.print_exc()
            print(f"[download] {delay}초 후 다시 시도합니다...")
            time.sleep(delay)
            delay = min(delay * 2, max(1, args.max_retry_delay_sec))


def main() -> int:
    args = _parse_args()
    _prepare_env()
    _ensure_package("huggingface_hub")

    final_path = _download_forever(args)
    print(f"[download] 모델 경로: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
