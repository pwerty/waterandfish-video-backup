#!/bin/bash
set -e

VIDEO_PATH="$1"
if [ -z "$VIDEO_PATH" ]; then
  echo "Usage: $0 <video_file>"
  exit 1
fi

# 1) JSON 추출
python3 smooth_to_webm.py "$VIDEO_PATH"

# 2) 베이스 이름 추출 (확장자 제거)
BASENAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')

# 3) JSON 파일 이름 (경로 포함하여 수정)
# 'result/' 경로를 추가하여 정확한 파일 위치를 지정합니다.
JSON_FILE="result/${BASENAME}.json"

# 4) Puppeteer 녹화 스크립트 실행
node record.js "$JSON_FILE"
