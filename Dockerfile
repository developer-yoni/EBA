FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 및 한글 폰트 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    fonts-nanum \
    fontconfig \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 복사 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

# 불필요한 파일 제외는 .dockerignore에서 처리

EXPOSE 5001

# gunicorn으로 프로덕션 실행 (타임아웃 300초, worker 1개 - 캐시 공유 문제 방지)
CMD ["python", "-m", "gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5001", "--timeout", "300", "app:app"]
