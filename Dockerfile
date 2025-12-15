FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 복사 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

# 불필요한 파일 제외는 .dockerignore에서 처리

EXPOSE 5000

# gunicorn으로 프로덕션 실행
CMD ["python", "-m", "gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
