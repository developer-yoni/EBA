#!/bin/bash
# 충전 인프라 앱 배포 스크립트
# 사용법: ./deploy.sh

set -e

# 설정
EC2_HOST="10.100.110.55"
EC2_USER="gs"
# SSH 키 없이 비밀번호 인증 사용
EC2_DIR="/usr2/EBA"
IMAGE_NAME="charging-infra-app"

echo "=========================================="
echo "🚀 충전 인프라 앱 배포 시작"
echo "=========================================="
echo "📍 대상 서버: ${EC2_USER}@${EC2_HOST}:${EC2_DIR}"
echo ""

# 1. Docker 이미지 빌드 (EC2가 AMD64이므로 플랫폼 지정)
echo "🔨 [1/4] Docker 이미지 빌드 중 (linux/amd64)..."
docker build --no-cache --platform linux/amd64 -t ${IMAGE_NAME} .
echo "✅ 이미지 빌드 완료"
echo ""

# 2. 이미지를 tar.gz로 저장
echo "💾 [2/4] 이미지를 tar.gz로 저장 중..."
docker save ${IMAGE_NAME} | gzip > ${IMAGE_NAME}.tar.gz
echo "✅ 저장 완료: ${IMAGE_NAME}.tar.gz ($(du -h ${IMAGE_NAME}.tar.gz | cut -f1))"
echo ""

# 3. EC2로 전송
echo "📤 [3/4] EC2로 이미지 전송 중..."
scp ${IMAGE_NAME}.tar.gz ${EC2_USER}@${EC2_HOST}:${EC2_DIR}/
echo "✅ 전송 완료"
echo ""

# 4. EC2에서 컨테이너 실행
echo "🚀 [4/4] EC2에서 컨테이너 실행 중..."
ssh ${EC2_USER}@${EC2_HOST} << ENDSSH
    cd ${EC2_DIR}
    
    # 이미지 로드
    echo "📦 이미지 로드 중..."
    gunzip -c ${IMAGE_NAME}.tar.gz | docker load
    
    # 기존 컨테이너 정리
    echo "🧹 기존 컨테이너 정리..."
    docker stop charging-app 2>/dev/null || true
    docker rm charging-app 2>/dev/null || true
    
    # 컨테이너 실행
    echo "🏃 컨테이너 실행..."
    docker run -d \\
        --name charging-app \\
        -p 5001:5001 \\
        --env-file .env \\
        --restart unless-stopped \\
        ${IMAGE_NAME}
    
    # tar.gz 파일 정리
    rm ${IMAGE_NAME}.tar.gz
    
    # 상태 확인
    echo ""
    echo "📊 컨테이너 상태:"
    docker ps | grep charging-app
ENDSSH

# 로컬 tar.gz 파일 정리
rm ${IMAGE_NAME}.tar.gz

echo ""
echo "=========================================="
echo "✅ 배포 완료!"
echo "🌐 접속: http://${EC2_HOST}:5001"
echo "=========================================="
