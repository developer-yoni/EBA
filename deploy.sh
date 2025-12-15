#!/bin/bash
# 임시 배포 스크립트 - EC2로 Docker 이미지 전송 및 실행

set -e

# 설정 (필요에 따라 수정)
EC2_HOST="${EC2_HOST:-your-ec2-ip}"
EC2_USER="${EC2_USER:-ec2-user}"
EC2_KEY="${EC2_KEY:-~/.ssh/your-key.pem}"
IMAGE_NAME="charging-infra-app"
IMAGE_TAG="latest"
CONTAINER_PORT=5000

echo "🔨 Docker 이미지 빌드 중..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo "💾 이미지를 tar로 저장 중..."
docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > ${IMAGE_NAME}.tar.gz

echo "📤 EC2로 이미지 전송 중..."
scp -i ${EC2_KEY} ${IMAGE_NAME}.tar.gz ${EC2_USER}@${EC2_HOST}:~

echo "🚀 EC2에서 컨테이너 실행 중..."
ssh -i ${EC2_KEY} ${EC2_USER}@${EC2_HOST} << 'ENDSSH'
    # 기존 컨테이너 정리
    docker stop charging-app 2>/dev/null || true
    docker rm charging-app 2>/dev/null || true
    
    # 이미지 로드
    gunzip -c ~/charging-infra-app.tar.gz | docker load
    
    # 컨테이너 실행 (환경변수는 EC2의 ~/.env에서 로드)
    docker run -d \
        --name charging-app \
        -p 5000:5000 \
        --env-file ~/.env \
        --restart unless-stopped \
        charging-infra-app:latest
    
    # 정리
    rm ~/charging-infra-app.tar.gz
    
    echo "✅ 배포 완료! http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
ENDSSH

# 로컬 tar 파일 정리
rm ${IMAGE_NAME}.tar.gz

echo "✅ 배포 완료!"
