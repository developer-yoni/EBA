# 🔒 보안 설정 가이드

## ⚠️ 중요: GitHub Secret Scanning 오류 해결

### 문제 상황
```
remote: (?) To push, remove secret from commit(s) or follow this URL to allow the secret.
remote: https://github.com/developer-yoni/EBA/security/secret-scanning/unblock-secret/...
error: failed to push some refs
```

### 원인
코드에 민감한 정보(슬랙 Webhook URL, API 키 등)가 하드코딩되어 GitHub가 보안상 위험하다고 판단

## 🛠️ 해결 방법

### 1. 즉시 해결 (임시)
GitHub에서 제공한 URL로 이동해서 시크릿 허용:
```
https://github.com/developer-yoni/EBA/security/secret-scanning/unblock-secret/[SECRET_ID]
```

### 2. 근본적 해결 (권장)

#### A. 환경 변수 설정
1. `.env` 파일 생성:
```bash
cp .env.example .env
```

2. `.env` 파일에 실제 값 입력:
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR_WORKSPACE/YOUR_CHANNEL/YOUR_TOKEN
MCP_SLACK_KEY=your-mcp-slack-key-here
```

#### B. 코드에서 환경 변수 사용
```python
import os
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
```

### 3. Git 히스토리 정리 (필요한 경우)

민감한 정보가 포함된 커밋을 히스토리에서 제거:

```bash
# 특정 파일의 히스토리에서 민감한 정보 제거
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/sensitive/file' \
  --prune-empty --tag-name-filter cat -- --all

# 강제 푸시 (주의: 협업 시 팀원과 상의 필요)
git push origin --force --all
```

## 🔐 보안 모범 사례

### 1. 민감한 정보 관리
- ✅ 환경 변수 사용 (`.env` 파일)
- ✅ `.gitignore`에 `.env` 추가
- ✅ `.env.example`로 템플릿 제공
- ❌ 코드에 직접 하드코딩 금지

### 2. 파일 구조
```
DataReporter/
├── .env                 # 실제 값 (Git 추적 안함)
├── .env.example         # 템플릿 (Git 추적함)
├── .gitignore           # .env 포함
└── app.py              # os.getenv() 사용
```

### 3. 배포 시 주의사항
- 프로덕션 환경에서는 시스템 환경 변수 사용
- Docker 사용 시 `--env-file` 옵션 활용
- 클라우드 배포 시 각 플랫폼의 환경 변수 설정 기능 사용

## 🚨 긴급 상황 대응

### 슬랙 Webhook URL이 노출된 경우
1. 슬랙 워크스페이스에서 해당 Webhook 비활성화
2. 새로운 Webhook URL 생성
3. 코드에서 환경 변수로 변경
4. Git 히스토리 정리

### API 키가 노출된 경우
1. 해당 서비스에서 키 즉시 비활성화
2. 새로운 키 생성
3. 환경 변수로 관리 체계 변경

---

💡 **Tip**: 민감한 정보는 절대 코드에 직접 포함하지 말고, 항상 환경 변수나 보안 저장소를 사용하세요!