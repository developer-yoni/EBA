# 🚀 설치 및 설정 가이드

## 사전 요구사항

- Python 3.8 이상
- AWS 계정 및 자격 증명
- S3 버킷 접근 권한
- Bedrock 모델 사용 권한

## 단계별 설정

### 1. 프로젝트 클론 또는 다운로드

```bash
cd your-project-directory
```

### 2. Python 가상환경 생성 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

설치되는 패키지:
- `boto3`: AWS SDK
- `pandas`: 데이터 분석
- `openpyxl`: 엑셀 파일 처리
- `python-dotenv`: 환경 변수 관리
- `flask`: 웹 서버
- `plotly`: 시각화 (선택사항)

### 4. AWS 자격 증명 설정

#### 방법 A: 환경 변수 (.env 파일)

1. `.env.example`을 `.env`로 복사:
```bash
copy .env.example .env  # Windows
cp .env.example .env    # macOS/Linux
```

2. `.env` 파일 편집:
```env
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-northeast-2
S3_BUCKET=s3-eba-team3
S3_PREFIX=충전인프라현황DB/
KNOWLEDGE_BASE_ID=XHG5MMFIYK
MODEL_ID=global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

#### 방법 B: AWS CLI 설정

```bash
# AWS CLI 설치 (아직 설치하지 않은 경우)
# Windows: https://aws.amazon.com/cli/
# macOS: brew install awscli
# Linux: pip install awscli

# 자격 증명 설정
aws configure
```

입력 정보:
- AWS Access Key ID
- AWS Secret Access Key
- Default region: `ap-northeast-2`
- Default output format: `json`

#### 방법 C: IAM Role (EC2/Lambda 등)

EC2나 Lambda에서 실행하는 경우, IAM Role을 인스턴스에 연결하면 자동으로 인증됩니다.

### 5. IAM 권한 확인

사용자 또는 Role에 다음 권한이 필요합니다:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::s3-eba-team3",
        "arn:aws:s3:::s3-eba-team3/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:Retrieve"
      ],
      "Resource": "arn:aws:bedrock:ap-northeast-2:*:knowledge-base/XHG5MMFIYK"
    }
  ]
}
```

### 6. 연결 테스트

#### S3 연결 테스트

```python
python -c "from data_loader import ChargingDataLoader; loader = ChargingDataLoader(); print(loader.list_available_files())"
```

성공 시 S3의 파일 목록이 출력됩니다.

#### Bedrock 연결 테스트

```python
python -c "from ai_report_generator import AIReportGenerator; gen = AIReportGenerator(); print(gen.invoke_bedrock('안녕하세요'))"
```

성공 시 Bedrock의 응답이 출력됩니다.

### 7. 실행

#### CLI 모드

```bash
python cli_runner.py
```

전체 프로세스가 자동으로 실행되고 결과가 JSON 파일로 저장됩니다.

#### 웹 앱 모드

```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속

## 문제 해결

### "Unable to locate credentials"

**원인**: AWS 자격 증명을 찾을 수 없음

**해결**:
1. `.env` 파일이 있는지 확인
2. AWS CLI 설정 확인: `aws configure list`
3. 환경 변수 확인:
   ```bash
   # Windows
   echo %AWS_ACCESS_KEY_ID%
   
   # macOS/Linux
   echo $AWS_ACCESS_KEY_ID
   ```

### "No module named 'openpyxl'"

**원인**: 필요한 패키지가 설치되지 않음

**해결**:
```bash
pip install openpyxl
```

### "Access Denied" (S3)

**원인**: S3 버킷 접근 권한 없음

**해결**:
1. IAM 권한 확인
2. 버킷 이름 확인: `s3-eba-team3`
3. 리전 확인: `ap-northeast-2`

### "ValidationException" (Bedrock)

**원인**: 모델 ID가 잘못되었거나 리전에서 사용 불가

**해결**:
1. 모델 ID 확인: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
2. 리전에서 모델 사용 가능 여부 확인
3. Bedrock 서비스 활성화 확인

### Knowledge Base 검색 오류

**원인**: Knowledge Base ID가 잘못되었거나 권한 없음

**해결**:
1. Knowledge Base ID 확인: `XHG5MMFIYK`
2. IAM 권한에 `bedrock:Retrieve` 포함 확인
3. Knowledge Base가 같은 리전에 있는지 확인

## 다음 단계

1. ✅ 설치 및 설정 완료
2. 📊 CLI로 첫 리포트 생성: `python cli_runner.py`
3. 🌐 웹 앱 실행: `python app.py`
4. 🎨 필요에 따라 커스터마이징

## 추가 리소스

- [AWS Bedrock 문서](https://docs.aws.amazon.com/bedrock/)
- [boto3 문서](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [Flask 문서](https://flask.palletsprojects.com/)

## 지원

문제가 발생하면 이슈를 등록해주세요!
