# ⚡ 빠른 시작 가이드

5분 안에 시스템을 실행하세요!

## 1️⃣ 의존성 설치 (1분)

```bash
pip install boto3 pandas openpyxl python-dotenv flask
```

## 2️⃣ AWS 자격 증명 설정 (2분)

### 옵션 A: .env 파일 사용

```bash
# .env.example을 .env로 복사
copy .env.example .env

# .env 파일 편집 (메모장으로 열기)
notepad .env
```

다음 내용 입력:
```env
AWS_ACCESS_KEY_ID=여기에_액세스_키
AWS_SECRET_ACCESS_KEY=여기에_시크릿_키
AWS_REGION=ap-northeast-2
```

### 옵션 B: AWS CLI 사용

```bash
aws configure
```

## 3️⃣ 연결 테스트 (1분)

```bash
python test_connection.py
```

모든 테스트가 ✅ 통과하면 준비 완료!

## 4️⃣ 실행 (1분)

### CLI 모드 (추천 - 첫 실행)

```bash
python cli_runner.py
```

결과가 `charging_infrastructure_report.json`에 저장됩니다.

### 웹 앱 모드

```bash
python app.py
```

브라우저에서 http://localhost:5000 접속

## 🎯 웹 앱 사용법

1. **"📥 최신 데이터 로드"** 클릭
   - S3에서 가장 최신 엑셀 파일을 자동으로 로드합니다
   - 데이터 정보가 화면에 표시됩니다

2. **"📊 데이터 분석"** 클릭
   - CPO별, 지역별, 충전기 유형별 분석을 수행합니다
   - 몇 초 안에 완료됩니다

3. **"🤖 AI 리포트 생성"** 클릭
   - Bedrock과 Knowledge Base를 활용하여 상세 리포트를 생성합니다
   - 1-2분 정도 소요됩니다

4. **생성된 리포트 확인**
   - 경영진 요약
   - CPO 분석
   - 지역별 분석
   - 트렌드 및 예측

5. **커스텀 질의** (선택사항)
   - 하단의 입력창에 질문을 입력하세요
   - 예: "서울 지역의 충전 인프라 현황은?"
   - AI가 데이터를 기반으로 답변합니다

## 🔧 문제 해결

### "Unable to locate credentials"
→ `.env` 파일을 확인하거나 `aws configure` 실행

### "Access Denied"
→ IAM 권한 확인 (S3, Bedrock, Knowledge Base)

### "No module named..."
→ `pip install -r requirements.txt` 실행

### 자세한 문제 해결
→ `SETUP_GUIDE.md` 참고

## 📚 더 알아보기

- **전체 문서**: `README.md`
- **설치 가이드**: `SETUP_GUIDE.md`
- **프로젝트 구조**: `PROJECT_STRUCTURE.md`

## 💡 팁

1. **첫 실행은 CLI 모드로**: 전체 프로세스를 한 번에 확인
2. **웹 앱으로 탐색**: 대화형으로 데이터 분석
3. **커스텀 질의 활용**: 특정 질문에 대한 답변 얻기

## 🎉 완료!

이제 한국 전기차 충전 인프라 데이터를 자동으로 분석할 수 있습니다!
