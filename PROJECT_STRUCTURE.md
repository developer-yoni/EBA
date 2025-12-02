# 📁 프로젝트 구조

```
charging-infrastructure-analysis/
│
├── 📄 config.py                    # 환경 설정 (AWS, S3, Bedrock, KB)
├── 📄 requirements.txt             # Python 의존성
├── 📄 .env.example                 # 환경 변수 템플릿
├── 📄 .gitignore                   # Git 제외 파일
│
├── 📊 데이터 처리
│   ├── data_loader.py              # S3 데이터 로드 및 파싱
│   └── data_analyzer.py            # 데이터 분석 및 집계
│
├── 🤖 AI 리포트
│   └── ai_report_generator.py      # Bedrock + KB 기반 리포트 생성
│
├── 🌐 웹 애플리케이션
│   ├── app.py                      # Flask 웹 서버 및 API
│   └── templates/
│       └── index.html              # 웹 UI
│
├── 🔧 유틸리티
│   ├── cli_runner.py               # CLI 실행 스크립트
│   ├── test_connection.py          # 연결 테스트
│   └── analyze_kb_data.py          # KB 데이터 구조 분석
│
└── 📚 문서
    ├── README.md                   # 프로젝트 개요
    ├── SETUP_GUIDE.md              # 설치 가이드
    └── PROJECT_STRUCTURE.md        # 이 파일
```

## 핵심 모듈 설명

### 1. config.py
**역할**: 전역 설정 관리

**주요 설정**:
- AWS 리전: `ap-northeast-2`
- S3 버킷: `s3-eba-team3`
- Bedrock 모델: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Knowledge Base ID: `XHG5MMFIYK`
- 데이터 구조: 헤더 행, 제목 위치

**사용 예시**:
```python
from config import Config
print(Config.AWS_REGION)
```

### 2. data_loader.py
**역할**: S3에서 엑셀 데이터 로드

**주요 클래스**: `ChargingDataLoader`

**주요 메서드**:
- `list_available_files()`: S3 파일 목록 조회
- `download_file(s3_key)`: 파일 다운로드
- `parse_snapshot_date(excel_file)`: 날짜 추출
- `load_data(s3_key)`: 데이터 로드 및 파싱
- `load_latest()`: 최신 파일 로드
- `load_multiple(months)`: 여러 월 데이터 로드

**사용 예시**:
```python
from data_loader import ChargingDataLoader

loader = ChargingDataLoader()
df = loader.load_latest()
print(df.head())
```

### 3. data_analyzer.py
**역할**: 데이터 분석 및 통계

**주요 클래스**: `ChargingDataAnalyzer`

**주요 메서드**:
- `get_summary_stats()`: 전체 요약
- `analyze_by_cpo()`: CPO별 분석
- `analyze_by_region()`: 지역별 분석
- `analyze_charger_types()`: 충전기 유형별 분석
- `trend_analysis()`: 시계열 트렌드
- `top_performers(n)`: 상위 N개 사업자
- `generate_insights()`: 종합 인사이트

**사용 예시**:
```python
from data_analyzer import ChargingDataAnalyzer

analyzer = ChargingDataAnalyzer(df)
insights = analyzer.generate_insights()
print(insights['summary'])
```

### 4. ai_report_generator.py
**역할**: AI 기반 분석 리포트 생성

**주요 클래스**: `AIReportGenerator`

**주요 메서드**:
- `retrieve_from_kb(query)`: Knowledge Base 검색
- `invoke_bedrock(prompt, context)`: Bedrock 모델 호출
- `generate_executive_summary(insights)`: 경영진 요약
- `generate_cpo_analysis(cpo_data)`: CPO 분석
- `generate_regional_analysis(region_data)`: 지역 분석
- `generate_trend_forecast(trend_data)`: 트렌드 예측
- `generate_full_report(insights)`: 전체 리포트

**사용 예시**:
```python
from ai_report_generator import AIReportGenerator

generator = AIReportGenerator()
report = generator.generate_full_report(insights)
print(report['executive_summary'])
```

### 5. app.py
**역할**: Flask 웹 서버 및 REST API

**API 엔드포인트**:
- `GET /`: 메인 페이지
- `GET /api/files`: S3 파일 목록
- `POST /api/load`: 데이터 로드
- `GET /api/analyze`: 데이터 분석
- `GET /api/generate-report`: AI 리포트 생성
- `POST /api/query`: 커스텀 질의

**사용 예시**:
```bash
python app.py
# 브라우저에서 http://localhost:5000 접속
```

### 6. cli_runner.py
**역할**: 명령줄 인터페이스 실행

**실행 흐름**:
1. 최신 데이터 로드
2. 데이터 분석
3. AI 리포트 생성
4. 결과를 JSON 파일로 저장

**사용 예시**:
```bash
python cli_runner.py
# 결과: charging_infrastructure_report.json
```

### 7. test_connection.py
**역할**: 시스템 연결 테스트

**테스트 항목**:
1. Python 패키지 import
2. AWS 자격 증명
3. S3 접근
4. Bedrock 모델 호출
5. Knowledge Base 검색

**사용 예시**:
```bash
python test_connection.py
```

## 데이터 흐름

```
┌─────────────┐
│  S3 Bucket  │
│  (엑셀 파일) │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  data_loader    │ ← 파일 다운로드 및 파싱
│  - 날짜 추출     │
│  - DataFrame 생성│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ data_analyzer   │ ← 통계 분석
│  - CPO별        │
│  - 지역별       │
│  - 트렌드       │
└──────┬──────────┘
       │
       ▼
┌──────────────────────┐
│ ai_report_generator  │ ← AI 리포트 생성
│  ┌────────────────┐  │
│  │ Knowledge Base │  │ ← 컨텍스트 검색
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │    Bedrock     │  │ ← 텍스트 생성
│  └────────────────┘  │
└──────┬───────────────┘
       │
       ▼
┌─────────────────┐
│  웹 UI / JSON   │ ← 결과 출력
└─────────────────┘
```

## 확장 가능성

### 새로운 분석 추가
`data_analyzer.py`에 메서드 추가:
```python
def analyze_new_metric(self):
    # 새로운 분석 로직
    return result
```

### 새로운 리포트 섹션 추가
`ai_report_generator.py`에 메서드 추가:
```python
def generate_new_section(self, data):
    prompt = f"분석: {data}"
    return self.invoke_bedrock(prompt)
```

### 새로운 API 엔드포인트 추가
`app.py`에 라우트 추가:
```python
@app.route('/api/new-endpoint')
def new_endpoint():
    # 로직
    return jsonify(result)
```

## 환경별 설정

### 개발 환경
```python
# config.py
DEBUG = True
```

### 프로덕션 환경
```python
# config.py
DEBUG = False
# 환경 변수 또는 AWS Secrets Manager 사용
```

## 보안 고려사항

1. **자격 증명**: `.env` 파일을 Git에 커밋하지 않음
2. **IAM 권한**: 최소 권한 원칙 적용
3. **데이터 암호화**: S3 버킷 암호화 활성화
4. **API 보안**: 프로덕션에서는 인증 추가 권장

## 성능 최적화

1. **캐싱**: `app.py`에서 데이터 캐싱 사용
2. **배치 처리**: 여러 파일 동시 로드
3. **비동기 처리**: 대용량 데이터 처리 시 고려

## 모니터링

- CloudWatch Logs로 로그 수집
- 에러 추적 및 알림 설정
- 성능 메트릭 모니터링
