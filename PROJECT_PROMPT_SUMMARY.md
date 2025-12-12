# 🔌 한국 전기차 충전 인프라 분석 시스템 - AI Agent 프롬프트

## 시스템 개요
AWS Bedrock + Knowledge Base 기반 충전 인프라 현황 자동 분석 및 AI 리포트 생성 시스템

## 핵심 아키텍처
```
S3(엑셀) → DataLoader → DataAnalyzer → AIReportGenerator(Bedrock+KB) → Flask API
                                     ↓
                              ScenarioSimulator(ML예측)
                                     ↓
                              QueryAnalyzer(RAG+차트)
```

## 데이터 구조

### 입력 데이터 (S3 엑셀)
- 파일명: `충전인프라 현황_YYMM.xlsx`
- 헤더 위치: 4행 (0-indexed)
- 요약 데이터: K2:P4 (전체CPO, 당월증감량)

### 핵심 컬럼
| 컬럼명 | 설명 | 타입 |
|--------|------|------|
| CPO명 | 충전사업자명 | 문자열 |
| 순위 | 시장점유율 순위 | 정수 |
| 충전소수 | 충전소 개수 | 정수 |
| 완속충전기/급속충전기/총충전기 | 충전기 수 | 정수 |
| 시장점유율 | 전체 대비 비율 | 백분율 |
| 완속증감/급속증감/총증감 | 전월 대비 변화량 | 정수 |
| snapshot_month | 기준 연월 (YYYY-MM) | 문자열 |

## 핵심 모듈 역할

### 1. DataLoader
- S3 파일 목록 조회 및 다운로드
- 파일명에서 날짜 추출 (`_YYMM` 패턴)
- 엑셀 K2:P4에서 요약 데이터 추출
- 다중 월 데이터 병합

### 2. DataAnalyzer
- CPO별/지역별 집계 분석
- 시계열 트렌드 분석
- 상위 N개 사업자 랭킹
- 기간별 증감량 계산

### 3. AIReportGenerator
- Knowledge Base 검색 (RAG)
- Bedrock Claude 모델 호출
- 3종 리포트 병렬 생성:
  - KPI Snapshot Report (현황)
  - CPO Ranking Report (경쟁분석)
  - Monthly Trend Report (추세)

### 4. ScenarioSimulator
- ML 기반 시장점유율 예측
- 핵심 공식: `점유율 = GS충전기 / 시장전체 * 100`
- LinearRegression으로 GS충전기, 시장전체 각각 예측 후 계산
- 백테스트 기반 신뢰도 평가 (MAPE 2% 이하 = 신뢰)

### 5. QueryAnalyzer
- 자연어 질의 의도 분석 (Multi-Step Reasoning)
- 컬럼 매핑 (Semantic Matching)
- 동적 차트 생성 (matplotlib)

## AI 리포트 생성 프롬프트 패턴

### 공통 규칙
```
<role>시니어 데이터 애널리스트</role>
<context>
- 데이터: CPO별 월별 충전소/충전기 수
- 우리 회사: GS차지비
- 금지: 차량수, 매출, 이용률 등 없는 데이터 생성
</context>
```

### 날짜 할루시네이션 방지
```
## ⚠️ 날짜 정보 - 반드시 이 날짜만 사용
**분석 기간:** {start_month} ~ {end_month}
**기준월:** {target_month}
⚠️ 다른 날짜(예: 2024-10)를 절대 만들지 마세요.
```

### 레이아웃 규칙
```
1) HTML 타이틀:
<div align="center">
<h1>📊 EV Infra KPI Report</h1>
<p>분석 기간: {start} ~ {end} | 기준월: {target}</p>
</div>

2) 섹션별 콜아웃:
> 💡 **Key Insight**
> 핵심 메시지 2문장

3) 표 형식:
| Rank | CPO | Chargers | Share |
|------|-----|----------|-------|
```

## ML 예측 로직

### 신뢰도 계산
```python
confidence_score = (
    data_score * 0.25 +      # 데이터 양 (3개월=30점, 12개월=100점)
    trend_score * 0.35 +     # 추세 안정성 (R² * 일관성)
    volatility_score * 0.40  # 변동성 역수
)
# HIGH >= 80, MEDIUM >= 60, LOW < 60
```

### 예측 방식 (Ratio Method)
```python
# 각각 예측 후 점유율 계산 (직접 예측보다 45.8% 정확)
pred_gs = lr_gs.predict(future_idx)
pred_market = lr_market.predict(future_idx)
pred_share = (pred_gs / pred_market) * 100
```

### 백테스트 결과
| 기간 | MAPE | MAE | 신뢰도 |
|------|------|-----|--------|
| 1개월 | 1.05% | 0.17 | HIGH |
| 3개월 | 1.20% | 0.19 | HIGH |
| 6개월 | 1.70% | 0.28 | HIGH |

## 질의 분석 프롬프트 (QueryAnalyzer)

### Multi-Step Reasoning
```
Step 1: 질의 핵심 요소 추출
- 대상: 완속충전기, 급속충전기 등
- 측정값: 개수, 증감량, 증가률
- 조건: 기간, CPO, 상위 N개

Step 2: 컬럼 매핑 (Semantic Matching)
- "완속 증가량" → 완속증감 (HIGH)
- "증가률" → REQUIRES_CALCULATION

Step 3: 확신도 판정
- HIGH: 명확한 매핑
- REQUIRES_CALCULATION: 계산 필요
- NOT_FOUND: 데이터 없음

Step 4: 최종 결정
- PROCEED / CLARIFY / CALCULATE
```

### 출력 형식 결정
```
- 차트 키워드 있음 → needs_chart: true
- 표/테이블 키워드 또는 없음 → needs_chart: false (기본값)
- "기타" 키워드 있음 → include_others: true
```

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| /api/load | POST | 전체 월 데이터 로드 |
| /api/filter | POST | 기준월 필터링 |
| /api/dashboard | POST | 대시보드 데이터 |
| /api/generate-all-reports | POST | AI 리포트 3종 병렬 생성 |
| /api/simulate | POST | 시장점유율 시뮬레이션 |
| /api/query | POST | 커스텀 질의 |

## 환경 설정

### 필수 환경변수
```
AWS_REGION=ap-northeast-2
S3_BUCKET=s3-eba-team3
S3_PREFIX=충전인프라현황DB/
KNOWLEDGE_BASE_ID=XHG5MMFIYK
MODEL_ID=global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

### 주요 설정값
```python
MAX_TOKENS = 5120
TEMPERATURE = 0.7
KB_NUMBER_OF_RESULTS = 2
HEADER_ROW = 4  # 엑셀 헤더 위치
```

## 핵심 설계 원칙

1. **RAG 기반 정확성**: Knowledge Base 검색 후 컨텍스트 주입
2. **할루시네이션 방지**: 제공된 데이터만 사용, 날짜 명시
3. **병렬 처리**: ThreadPoolExecutor로 3개 리포트 동시 생성
4. **백테스트 검증**: ML 예측 신뢰도를 실제 데이터로 검증
5. **Semantic Matching**: 자연어 → 컬럼 매핑 자동화

## 확장 포인트

- `data_analyzer.py`: 새 분석 메서드 추가
- `ai_report_generator.py`: 새 리포트 섹션 추가
- `scenario_simulator.py`: 예측 모델 개선
- `query_analyzer.py`: 질의 패턴 확장
