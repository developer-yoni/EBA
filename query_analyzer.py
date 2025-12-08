"""
커스텀 질의 분석 및 차트 생성 모듈
- RAG 연동
- 프롬프트 엔지니어링
- 코드 인터프리터 연동
"""
import json
import re
import boto3
import pandas as pd
from config import Config
from chart_generator import ChartGenerator

class QueryAnalyzer:
    """질의 분석 및 동적 차트 생성"""
    
    def __init__(self):
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
        )
        self.kb_client = boto3.client(
            'bedrock-agent-runtime',
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
        )
        self.chart_generator = ChartGenerator()
    
    def retrieve_from_kb(self, query: str) -> str:
        """Knowledge Base에서 관련 정보 검색 (RAG)"""
        try:
            response = self.kb_client.retrieve(
                knowledgeBaseId=Config.KNOWLEDGE_BASE_ID,
                retrievalQuery={'text': query},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': Config.KB_NUMBER_OF_RESULTS
                    }
                }
            )
            
            results = response.get('retrievalResults', [])
            
            # RAG 검색 결과 상세 로깅
            print(f'   └─ 🔍 RAG 검색 결과: {len(results)}개 문서 검색됨', flush=True)
            
            if not results:
                print(f'      └─ ⚠️ 관련 문서 없음', flush=True)
                return ''
            
            for i, r in enumerate(results):
                score = r.get('score', 0)
                location = r.get('location', {})
                s3_uri = location.get('s3Location', {}).get('uri', 'N/A')
                content_preview = r.get('content', {}).get('text', '')[:100]
                print(f'      [{i+1}] 관련도: {score:.4f}', flush=True)
                print(f'          소스: {s3_uri}', flush=True)
                print(f'          내용: {content_preview}...', flush=True)
            
            context = '\n\n'.join([
                f"[참고자료 {i+1}] (관련도: {r.get('score', 0):.2f})\n{r.get('content', {}).get('text', '')}"
                for i, r in enumerate(results)
            ])
            
            return context
        except Exception as e:
            print(f'   └─ ❌ KB 검색 오류: {e}', flush=True)
            return ''
    
    def analyze_query_intent(self, query: str, available_data: dict) -> dict:
        """질의 의도 분석 - Multi-Step Reasoning + Semantic Column Matching"""
        
        analysis_prompt = f"""
당신은 데이터 분석 질의를 정확하게 분석하는 전문가입니다.
사용자의 질의를 분석하여 어떤 데이터를 어떤 형식으로 보여줄지 결정합니다.

## 🚨 매우 중요: 출력 형식 결정 규칙

### 시각화(차트/그래프)가 필요한 경우 (needs_chart: true)
다음 키워드가 **명시적으로** 포함된 경우에만 차트를 생성합니다:
- "그래프", "차트", "시각화", "그려줘", "막대", "원형", "파이", "선형", "라인"
- 예: "막대그래프로 보여줘", "원형 차트로 그려줘", "시각화해줘"

### 표(테이블) 형식으로 답변하는 경우 (needs_chart: false) - 기본값!
다음 경우에는 차트 없이 **표(테이블) 형식**으로 답변합니다:
- "표", "테이블", "목록", "리스트" 키워드가 있는 경우
- 시각화 관련 키워드가 **전혀 없는** 경우 (기본값)
- "표로 보여줘", "목록으로 알려줘" 등

### 표도 차트도 필요 없는 경우 (needs_chart: false, show_table: false)
- "표 없이", "표 말고", "텍스트로만", "간단히" 등의 표현이 있는 경우

⚠️ 주의: "원형 표"는 "원형 그래프"가 아닙니다! "표"가 포함되면 테이블 형식입니다.

## 사용자 질의
{query}

## 데이터베이스 스키마 (사용 가능한 컬럼)

### 기본 수량 컬럼 (절대값)
| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 충전소수 | 충전소 개수 | 정수 (예: 1000, 2000) |
| 완속충전기 | 완속충전기 개수 | 정수 (예: 5000, 10000) |
| 급속충전기 | 급속충전기 개수 | 정수 (예: 1000, 2000) |
| 총충전기 | 총충전기 개수 | 정수 (예: 6000, 12000) |

### 증감 컬럼 (전월 대비 변화량, 양수/음수)
| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 충전소증감 | 전월 대비 충전소 증감량 | 정수 (예: +50, -10) |
| 완속증감 | 전월 대비 완속충전기 증감량 | 정수 (예: +100, -50) |
| 급속증감 | 전월 대비 급속충전기 증감량 | 정수 (예: +30, -20) |
| 총증감 | 전월 대비 총충전기 증감량 | 정수 (예: +130, -70) |

### 비율/순위 컬럼
| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 시장점유율 | 전체 대비 점유율 | 백분율 (예: 15.5%) |
| 순위 | 시장점유율 순위 | 정수 (예: 1, 2, 3) |
| 순위변동 | 전월 대비 순위 변동 | 정수 (예: +1, -2) |

### 식별 컬럼
| 컬럼명 | 설명 |
|--------|------|
| CPO명 | 충전사업자명 (예: GS차지비, 파워큐브) |
| snapshot_month | 기준 연월 (예: 2025-10) |

## 🔑 매우 중요: "전체 CPO" 용어 이해 (엑셀 셀 위치 기반)

### "전체 CPO" 또는 "충전사업자" 키워드 처리 (대소문자/공백 무시)
다음 표현들은 모두 **전체 합계 데이터**를 의미합니다:
- "전체 CPO", "전체CPO", "전체cpo", "전체 cpo"
- "전체 충전사업자", "전체충전사업자", "충전사업자"
- "CPO 개수", "충전사업자 개수", "충전사업자 수"

### 엑셀 셀 위치 매핑 (L3:P4 범위)

#### 전체 현황 (3행 = 전체CPO 행)
| 질의 표현 | 엑셀 셀 | 매핑 컬럼 |
|-----------|---------|-----------|
| 전체 CPO 개수, 충전사업자 개수 | L3 | total_cpos |
| 전체 충전소 개수 | M3 | total_stations |
| 전체 완속충전기 개수 | N3 | total_slow_chargers |
| 전체 급속충전기 개수 | O3 | total_fast_chargers |
| 전체 충전기 개수 | P3 | total_chargers |

#### 당월 증감량 (4행 = 당월증감량 행)
| 질의 표현 | 엑셀 셀 | 매핑 컬럼 |
|-----------|---------|-----------|
| 전체 CPO 당월 증감량 | L4 | change_cpos |
| 전체 충전소 당월 증감량 | M4 | change_stations |
| 전체 완속충전기 당월 증감량 | N4 | change_slow_chargers |
| 전체 급속충전기 당월 증감량 | O4 | change_fast_chargers |
| 전체 충전기 당월 증감량 | P4 | change_total_chargers |

### 중요 규칙:
1. "전체 CPO"는 모든 CPO를 나열하는 것이 **아닙니다**!
2. "전체 CPO"는 엑셀 요약 행(L3:P4)의 합계 데이터를 의미합니다
3. cpo_name: "전체"로 설정하면 엑셀 요약 행 데이터를 사용합니다
4. display_column에 위 매핑 컬럼명을 사용합니다

### 예시:
- "전체 CPO 개수 변화" → cpo_name: "전체", display_column: "total_cpos"
- "전체 완속충전기 증가량" → cpo_name: "전체", display_column: "change_slow_chargers"
- "전체 급속충전기 당월 증감량" → cpo_name: "전체", display_column: "change_fast_chargers"

## ⚠️ 중요: 데이터에 없는 항목
다음 항목들은 데이터베이스에 **직접 존재하지 않습니다**:
- "증가률", "증가율", "성장률" → 계산 필요 (증감량 / 이전값 * 100)
- "감소률", "감소율" → 계산 필요
- "점유율 변동" → 직접 컬럼 없음

## 사용 가능한 데이터
- 기간: {available_data.get('available_months', [])}
- CPO 수: {len(available_data.get('available_cpos', []))}개
- 실제 컬럼: {available_data.get('available_columns', [])}

---
## Multi-Step Reasoning 분석 과정

### Step 1: 질의 핵심 요소 추출
사용자 질의에서 다음을 식별하세요:
- 대상: 무엇에 대한 질의인가? (완속충전기, 급속충전기, 충전소 등)
- 측정값: 어떤 값을 보고 싶은가? (개수, 증감량, 증가률, 점유율 등)
- 조건: 기간, 특정 CPO, 상위/하위 몇 개 등
- **CPO 범위**: 
  - "전체 CPO", "충전사업자 개수", "CPO 개수" → cpo_name: "전체" (요약 행 데이터)
  - "GS차지비", "에버온" 등 특정 CPO명 → cpo_name: 해당 CPO명
  - CPO명 언급 없음 → cpo_name: null (전체 또는 상위 N개)
- 출력형식: 
  - "chart" = 차트/그래프 키워드가 명시적으로 있음
  - "table" = 표/테이블 키워드가 있거나, 시각화 키워드가 없음 (기본값)
  - "text_only" = 표도 필요 없다고 명시함

### Step 2: 측정값 → 컬럼 매핑 (Semantic Matching)
사용자가 요청한 "측정값"을 실제 컬럼에 매핑합니다.

**직접 매핑 가능한 경우:**
| 사용자 표현 | 매핑 컬럼 | 확신도 |
|------------|----------|--------|
| "완속충전기 개수", "완속 수" | 완속충전기 | HIGH |
| "완속충전기 증감량", "완속 증가량" | 완속증감 | HIGH |
| "급속충전기 개수" | 급속충전기 | HIGH |
| "급속충전기 증감량" | 급속증감 | HIGH |
| "시장점유율", "점유율" | 시장점유율 | HIGH |
| "충전소 수", "충전소 개수" | 충전소수 | HIGH |
| "충전소 증감량" | 충전소증감 | HIGH |

**계산이 필요한 경우 (데이터에 직접 없음):**
| 사용자 표현 | 필요한 계산 | 확신도 |
|------------|------------|--------|
| "증가률", "증가율", "성장률" | (증감량/이전값)*100 | REQUIRES_CALCULATION |
| "감소률" | 계산 필요 | REQUIRES_CALCULATION |

**매핑 불가능한 경우:**
- 컬럼을 특정할 수 없는 모호한 표현
- 데이터에 없는 항목 요청

### Step 3: 확신도 판정
- HIGH: 명확하게 컬럼 매핑 가능
- MEDIUM: 유사한 컬럼이 있지만 확인 필요
- LOW: 모호하여 사용자 확인 필요
- REQUIRES_CALCULATION: 계산이 필요한 파생 지표
- NOT_FOUND: 해당 데이터 없음

### Step 4: 최종 결정
- 확신도가 HIGH면 → 차트 생성 진행
- 확신도가 MEDIUM이면 → 가장 유사한 컬럼으로 진행하되 설명 추가
- 확신도가 LOW/NOT_FOUND면 → 사용자에게 명확화 요청
- REQUIRES_CALCULATION이면 → 계산 로직 적용 또는 대안 제시

---
## 출력 형식

```json
{{
    "reasoning": {{
        "step1_extraction": {{
            "target": "대상 (예: 완속충전기)",
            "metric": "측정값 (예: 증가률)",
            "conditions": "조건 (예: 2025-10, top 3)",
            "visualization": "시각화 타입"
        }},
        "step2_column_mapping": {{
            "user_expression": "사용자가 사용한 표현",
            "mapped_column": "매핑된 컬럼명 또는 null",
            "mapping_reason": "매핑 이유 설명"
        }},
        "step3_confidence": {{
            "level": "HIGH | MEDIUM | LOW | REQUIRES_CALCULATION | NOT_FOUND",
            "reason": "확신도 판정 이유"
        }},
        "step4_decision": {{
            "action": "PROCEED | CLARIFY | CALCULATE",
            "explanation": "결정 설명"
        }}
    }},
    "needs_chart": true | false,
    "show_table": true | false,
    "output_format": "chart | table | text_only",
    "needs_clarification": true | false,
    "clarification_message": "사용자에게 요청할 명확화 메시지 (needs_clarification이 true일 때)",
    "chart_type": "line | bar | pie | area | none",
    "chart_title": "차트/표 제목",
    "data_filter": {{
        "cpo_name": null,
        "start_month": "YYYY-MM",
        "end_month": "YYYY-MM",
        "sort_column": "정렬 기준 컬럼",
        "display_column": "표시할 값 컬럼",
        "limit": 숫자,
        "sort_order": "desc | asc",
        "include_others": true | false,
        "others_label": "기타 또는 사용자가 지정한 라벨 (예: others, 나머지 등)"
    }},

## 🚨 매우 중요: include_others 규칙 (기타 항목 포함 여부)

### include_others: true로 설정하는 경우 (명시적 요청 필수!)
다음 키워드가 **명시적으로** 포함된 경우에만 true로 설정합니다:
- "기타", "나머지", "others", "그 외", "외 나머지", "포함해서", "합쳐서"
- 예: "top 5와 기타", "나머지는 others로", "기타 포함", "나머지 합쳐서"

### include_others: false로 설정하는 경우 (기본값!)
- 위 키워드가 **전혀 없는** 경우 → 기본값 false
- "top 5를 원형그래프로" → include_others: false (기타 언급 없음)
- "시장점유율 상위 3개" → include_others: false (기타 언급 없음)

⚠️ 주의: 원형그래프(파이차트)라고 해서 자동으로 기타를 포함하지 않습니다!
사용자가 명시적으로 "기타", "나머지" 등을 요청한 경우에만 include_others: true로 설정하세요.
    "chart_config": {{
        "x_axis": "CPO명",
        "y_axis": "표시할 데이터명",
        "y_axis_type": "value | percentage | calculated_rate",
        "y_axis_label": "y축 라벨"
    }},
    "analysis_type": "ranking | trend | comparison",
    "calculation_required": {{
        "needed": true | false,
        "type": "growth_rate | null",
        "base_column": "기준 컬럼",
        "change_column": "변화량 컬럼"
    }}
}}
```

## 예시

### 예시 1: "완속충전기 증가량 top 5" (표 형식 - 시각화 키워드 없음)
```json
{{
    "reasoning": {{
        "step1_extraction": {{"output_format": "table"}},
        "step2_column_mapping": {{"mapped_column": "완속증감"}}
    }},
    "needs_chart": false,
    "show_table": true,
    "output_format": "table",
    "chart_type": "none",
    "data_filter": {{"sort_column": "완속증감", "display_column": "완속증감", "limit": 5}}
}}
```

### 예시 2: "완속충전기 증가량 top 5 막대그래프로 그려줘" (차트 필요)
```json
{{
    "reasoning": {{
        "step1_extraction": {{"output_format": "chart"}}
    }},
    "needs_chart": true,
    "show_table": true,
    "output_format": "chart",
    "chart_type": "bar",
    "data_filter": {{"sort_column": "완속증감", "display_column": "완속증감", "limit": 5}}
}}
```

### 예시 3: "완속충전기 증가률 top 3를 표로 보여줘" (표 형식 + 계산 필요)
```json
{{
    "reasoning": {{
        "step1_extraction": {{"output_format": "table"}},
        "step3_confidence": {{"level": "REQUIRES_CALCULATION"}}
    }},
    "needs_chart": false,
    "show_table": true,
    "output_format": "table",
    "chart_type": "none",
    "calculation_required": {{"needed": true, "type": "growth_rate", "base_column": "완속충전기", "change_column": "완속증감"}}
}}
```

### 예시 4: "완속충전기 증가률 top 3 간단히 알려줘" (텍스트만)
```json
{{
    "needs_chart": false,
    "show_table": false,
    "output_format": "text_only"
}}
```

### 예시 5: "원형 표로 보여줘" (표 형식! 차트 아님!)
```json
{{
    "reasoning": {{"step1_extraction": {{"output_format": "table", "note": "원형 표는 차트가 아닌 테이블 형식"}}}},
    "needs_chart": false,
    "show_table": true,
    "output_format": "table",
    "chart_type": "none"
}}
```

### 예시 6: "2025년 1월부터 10월까지 전체 CPO 개수 변화를 알려줘" (엑셀 L3 데이터)
```json
{{
    "reasoning": {{
        "step1_extraction": {{
            "target": "CPO 개수 (충전사업자 수)",
            "metric": "개수 변화",
            "conditions": "2025-01~2025-10, 전체 CPO (엑셀 L3)",
            "cpo_scope": "전체 CPO = 엑셀 요약 행 L3 데이터"
        }},
        "step2_column_mapping": {{
            "user_expression": "전체 CPO 개수",
            "mapped_column": "total_cpos",
            "mapping_reason": "전체 CPO 개수는 엑셀 L3 셀 값 (total_cpos)"
        }}
    }},
    "needs_chart": false,
    "show_table": true,
    "output_format": "table",
    "chart_type": "none",
    "analysis_type": "trend",
    "data_filter": {{
        "cpo_name": "전체",
        "start_month": "2025-01",
        "end_month": "2025-10",
        "sort_column": "snapshot_month",
        "display_column": "total_cpos",
        "sort_order": "asc"
    }}
}}
```

### 예시 7: "전체 CPO의 완속충전기 당월 증감량을 그래프로 그려줘" (엑셀 N4 데이터)
```json
{{
    "reasoning": {{
        "step1_extraction": {{
            "target": "완속충전기",
            "metric": "당월 증감량",
            "conditions": "전체 CPO (엑셀 N4)",
            "output_format": "chart"
        }},
        "step2_column_mapping": {{
            "user_expression": "전체 완속충전기 증감량",
            "mapped_column": "change_slow_chargers",
            "mapping_reason": "전체 완속충전기 당월 증감량은 엑셀 N4 셀 값"
        }}
    }},
    "needs_chart": true,
    "show_table": true,
    "output_format": "chart",
    "chart_type": "line",
    "analysis_type": "trend",
    "data_filter": {{
        "cpo_name": "전체",
        "display_column": "change_slow_chargers"
    }}
}}
```

### 예시 8: "전체 충전사업자의 급속충전기 개수를 알려줘" (엑셀 O3 데이터)
```json
{{
    "reasoning": {{
        "step1_extraction": {{
            "target": "급속충전기",
            "metric": "개수",
            "conditions": "전체 충전사업자 (엑셀 O3)"
        }},
        "step2_column_mapping": {{
            "mapped_column": "total_fast_chargers"
        }}
    }},
    "needs_chart": false,
    "show_table": true,
    "output_format": "table",
    "data_filter": {{
        "cpo_name": "전체",
        "display_column": "total_fast_chargers"
    }}
}}
```

### 예시 9: "시장점유율 top 5를 원형그래프로 그려줘" (기타 항목 없음 - 기본값!)
```json
{{
    "reasoning": {{
        "step1_extraction": {{
            "target": "CPO",
            "metric": "시장점유율",
            "conditions": "top 5, 기타 언급 없음",
            "output_format": "chart"
        }}
    }},
    "needs_chart": true,
    "show_table": true,
    "output_format": "chart",
    "chart_type": "pie",
    "chart_title": "시장점유율 Top 5 CPO",
    "analysis_type": "ranking",
    "data_filter": {{
        "sort_column": "시장점유율",
        "display_column": "시장점유율",
        "limit": 5,
        "sort_order": "desc",
        "include_others": false
    }},
    "chart_config": {{
        "x_axis": "CPO명",
        "y_axis": "시장점유율",
        "y_axis_type": "percentage",
        "y_axis_label": "시장점유율 (%)"
    }}
}}
```

### 예시 10: "시장점유율 top 5를 원형그래프로 그려줘, others로 나머지 표시" (기타 항목 + 영어 라벨)
```json
{{
    "reasoning": {{
        "step1_extraction": {{
            "target": "CPO",
            "metric": "시장점유율",
            "conditions": "top 5, 나머지는 others로 표시",
            "output_format": "chart"
        }}
    }},
    "needs_chart": true,
    "show_table": true,
    "output_format": "chart",
    "chart_type": "pie",
    "chart_title": "시장점유율 Top 5 + Others",
    "analysis_type": "ranking",
    "data_filter": {{
        "sort_column": "시장점유율",
        "display_column": "시장점유율",
        "limit": 5,
        "sort_order": "desc",
        "include_others": true,
        "others_label": "others"
    }},
    "chart_config": {{
        "x_axis": "CPO명",
        "y_axis": "시장점유율",
        "y_axis_type": "percentage",
        "y_axis_label": "시장점유율 (%)"
    }}
}}
```

### 예시 11: "시장점유율 top 3를 파이차트로, 기타 포함" (기타 항목 + 한국어 기본값)
```json
{{
    "reasoning": {{
        "step1_extraction": {{
            "target": "CPO",
            "metric": "시장점유율",
            "conditions": "top 3, 기타 포함",
            "output_format": "chart"
        }}
    }},
    "needs_chart": true,
    "show_table": true,
    "output_format": "chart",
    "chart_type": "pie",
    "chart_title": "시장점유율 Top 3 + 기타",
    "analysis_type": "ranking",
    "data_filter": {{
        "sort_column": "시장점유율",
        "display_column": "시장점유율",
        "limit": 3,
        "sort_order": "desc",
        "include_others": true,
        "others_label": "기타"
    }},
    "chart_config": {{
        "x_axis": "CPO명",
        "y_axis": "시장점유율",
        "y_axis_type": "percentage",
        "y_axis_label": "시장점유율 (%)"
    }}
}}
```

JSON만 출력하세요.
"""
        
        try:
            payload = {
                'anthropic_version': Config.ANTHROPIC_VERSION,
                'max_tokens': 2048,  # Chain of Thought 응답을 위해 증가
                'temperature': 0.1,
                'messages': [{'role': 'user', 'content': analysis_prompt}]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=Config.MODEL_ID,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response['body'].read())
            result_text = response_body['content'][0]['text']
            
            # JSON 추출
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                return json.loads(json_match.group())
            
            return {'needs_chart': False, 'analysis_type': 'single'}
            
        except Exception as e:
            print(f'❌ 질의 분석 오류: {e}')
            return {'needs_chart': False, 'analysis_type': 'single'}
    
    def _calculate_y_values(self, top_df, col, y_axis_type, full_df, calculation_info=None):
        """y축 값 계산 (절대값, 점유율, 또는 증가률)"""
        
        def to_python_type(val):
            """numpy 타입을 Python 기본 타입으로 변환"""
            import numpy as np
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            return val
        
        # 증가률 계산이 필요한 경우
        if y_axis_type == 'calculated_rate' and calculation_info:
            calc_type = calculation_info.get('type')
            base_col = calculation_info.get('base_column')
            change_col = calculation_info.get('change_column')
            
            if calc_type == 'growth_rate' and base_col in top_df.columns and change_col in top_df.columns:
                # 증가률 = (증감량 / (현재값 - 증감량)) * 100
                # 현재값 - 증감량 = 이전 월 값
                values = []
                for idx, row in top_df.iterrows():
                    current_val = row[base_col]
                    change_val = row[change_col]
                    prev_val = current_val - change_val
                    
                    if prev_val > 0:
                        rate = (change_val / prev_val) * 100
                        values.append(round(float(rate), 2))
                    else:
                        # 이전 값이 0이면 증가률 계산 불가 (무한대 방지)
                        values.append(0.0 if change_val == 0 else 100.0)
                
                print(f'      ├─ 증가률 계산: {change_col}/{base_col}-{change_col}*100', flush=True)
                return values
        
        # 점유율 계산
        if y_axis_type == 'percentage':
            total = full_df[col].sum()
            if total > 0:
                values = [(to_python_type(v) / float(total) * 100) for v in top_df[col].tolist()]
                print(f'      ├─ 점유율 계산: 전체 합계 {total:,}, 점유율로 변환', flush=True)
                return [round(v, 2) for v in values]
            return [to_python_type(v) for v in top_df[col].tolist()]
        
        # 절대값 그대로 반환 (numpy 타입 변환)
        return [to_python_type(v) for v in top_df[col].tolist()]
    
    def _validate_column_exists(self, col: str, df) -> tuple:
        """컬럼 존재 여부 확인 및 유사 컬럼 추천"""
        if col in df.columns:
            return True, col, None
        
        # 유사 컬럼 찾기
        similar_cols = []
        col_lower = col.lower()
        for c in df.columns:
            c_lower = c.lower()
            # 부분 문자열 매칭
            if col_lower in c_lower or c_lower in col_lower:
                similar_cols.append(c)
            # 키워드 매칭
            keywords = ['충전', '증감', '점유', '순위']
            for kw in keywords:
                if kw in col_lower and kw in c_lower:
                    if c not in similar_cols:
                        similar_cols.append(c)
        
        return False, None, similar_cols
    
    def extract_chart_data(self, df, intent: dict) -> dict:
        """DataFrame에서 차트 데이터 추출 (Text-to-SQL 방식)"""
        try:
            import numpy as np
            
            # numpy 타입을 Python 기본 타입으로 변환하는 헬퍼 함수
            def to_python_type(val):
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    return int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    return float(val)
                return val
            
            def convert_values_list(values):
                """리스트 내 모든 값을 Python 기본 타입으로 변환"""
                return [to_python_type(v) for v in values]
            
            data_filter = intent.get('data_filter', {})
            chart_config = intent.get('chart_config', {})
            
            cpo_name = data_filter.get('cpo_name')
            start_month = data_filter.get('start_month')
            end_month = data_filter.get('end_month')
            
            # Text-to-SQL: sort_column과 display_column 분리
            sort_column = data_filter.get('sort_column')  # ORDER BY 컬럼
            display_column = data_filter.get('display_column')  # SELECT 컬럼 (y축 값)
            column = data_filter.get('column', '총충전기')  # 기존 호환성
            
            # sort_column이 없으면 column 사용 (기존 호환성)
            if not sort_column:
                sort_column = column
            if not display_column:
                display_column = column
            
            limit = data_filter.get('limit')
            sort_order = data_filter.get('sort_order', 'desc')
            
            # 차트 축 설정
            y_axis_type = chart_config.get('y_axis_type', 'value')
            y_axis_label = chart_config.get('y_axis_label', '값')
            x_axis = chart_config.get('x_axis', 'CPO명')
            
            # limit이 None이면 기본값 10, 명시되면 해당 값 사용
            result_limit = limit if limit is not None else 10
            
            print(f'      ├─ 정렬 컬럼 (ORDER BY): {sort_column}', flush=True)
            print(f'      ├─ 표시 컬럼 (SELECT): {display_column}', flush=True)
            print(f'      ├─ 개수 제한 (LIMIT): {limit} → 적용값: {result_limit}', flush=True)
            print(f'      ├─ 정렬 순서: {sort_order}', flush=True)
            print(f'      ├─ Y축 타입: {y_axis_type} ({y_axis_label})', flush=True)
            print(f'      └─ X축: {x_axis}', flush=True)
            
            # 컬럼명 정규화 함수
            def normalize_column(col):
                if col is None:
                    return '총충전기'
                col_str = str(col)
                
                # 영어 컬럼명 → 한국어 DataFrame 컬럼명 매핑 (엑셀 요약 행 컬럼)
                english_to_korean_col = {
                    # 전체 현황 (L3:P3) - DataFrame 컬럼명으로 변환
                    'total_cpos': 'total_cpos',  # 엑셀 전용 (DataFrame에 없음)
                    'total_stations': '충전소수',
                    'total_slow_chargers': '완속충전기',
                    'total_fast_chargers': '급속충전기',
                    'total_chargers': '총충전기',
                    # 당월 증감량 (L4:P4) - DataFrame 컬럼명으로 변환
                    'change_cpos': 'change_cpos',  # 엑셀 전용 (DataFrame에 없음)
                    'change_stations': '충전소증감',
                    'change_slow_chargers': '완속증감',
                    'change_fast_chargers': '급속증감',
                    'change_total_chargers': '총증감',
                }
                
                # 영어 컬럼명이면 한국어로 변환
                if col_str in english_to_korean_col:
                    return english_to_korean_col[col_str]
                
                # 정확한 매칭 우선 (더 구체적인 키를 먼저 체크)
                exact_mapping = {
                    '완속증감': '완속증감',
                    '급속증감': '급속증감',
                    '총증감': '총증감',
                    '충전소증감': '충전소증감',
                    '완속충전기': '완속충전기',
                    '급속충전기': '급속충전기',
                    '총충전기': '총충전기',
                    '충전소수': '충전소수',
                    '시장점유율': '시장점유율'
                }
                
                # 정확히 일치하면 바로 반환
                if col_str in exact_mapping:
                    return exact_mapping[col_str]
                
                # 부분 매칭 (증감 키워드 먼저 체크 - 순서 중요!)
                partial_mapping = [
                    ('완속증감', '완속증감'),
                    ('급속증감', '급속증감'),
                    ('총증감', '총증감'),
                    ('충전소증감', '충전소증감'),
                    ('완속', '완속충전기'),
                    ('급속', '급속충전기'),
                    ('총', '총충전기'),
                    ('충전소', '충전소수'),
                    ('점유율', '시장점유율')
                ]
                
                for key, val in partial_mapping:
                    if key in col_str:
                        return val
                return col
            
            # 시장점유율 값 변환 함수 (소수점 → 퍼센트) + numpy 타입 변환
            def convert_market_share(col_name, values):
                """시장점유율 컬럼인 경우 소수점을 퍼센트로 변환하고 numpy 타입을 Python 타입으로 변환"""
                converted = []
                for v in values:
                    # numpy 타입을 Python 타입으로 변환
                    v = to_python_type(v)
                    if col_name == '시장점유율' and v is not None and v < 1:
                        # 값이 1 미만이면 소수점 형태이므로 100을 곱해 퍼센트로 변환
                        converted.append(round(float(v) * 100, 2))
                    else:
                        converted.append(v)
                return converted
            
            # 영어 컬럼명 → 한국어 라벨 변환 함수
            def to_korean_label(col_name):
                """영어 컬럼명을 한국어 라벨로 변환"""
                korean_mapping = {
                    # 전체 현황 (L3:P3)
                    'total_cpos': 'CPO 개수',
                    'total_stations': '충전소 개수',
                    'total_slow_chargers': '완속충전기 개수',
                    'total_fast_chargers': '급속충전기 개수',
                    'total_chargers': '전체충전기 개수',
                    # 당월 증감량 (L4:P4)
                    'change_cpos': 'CPO 증감량',
                    'change_stations': '충전소 증감량',
                    'change_slow_chargers': '완속충전기 증감량',
                    'change_fast_chargers': '급속충전기 증감량',
                    'change_total_chargers': '전체충전기 증감량',
                    # 기존 한국어 컬럼명 (그대로 유지)
                    '완속증감': '완속충전기 증감량',
                    '급속증감': '급속충전기 증감량',
                    '총증감': '전체충전기 증감량',
                    '충전소증감': '충전소 증감량',
                    '완속충전기': '완속충전기 개수',
                    '급속충전기': '급속충전기 개수',
                    '총충전기': '전체충전기 개수',
                    '충전소수': '충전소 개수',
                    '시장점유율': '시장점유율',
                    '순위': '순위',
                    '순위변동': '순위 변동',
                }
                return korean_mapping.get(col_name, col_name)
            
            # 다중 컬럼 지원 (리스트 또는 쉼표 구분 문자열)
            columns = []
            
            # display_column이 리스트인 경우
            if isinstance(display_column, list):
                columns = [normalize_column(c) for c in display_column]
                print(f'      ├─ 🔀 다중 컬럼 감지 (리스트): {display_column} → {columns}', flush=True)
            # display_column이 쉼표로 구분된 다중 컬럼인지 확인
            elif display_column and ',' in str(display_column):
                # 쉼표로 구분된 다중 컬럼 파싱
                multi_cols = [c.strip() for c in str(display_column).split(',')]
                columns = [normalize_column(c) for c in multi_cols]
                print(f'      ├─ 🔀 다중 컬럼 감지: {multi_cols} → {columns}', flush=True)
            # display_column이 단일 값인 경우 (수정: column 대신 display_column 사용)
            elif display_column:
                columns = [normalize_column(display_column)]
            elif isinstance(column, list):
                columns = [normalize_column(c) for c in column]
            else:
                columns = [normalize_column(column)]
            
            # 데이터 필터링
            filtered_df = df.copy()
            
            # CPO 필터 (단일 또는 다중 CPO 지원)
            cpo_list = []
            has_total_cpo = False  # '전체' CPO 요청 여부
            if cpo_name:
                if isinstance(cpo_name, list):
                    cpo_list = cpo_name
                else:
                    cpo_list = [cpo_name]
                # '전체' 키워드 확인
                total_keywords = ['전체', '전체cpo', '전체 cpo', 'all', 'total']
                has_total_cpo = any(kw in [c.lower() for c in cpo_list] for kw in total_keywords)
                # '전체'를 제외한 실제 CPO 목록
                actual_cpo_list = [c for c in cpo_list if c.lower() not in total_keywords]
            else:
                actual_cpo_list = []
            
            if actual_cpo_list and 'CPO명' in filtered_df.columns:
                # 다중 CPO 필터링 (전체 제외) - 띄어쓰기 무시
                def normalize_cpo_name(name):
                    """CPO명 정규화: 띄어쓰기 제거, 소문자 변환"""
                    return str(name).replace(' ', '').replace('\u3000', '').lower()
                
                mask = filtered_df['CPO명'].apply(
                    lambda x: any(normalize_cpo_name(cpo) in normalize_cpo_name(x) for cpo in actual_cpo_list) if pd.notna(x) else False
                )
                filtered_df = filtered_df[mask]
                print(f'      ├─ CPO 필터 (다중): {actual_cpo_list}', flush=True)
            
            if has_total_cpo:
                print(f'      ├─ 📊 전체 CPO 합계 요청 감지', flush=True)
            
            # 기간 필터
            if 'snapshot_month' in filtered_df.columns:
                if start_month:
                    filtered_df = filtered_df[filtered_df['snapshot_month'] >= start_month]
                if end_month:
                    filtered_df = filtered_df[filtered_df['snapshot_month'] <= end_month]
            
            if len(filtered_df) == 0:
                return {'labels': [], 'values': [], 'error': '해당 조건의 데이터가 없습니다'}
            
            # 차트 타입에 따른 데이터 구성
            analysis_type = intent.get('analysis_type', 'trend')
            
            # 단일 컬럼 (다중 컬럼은 각 analysis_type 내부에서 처리)
            col = columns[0]
            
            if analysis_type == 'trend':
                # 시간별 추이
                print(f'      ├─ Trend 분석: 컬럼 수={len(columns)}, 컬럼={columns}', flush=True)
                print(f'      ├─ 필터링 후 데이터: {len(filtered_df)}행', flush=True)
                
                if cpo_name:
                    print(f'      ├─ CPO 필터: {cpo_name}', flush=True)
                    unique_cpos = filtered_df['CPO명'].unique().tolist() if 'CPO명' in filtered_df.columns else []
                    print(f'      ├─ 필터링된 CPO: {unique_cpos[:5]}...', flush=True)
                
                # 전체 CPO 합계가 필요한 경우 (엑셀 L3:P4 범위)
                # "전체", "전체CPO", "전체 CPO", "충전사업자" 등의 키워드 감지
                def is_total_cpo_query(cpo_name_val):
                    if cpo_name_val is None:
                        return False
                    if isinstance(cpo_name_val, list):
                        return any(is_total_cpo_query(c) for c in cpo_name_val)
                    cpo_lower = str(cpo_name_val).lower().replace(' ', '')
                    total_keywords = ['전체', '전체cpo', '충전사업자', 'total', 'all']
                    return any(kw in cpo_lower for kw in total_keywords)
                
                is_total_query = is_total_cpo_query(cpo_name)
                is_total_with_cpo_query = has_total_cpo and actual_cpo_list  # 전체 + 특정 CPO 비교 요청
                
                # 전체 CPO 관련 컬럼 매핑 (엑셀 셀 위치 기반)
                total_column_mapping = {
                    # 전체 현황 (L3:P3)
                    'total_cpos': ('total', 'cpos'),           # L3
                    'total_stations': ('total', 'stations'),   # M3
                    'total_slow_chargers': ('total', 'slow_chargers'),  # N3
                    'total_fast_chargers': ('total', 'fast_chargers'),  # O3
                    'total_chargers': ('total', 'total_chargers'),      # P3
                    # 당월 증감량 (L4:P4)
                    'change_cpos': ('change', 'cpos'),         # L4
                    'change_stations': ('change', 'stations'), # M4
                    'change_slow_chargers': ('change', 'slow_chargers'),  # N4
                    'change_fast_chargers': ('change', 'fast_chargers'),  # O4
                    'change_total_chargers': ('change', 'total_chargers'), # P4
                    # 기존 컬럼명 호환
                    '완속증감': ('change', 'slow_chargers'),
                    '급속증감': ('change', 'fast_chargers'),
                    '총증감': ('change', 'total_chargers'),
                    '충전소증감': ('change', 'stations'),
                }
                
                # 전체 CPO 관련 컬럼인지 확인
                is_total_column = any(c in total_column_mapping for c in columns)
                
                if is_total_query and is_total_column:
                    print(f'      ├─ 📊 전체 CPO 합계 조회 - 엑셀 요약 행(L3:P4)에서 직접 추출', flush=True)
                    print(f'      ├─ 요청 컬럼: {columns}', flush=True)
                    
                    # 엑셀 파일에서 직접 합계 데이터 추출
                    from data_loader import ChargingDataLoader
                    loader = ChargingDataLoader()
                    
                    # 월별 합계 데이터 수집
                    monthly_totals = {}
                    files = loader.list_available_files()
                    
                    for file_info in files:
                        s3_key = file_info['key']
                        filename = file_info['filename']
                        _, snapshot_month = loader.parse_snapshot_date_from_filename(filename)
                        
                        if snapshot_month and (not start_month or snapshot_month >= start_month) and (not end_month or snapshot_month <= end_month):
                            summary = loader.extract_summary_data(s3_key)
                            if summary:
                                monthly_totals[snapshot_month] = {
                                    # 전체 현황 (L3:P3)
                                    'total_cpos': summary.get('total', {}).get('cpos', 0),
                                    'total_stations': summary.get('total', {}).get('stations', 0),
                                    'total_slow_chargers': summary.get('total', {}).get('slow_chargers', 0),
                                    'total_fast_chargers': summary.get('total', {}).get('fast_chargers', 0),
                                    'total_chargers': summary.get('total', {}).get('total_chargers', 0),
                                    # 당월 증감량 (L4:P4)
                                    'change_cpos': summary.get('change', {}).get('cpos', 0),
                                    'change_stations': summary.get('change', {}).get('stations', 0),
                                    'change_slow_chargers': summary.get('change', {}).get('slow_chargers', 0),
                                    'change_fast_chargers': summary.get('change', {}).get('fast_chargers', 0),
                                    'change_total_chargers': summary.get('change', {}).get('total_chargers', 0),
                                    # 기존 컬럼명 호환
                                    '완속증감': summary.get('change', {}).get('slow_chargers', 0),
                                    '급속증감': summary.get('change', {}).get('fast_chargers', 0),
                                    '총증감': summary.get('change', {}).get('total_chargers', 0),
                                    '충전소증감': summary.get('change', {}).get('stations', 0),
                                }
                                print(f'      ├─ {snapshot_month}: 추출 완료', flush=True)
                    
                    if monthly_totals:
                        sorted_months = sorted(monthly_totals.keys())
                        
                        # 전체 CPO + 특정 CPO 비교인 경우
                        if is_total_with_cpo_query and actual_cpo_list:
                            result = {'labels': sorted_months, 'series': [], 'multi_series': True}
                            
                            # 1. 전체 CPO 시리즈 추가
                            for target_col in columns:
                                values = [monthly_totals.get(m, {}).get(target_col, 0) for m in sorted_months]
                                korean_label = to_korean_label(target_col)
                                result['series'].append({'name': f'전체 {korean_label}', 'values': values})
                                print(f'      ├─ 시리즈 추가 (전체 CPO): 전체 {korean_label} = {values[:3]}...', flush=True)
                            
                            # 2. 특정 CPO 시리즈 추가
                            for cpo in actual_cpo_list:
                                # 해당 CPO 데이터 필터링
                                cpo_mask = df['CPO명'].apply(
                                    lambda x: cpo.lower() in str(x).lower() if pd.notna(x) else False
                                )
                                cpo_df = df[cpo_mask]
                                
                                # 기간 필터
                                if 'snapshot_month' in cpo_df.columns:
                                    if start_month:
                                        cpo_df = cpo_df[cpo_df['snapshot_month'] >= start_month]
                                    if end_month:
                                        cpo_df = cpo_df[cpo_df['snapshot_month'] <= end_month]
                                
                                for target_col in columns:
                                    if target_col in cpo_df.columns:
                                        grouped = cpo_df.groupby('snapshot_month')[target_col].first().reset_index()
                                        grouped = grouped.sort_values('snapshot_month')
                                        
                                        # sorted_months에 맞춰 값 정렬
                                        values = []
                                        for m in sorted_months:
                                            month_val = grouped[grouped['snapshot_month'] == m][target_col].values
                                            values.append(float(month_val[0]) if len(month_val) > 0 else 0)
                                        
                                        korean_label = to_korean_label(target_col)
                                        result['series'].append({'name': f'{cpo} {korean_label}', 'values': values})
                                        print(f'      ├─ 시리즈 추가 ({cpo}): {cpo} {korean_label} = {values[:3]}...', flush=True)
                            
                            result['y_axis_label'] = chart_config.get('y_axis_label', '값')
                            print(f'      └─ 전체+CPO 비교 완료: {len(result["series"])}개 시리즈', flush=True)
                            return result
                        
                        # 전체 CPO만 요청한 경우 (기존 로직)
                        # 다중 컬럼인 경우
                        if len(columns) > 1:
                            result = {'labels': sorted_months, 'series': [], 'multi_series': True}
                            for target_col in columns:
                                values = [monthly_totals.get(m, {}).get(target_col, 0) for m in sorted_months]
                                korean_label = to_korean_label(target_col)
                                result['series'].append({'name': korean_label, 'values': values})
                                print(f'      ├─ 시리즈 추가 (엑셀 합계): {korean_label} = {values[:3]}...', flush=True)
                            result['y_axis_label'] = chart_config.get('y_axis_label', '값')
                            return result
                        else:
                            # 단일 컬럼
                            values = [monthly_totals.get(m, {}).get(col, 0) for m in sorted_months]
                            print(f'      └─ 추출된 값 (엑셀 합계): {values[:5]}...', flush=True)
                            korean_label = to_korean_label(col)
                            return {
                                'labels': sorted_months,
                                'values': values,
                                'y_axis_label': chart_config.get('y_axis_label', korean_label)
                            }
                
                # 다중 CPO + 다중 컬럼 조합 처리
                unique_cpos = filtered_df['CPO명'].unique().tolist() if 'CPO명' in filtered_df.columns else []
                is_multi_cpo = len(unique_cpos) > 1
                is_multi_col = len(columns) > 1
                
                if (is_multi_cpo or is_multi_col) and 'snapshot_month' in filtered_df.columns:
                    print(f'      ├─ 🔀 다중 시리즈 차트 생성: CPO={unique_cpos}, 컬럼={columns}', flush=True)
                    result = {'labels': [], 'series': [], 'multi_series': True}
                    
                    # 다중 CPO + 다중 컬럼: CPO별 컬럼별 시리즈 생성
                    if is_multi_cpo and is_multi_col:
                        for cpo in unique_cpos:
                            cpo_df = filtered_df[filtered_df['CPO명'] == cpo]
                            for target_col in columns:
                                if target_col in cpo_df.columns:
                                    grouped = cpo_df.groupby('snapshot_month')[target_col].first().reset_index()
                                    grouped = grouped.sort_values('snapshot_month')
                                    
                                    if not result['labels']:
                                        result['labels'] = grouped['snapshot_month'].tolist()
                                    
                                    series_name = f'{cpo}_{target_col}'
                                    # 시장점유율 변환 적용
                                    values = convert_market_share(target_col, grouped[target_col].tolist())
                                    korean_label = to_korean_label(target_col)
                                    series_name_kr = f'{cpo} {korean_label}'
                                    result['series'].append({
                                        'name': series_name_kr,
                                        'values': values
                                    })
                                    print(f'      ├─ 시리즈 추가: {series_name_kr} = {values[:3]}...', flush=True)
                    
                    # 다중 CPO + 단일 컬럼: CPO별 시리즈 생성
                    elif is_multi_cpo:
                        target_col = columns[0]
                        korean_label = to_korean_label(target_col)
                        for cpo in unique_cpos:
                            cpo_df = filtered_df[filtered_df['CPO명'] == cpo]
                            if target_col in cpo_df.columns:
                                grouped = cpo_df.groupby('snapshot_month')[target_col].first().reset_index()
                                grouped = grouped.sort_values('snapshot_month')
                                
                                if not result['labels']:
                                    result['labels'] = grouped['snapshot_month'].tolist()
                                
                                # 시장점유율 변환 적용
                                values = convert_market_share(target_col, grouped[target_col].tolist())
                                result['series'].append({
                                    'name': f'{cpo} {korean_label}',
                                    'values': values
                                })
                                print(f'      ├─ 시리즈 추가: {cpo} {korean_label} = {values[:3]}...', flush=True)
                    
                    # 단일 CPO + 다중 컬럼: 컬럼별 시리즈 생성
                    else:
                        for target_col in columns:
                            if target_col in filtered_df.columns:
                                if len(unique_cpos) == 1:
                                    grouped = filtered_df.groupby('snapshot_month')[target_col].first().reset_index()
                                else:
                                    grouped = filtered_df.groupby('snapshot_month')[target_col].sum().reset_index()
                                
                                grouped = grouped.sort_values('snapshot_month')
                                
                                if not result['labels']:
                                    result['labels'] = grouped['snapshot_month'].tolist()
                                
                                # 시장점유율 변환 적용
                                values = convert_market_share(target_col, grouped[target_col].tolist())
                                korean_label = to_korean_label(target_col)
                                result['series'].append({
                                    'name': korean_label,
                                    'values': values
                                })
                                print(f'      ├─ 시리즈 추가: {korean_label} = {values[:3]}...', flush=True)
                    
                    result['y_axis_label'] = chart_config.get('y_axis_label', '값')
                    print(f'      └─ ✅ 다중 시리즈 데이터 추출 완료', flush=True)
                    print(f'         ├─ 시리즈 수: {len(result["series"])}개', flush=True)
                    print(f'         ├─ 데이터 포인트: {len(result["labels"])}개', flush=True)
                    for s in result['series']:
                        print(f'         ├─ {s["name"]}: {s["values"][:3]}...', flush=True)
                    return result
                
                # 단일 컬럼인 경우
                target_col = col
                if 'snapshot_month' in filtered_df.columns and target_col in filtered_df.columns:
                    if cpo_name and 'CPO명' in filtered_df.columns and len(filtered_df['CPO명'].unique()) == 1:
                        grouped = filtered_df.groupby('snapshot_month')[target_col].first().reset_index()
                    else:
                        grouped = filtered_df.groupby('snapshot_month')[target_col].sum().reset_index()
                    
                    grouped = grouped.sort_values('snapshot_month')
                    # 시장점유율 변환 적용
                    values = convert_market_share(target_col, grouped[target_col].tolist())
                    print(f'      └─ 추출된 값: {values[:5]}...', flush=True)
                    korean_label = to_korean_label(target_col)
                    return {
                        'labels': grouped['snapshot_month'].tolist(),
                        'values': values,
                        'y_axis_label': chart_config.get('y_axis_label', korean_label)
                    }
            
            elif analysis_type == 'comparison':
                # 전체 CPO + 특정 CPO 비교인 경우 (증감 컬럼)
                # 한국어 및 영어 컬럼명 모두 체크
                change_columns_kr = ['완속증감', '급속증감', '총증감', '충전소증감']
                change_columns_en = ['change_slow_chargers', 'change_fast_chargers', 'change_total_chargers', 'change_stations']
                is_change_column = any(c in change_columns_kr or c in change_columns_en for c in columns)
                
                if has_total_cpo and (is_change_column or actual_cpo_list):
                    print(f'      ├─ 📊 전체 CPO + 특정 CPO 비교 (comparison)', flush=True)
                    print(f'      ├─ 원본 컬럼: {columns}', flush=True)
                    
                    # 컬럼명 정규화 (영어 → 한국어)
                    normalized_columns = [normalize_column(c) for c in columns]
                    print(f'      ├─ 정규화된 컬럼: {normalized_columns}', flush=True)
                    
                    # 엑셀 파일에서 전체 합계 데이터 추출
                    from data_loader import ChargingDataLoader
                    loader = ChargingDataLoader()
                    
                    monthly_totals = {}
                    files = loader.list_available_files()
                    
                    for file_info in files:
                        s3_key = file_info['key']
                        filename = file_info['filename']
                        _, snapshot_month = loader.parse_snapshot_date_from_filename(filename)
                        
                        if snapshot_month and (not start_month or snapshot_month >= start_month) and (not end_month or snapshot_month <= end_month):
                            summary = loader.extract_summary_data(s3_key)
                            if summary:
                                monthly_totals[snapshot_month] = {
                                    # 증감량 (change)
                                    '완속증감': summary.get('change', {}).get('slow_chargers', 0),
                                    '급속증감': summary.get('change', {}).get('fast_chargers', 0),
                                    '총증감': summary.get('change', {}).get('total_chargers', 0),
                                    '충전소증감': summary.get('change', {}).get('stations', 0),
                                    # 전체 현황 (total)
                                    '완속충전기': summary.get('total', {}).get('slow_chargers', 0),
                                    '급속충전기': summary.get('total', {}).get('fast_chargers', 0),
                                    '총충전기': summary.get('total', {}).get('total_chargers', 0),
                                    '충전소수': summary.get('total', {}).get('stations', 0),
                                }
                    
                    if monthly_totals:
                        sorted_months = sorted(monthly_totals.keys())
                        result = {'labels': sorted_months, 'series': [], 'multi_series': True}
                        
                        # 1. 전체 CPO 시리즈 추가 (정규화된 컬럼명 사용)
                        for i, target_col in enumerate(normalized_columns):
                            original_col = columns[i]  # 원본 컬럼명 (한국어 라벨용)
                            values = [monthly_totals.get(m, {}).get(target_col, 0) for m in sorted_months]
                            korean_label = to_korean_label(original_col)
                            result['series'].append({'name': f'전체 {korean_label}', 'values': values})
                            print(f'      ├─ 시리즈 추가 (전체 CPO): 전체 {korean_label} = {values[:3]}...', flush=True)
                        
                        # 2. 특정 CPO 시리즈 추가
                        for cpo in actual_cpo_list:
                            cpo_mask = df['CPO명'].apply(
                                lambda x: cpo.lower() in str(x).lower() if pd.notna(x) else False
                            )
                            cpo_df = df[cpo_mask]
                            
                            if 'snapshot_month' in cpo_df.columns:
                                if start_month:
                                    cpo_df = cpo_df[cpo_df['snapshot_month'] >= start_month]
                                if end_month:
                                    cpo_df = cpo_df[cpo_df['snapshot_month'] <= end_month]
                            
                            for i, target_col in enumerate(normalized_columns):
                                original_col = columns[i]
                                if target_col in cpo_df.columns:
                                    grouped = cpo_df.groupby('snapshot_month')[target_col].first().reset_index()
                                    grouped = grouped.sort_values('snapshot_month')
                                    
                                    values = []
                                    for m in sorted_months:
                                        month_val = grouped[grouped['snapshot_month'] == m][target_col].values
                                        values.append(float(month_val[0]) if len(month_val) > 0 else 0)
                                    
                                    korean_label = to_korean_label(original_col)
                                    result['series'].append({'name': f'{cpo} {korean_label}', 'values': values})
                                    print(f'      ├─ 시리즈 추가 ({cpo}): {cpo} {korean_label} = {values[:3]}...', flush=True)
                        
                        result['y_axis_label'] = chart_config.get('y_axis_label', '값')
                        print(f'      └─ 전체+CPO 비교 완료: {len(result["series"])}개 시리즈', flush=True)
                        return result
                
                # 다중 CPO + 다중 컬럼 시계열 비교인 경우 (trend와 유사하게 처리)
                unique_cpos = filtered_df['CPO명'].unique().tolist() if 'CPO명' in filtered_df.columns else []
                is_multi_cpo = len(unique_cpos) > 1
                is_multi_col = len(columns) > 1
                
                # 시계열 비교가 필요한 경우 (다중 CPO 또는 다중 컬럼 + 기간 필터)
                if (is_multi_cpo or is_multi_col) and 'snapshot_month' in filtered_df.columns and start_month and end_month:
                    print(f'      ├─ 🔀 시계열 비교 차트 생성: CPO={unique_cpos}, 컬럼={columns}', flush=True)
                    result = {'labels': [], 'series': [], 'multi_series': True}
                    
                    # 다중 CPO + 다중 컬럼: CPO별 컬럼별 시리즈 생성
                    if is_multi_cpo and is_multi_col:
                        for cpo in unique_cpos:
                            cpo_df = filtered_df[filtered_df['CPO명'] == cpo]
                            for target_col in columns:
                                if target_col in cpo_df.columns:
                                    grouped = cpo_df.groupby('snapshot_month')[target_col].first().reset_index()
                                    grouped = grouped.sort_values('snapshot_month')
                                    
                                    if not result['labels']:
                                        result['labels'] = grouped['snapshot_month'].tolist()
                                    
                                    korean_label = to_korean_label(target_col)
                                    series_name_kr = f'{cpo} {korean_label}'
                                    # 시장점유율 변환 적용
                                    values = convert_market_share(target_col, grouped[target_col].tolist())
                                    result['series'].append({
                                        'name': series_name_kr,
                                        'values': values
                                    })
                                    print(f'      ├─ 시리즈 추가: {series_name_kr} = {values[:3]}...', flush=True)
                    
                    # 다중 CPO + 단일 컬럼
                    elif is_multi_cpo:
                        target_col = columns[0]
                        korean_label = to_korean_label(target_col)
                        for cpo in unique_cpos:
                            cpo_df = filtered_df[filtered_df['CPO명'] == cpo]
                            if target_col in cpo_df.columns:
                                grouped = cpo_df.groupby('snapshot_month')[target_col].first().reset_index()
                                grouped = grouped.sort_values('snapshot_month')
                                
                                if not result['labels']:
                                    result['labels'] = grouped['snapshot_month'].tolist()
                                
                                # 시장점유율 변환 적용
                                values = convert_market_share(target_col, grouped[target_col].tolist())
                                result['series'].append({
                                    'name': f'{cpo} {korean_label}',
                                    'values': values
                                })
                                print(f'      ├─ 시리즈 추가: {cpo} {korean_label} = {values[:3]}...', flush=True)
                    
                    # 단일 CPO + 다중 컬럼
                    else:
                        for target_col in columns:
                            if target_col in filtered_df.columns:
                                grouped = filtered_df.groupby('snapshot_month')[target_col].first().reset_index()
                                grouped = grouped.sort_values('snapshot_month')
                                
                                if not result['labels']:
                                    result['labels'] = grouped['snapshot_month'].tolist()
                                
                                # 시장점유율 변환 적용
                                values = convert_market_share(target_col, grouped[target_col].tolist())
                                korean_label = to_korean_label(target_col)
                                result['series'].append({
                                    'name': korean_label,
                                    'values': values
                                })
                                print(f'      ├─ 시리즈 추가: {korean_label} = {values[:3]}...', flush=True)
                    
                    result['y_axis_label'] = chart_config.get('y_axis_label', '값')
                    print(f'      └─ 시계열 비교 완료: {len(result["series"])}개 시리즈', flush=True)
                    return result
                
                # 기존 comparison 로직 (단일 시점 비교)
                sort_col = normalize_column(sort_column) if sort_column else col
                # display_column이 리스트인 경우 첫 번째 컬럼 사용
                if isinstance(display_column, list):
                    display_col = normalize_column(display_column[0])
                else:
                    display_col = normalize_column(display_column) if display_column else col
                
                # 계산이 필요한 경우 (증가률 등)
                calculation_info = intent.get('calculation_required', {})
                needs_calculation = calculation_info.get('needed', False)
                
                if needs_calculation:
                    calc_type = calculation_info.get('type')
                    base_col = calculation_info.get('base_column')
                    change_col = calculation_info.get('change_column')
                    
                    base_col = normalize_column(base_col) if base_col else None
                    change_col = normalize_column(change_col) if change_col else None
                    
                    if calc_type == 'growth_rate' and base_col and change_col:
                        sort_col = change_col
                        display_col = base_col
                        y_axis_type = 'calculated_rate'
                        calculation_info['base_column'] = base_col
                        calculation_info['change_column'] = change_col
                
                if 'CPO명' in filtered_df.columns and sort_col in filtered_df.columns:
                    latest_month = filtered_df['snapshot_month'].max()
                    latest_df = filtered_df[filtered_df['snapshot_month'] == latest_month]
                    
                    # sort_order가 None이면 기본값 'desc' 사용
                    effective_sort_order = sort_order if sort_order else 'desc'
                    print(f'      ├─ SQL 실행: ORDER BY {sort_col} {effective_sort_order.upper()} LIMIT {result_limit}', flush=True)
                    
                    # 정렬 컬럼 기준으로 상위/하위 추출
                    if effective_sort_order == 'asc':
                        top_df = latest_df.nsmallest(result_limit, sort_col)
                    else:
                        top_df = latest_df.nlargest(result_limit, sort_col)
                    
                    # y축 타입에 따른 값 계산 (display_column 사용)
                    value_col = display_col if display_col in top_df.columns else sort_col
                    calc_info_for_values = calculation_info if needs_calculation else None
                    values = self._calculate_y_values(top_df, value_col, y_axis_type, latest_df, calc_info_for_values)
                    
                    print(f'      └─ 결과: {len(top_df)}개 CPO, 값 컬럼={value_col}, y축타입={y_axis_type}', flush=True)
                    
                    return {
                        'labels': top_df['CPO명'].tolist(),
                        'values': values,
                        'y_axis_type': y_axis_type,
                        'y_axis_label': y_axis_label
                    }
            
            elif analysis_type == 'ranking':
                # 순위 (Text-to-SQL: sort_column으로 정렬, display_column으로 표시)
                sort_col = normalize_column(sort_column) if sort_column else col
                display_col = normalize_column(display_column) if display_column else col
                
                # 계산이 필요한 경우 (증가률 등)
                calculation_info = intent.get('calculation_required', {})
                needs_calculation = calculation_info.get('needed', False)
                
                if needs_calculation:
                    calc_type = calculation_info.get('type')
                    base_col = calculation_info.get('base_column')
                    change_col = calculation_info.get('change_column')
                    
                    # 컬럼 정규화
                    base_col = normalize_column(base_col) if base_col else None
                    change_col = normalize_column(change_col) if change_col else None
                    
                    print(f'      ├─ 계산 필요: {calc_type}', flush=True)
                    print(f'      ├─ 기준 컬럼: {base_col}, 변화 컬럼: {change_col}', flush=True)
                    
                    # 증가률 계산을 위해 sort_col과 display_col 조정
                    if calc_type == 'growth_rate' and base_col and change_col:
                        sort_col = change_col  # 증감량 기준 정렬
                        display_col = base_col  # 기준 컬럼 (계산에 사용)
                        y_axis_type = 'calculated_rate'
                        calculation_info['base_column'] = base_col
                        calculation_info['change_column'] = change_col
                
                if 'CPO명' in filtered_df.columns and sort_col in filtered_df.columns:
                    latest_month = filtered_df['snapshot_month'].max()
                    latest_df = filtered_df[filtered_df['snapshot_month'] == latest_month]
                    
                    # sort_order가 None이면 기본값 'desc' 사용
                    effective_sort_order = sort_order if sort_order else 'desc'
                    print(f'      ├─ SQL 실행: ORDER BY {sort_col} {effective_sort_order.upper()} LIMIT {result_limit}', flush=True)
                    
                    # 정렬 컬럼 기준으로 상위/하위 추출
                    if effective_sort_order == 'asc':
                        top_df = latest_df.nsmallest(result_limit, sort_col)
                    else:
                        top_df = latest_df.nlargest(result_limit, sort_col)
                    
                    # y축 타입에 따른 값 계산 (display_column 사용)
                    value_col = display_col if display_col in top_df.columns else sort_col
                    
                    # 계산 정보 전달
                    calc_info_for_values = calculation_info if needs_calculation else None
                    values = self._calculate_y_values(top_df, value_col, y_axis_type, latest_df, calc_info_for_values)
                    labels = top_df['CPO명'].tolist()
                    
                    # "기타" 항목 추가 (include_others 옵션)
                    include_others = data_filter.get('include_others', False)
                    if include_others and y_axis_type == 'percentage':
                        # Top N을 제외한 나머지 CPO의 점유율 합계
                        top_cpos = set(labels)
                        others_df = latest_df[~latest_df['CPO명'].isin(top_cpos)]
                        if len(others_df) > 0 and value_col in others_df.columns:
                            others_sum = others_df[value_col].sum()
                            # 소수점 형태면 퍼센트로 변환
                            if others_sum < 1:
                                others_sum = others_sum * 100
                            others_sum = round(float(others_sum), 2)
                            # 사용자가 지정한 라벨 사용 (기본값: '기타')
                            others_label = data_filter.get('others_label', '기타')
                            labels.append(others_label)
                            values.append(others_sum)
                            print(f'      ├─ {others_label} 항목 추가: {len(others_df)}개 CPO, 합계 {others_sum}%', flush=True)
                    
                    print(f'      └─ 결과: {len(labels)}개 항목, 값 컬럼={value_col}, y축타입={y_axis_type}', flush=True)
                    
                    return {
                        'labels': labels,
                        'values': values,
                        'y_axis_type': y_axis_type,
                        'y_axis_label': y_axis_label
                    }
            
            # 기본: 시간별 추이
            if 'snapshot_month' in filtered_df.columns and col in filtered_df.columns:
                grouped = filtered_df.groupby('snapshot_month')[col].sum().reset_index()
                grouped = grouped.sort_values('snapshot_month')
                return {
                    'labels': grouped['snapshot_month'].tolist(),
                    'values': convert_values_list(grouped[col].tolist())
                }
            
            return {'labels': [], 'values': [], 'error': '데이터 추출 실패'}
            
        except Exception as e:
            print(f'❌ 데이터 추출 오류: {e}')
            return {'labels': [], 'values': [], 'error': str(e)}
    
    def generate_table_answer(self, query: str, df, kb_context: str, intent: dict,
                               table_data: dict, show_table: bool = True) -> tuple:
        """표 기반 답변 생성 (시각화 라이브러리 사용 안함)"""
        import time
        
        labels = table_data.get('labels', [])
        values = table_data.get('values', [])
        series = table_data.get('series', [])
        is_multi_series = table_data.get('multi_series', False)
        y_axis_type = table_data.get('y_axis_type', 'value')
        y_axis_label = table_data.get('y_axis_label', '값')
        chart_title = intent.get('chart_title', '데이터 분석 결과')
        
        table_md = ""
        data_summary = ""
        
        # 다중 시리즈 표 생성
        if show_table and is_multi_series and series:
            # 헤더 생성: 기간 | 컬럼1 | 컬럼2 | ...
            col_names = [s['name'] for s in series]
            header = "| 기간 | " + " | ".join(col_names) + " |"
            separator = "|------|" + "|".join(["------"] * len(col_names)) + "|"
            
            table_rows = []
            for i, label in enumerate(labels):
                row_values = [label]
                for s in series:
                    val = s['values'][i] if i < len(s['values']) else 0
                    if y_axis_type in ['percentage', 'calculated_rate']:
                        row_values.append(f"{val:.1f}%")
                    else:
                        row_values.append(f"{val:,}" if isinstance(val, (int, float)) else str(val))
                table_rows.append("| " + " | ".join(row_values) + " |")
            
            table_md = f"""
## {chart_title}

{header}
{separator}
{chr(10).join(table_rows)}
"""
            # 데이터 요약
            data_summary = f"- 기간: {labels[0]} ~ {labels[-1]}\n- 항목 수: {len(labels)}개\n- 컬럼: {', '.join(col_names)}"
        
        # 단일 시리즈 표 생성
        elif show_table and labels and values:
            if y_axis_type in ['percentage', 'calculated_rate']:
                formatted_values = [f"{v:.1f}%" for v in values]
            else:
                formatted_values = [f"{v:,}" if isinstance(v, (int, float)) else str(v) for v in values]
            
            table_rows = []
            for i, (label, value) in enumerate(zip(labels, formatted_values), 1):
                table_rows.append(f"| {i} | {label} | {value} |")
            
            table_md = f"""
## {chart_title}

| 순위 | 항목 | {y_axis_label} |
|------|------|----------------|
{chr(10).join(table_rows)}
"""
            data_summary = f"- 항목 수: {len(labels)}개\n- 데이터: {list(zip(labels, values))[:5]}..."
        
        # LLM 프롬프트 구성
        prompt = f"""
당신은 전기차 충전 인프라 데이터 분석 전문가입니다.

## 사용자 질문
{query}

## 분석 결과 데이터
{data_summary if data_summary else f"- 항목 수: {len(labels)}개"}

{f"## 표 형식 결과{table_md}" if show_table and table_md else ""}

## Knowledge Base 참고 자료
{kb_context[:1500] if kb_context else '없음'}

## 답변 작성 지침
1. {"위 표를 참고하여 " if show_table else ""}데이터를 분석하세요
2. 주요 특징이나 인사이트를 설명하세요
3. 간결하고 명확하게 답변하세요
4. 시각화(차트/그래프)는 생성하지 마세요 - 표 형식으로만 답변합니다
{"5. 표를 그대로 포함하여 답변하세요" if show_table else "5. 표 없이 텍스트로만 답변하세요"}

한국어로 답변해주세요.
"""
        
        try:
            start_time = time.time()
            
            payload = {
                'anthropic_version': Config.ANTHROPIC_VERSION,
                'max_tokens': 2048,
                'temperature': 0.5,
                'messages': [{'role': 'user', 'content': prompt}]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=Config.MODEL_ID,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response['body'].read())
            result = response_body['content'][0]['text']
            
            elapsed_time = time.time() - start_time
            print(f'✅ Bedrock 응답 완료 (⏱️ {elapsed_time:.2f}초)', flush=True)
            
            return result, elapsed_time
            
        except Exception as e:
            # 오류 시 기본 표 형식 답변 반환
            if show_table:
                return f"데이터 분석 결과입니다.\n{table_md}", 0
            else:
                return f"데이터 분석 결과: {list(zip(labels, values))}", 0
    
    def generate_chart(self, intent: dict, chart_data: dict) -> dict:
        """차트 생성"""
        try:
            chart_type = intent.get('chart_type', 'line')
            chart_title = intent.get('chart_title', '데이터 분석')
            
            # 차트 코드 생성
            code = self.chart_generator.generate_chart_code(
                chart_type=chart_type,
                data=chart_data,
                title=chart_title
            )
            
            # 코드 실행 및 이미지 생성
            result = self.chart_generator.execute_chart_code(code)
            
            return result
            
        except Exception as e:
            print(f'❌ 차트 생성 오류: {e}')
            return {'success': False, 'error': str(e)}
    
    def generate_answer_with_chart(self, query: str, df, kb_context: str, intent: dict, 
                                    chart_data: dict, chart_result: dict) -> tuple:
        """차트와 함께 답변 생성"""
        import time
        
        # 다중 시리즈 여부 확인
        is_multi_series = chart_data.get('multi_series', False)
        
        def format_value(val):
            """값을 안전하게 포맷팅 (숫자면 천단위 구분, 문자열이면 그대로)"""
            if isinstance(val, (int, float)):
                return f"{val:,}"
            return str(val)
        
        if is_multi_series:
            # 다중 시리즈 데이터 요약
            series_info = []
            for s in chart_data.get('series', []):
                values = s.get('values', [])
                if values:
                    # 숫자 값만 필터링하여 min/max 계산
                    numeric_values = [v for v in values if isinstance(v, (int, float))]
                    if numeric_values:
                        min_val = min(numeric_values)
                        max_val = max(numeric_values)
                        series_info.append(f"- {s['name']}: 최소 {format_value(min_val)}, 최대 {format_value(max_val)}")
                    else:
                        series_info.append(f"- {s['name']}: {len(values)}개 항목")
            
            data_summary = f"""
- 조회 기간: {chart_data.get('labels', ['N/A'])[0]} ~ {chart_data.get('labels', ['N/A'])[-1]}
- 데이터 포인트: {len(chart_data.get('labels', []))}개
- 시리즈 수: {len(chart_data.get('series', []))}개
{chr(10).join(series_info)}
"""
            # 다중 시리즈 테이블 생성
            headers = ['기간'] + [s['name'] for s in chart_data.get('series', [])]
            table_header = '| ' + ' | '.join(headers) + ' |'
            table_sep = '|' + '|'.join(['------'] * len(headers)) + '|'
            
            rows = []
            labels = chart_data.get('labels', [])
            series_list = chart_data.get('series', [])
            for i, label in enumerate(labels):
                row_values = [label] + [format_value(s['values'][i]) if i < len(s['values']) else 'N/A' for s in series_list]
                rows.append('| ' + ' | '.join(row_values) + ' |')
            
            detail_table = f"{table_header}\n{table_sep}\n" + '\n'.join(rows)
        else:
            # 단일 시리즈 데이터 요약
            values = chart_data.get('values', [0])
            data_summary = f"""
- 조회 기간: {chart_data.get('labels', ['N/A'])[0]} ~ {chart_data.get('labels', ['N/A'])[-1]}
- 데이터 포인트: {len(values)}개
- 최소값: {min(values) if values else 0:,}
- 최대값: {max(values) if values else 0:,}
- 평균값: {sum(values) / max(len(values), 1):,.0f}
"""
            detail_table = "| 기간 | 값 |\n|------|-----|\n" + '\n'.join([
                f"| {l} | {v:,} |" for l, v in zip(chart_data.get('labels', []), values)
            ])
        
        prompt = f"""
당신은 전기차 충전 인프라 데이터 분석 전문가입니다.

## 사용자 질문
{query}

## 분석 결과
{data_summary}

## 상세 데이터
{detail_table}

## Knowledge Base 참고 자료
{kb_context[:2000] if kb_context else '없음'}

## 답변 작성 지침
1. 차트가 생성되었음을 언급하세요
2. 데이터의 주요 트렌드를 설명하세요
3. 눈에 띄는 변화나 특이점을 분석하세요
4. 간결하고 명확하게 답변하세요

한국어로 답변해주세요.
"""
        
        try:
            start_time = time.time()
            
            payload = {
                'anthropic_version': Config.ANTHROPIC_VERSION,
                'max_tokens': 2048,
                'temperature': 0.5,
                'messages': [{'role': 'user', 'content': prompt}]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=Config.MODEL_ID,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response['body'].read())
            result = response_body['content'][0]['text']
            
            elapsed_time = time.time() - start_time
            print(f'✅ Bedrock 응답 완료 (⏱️ {elapsed_time:.2f}초)', flush=True)
            
            return result, elapsed_time
            
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}", 0
    
    def _log_separator(self, title: str):
        """로그 구분선 출력"""
        print(f'\n{"="*60}', flush=True)
        print(f'🤖 {title}', flush=True)
        print(f'{"="*60}', flush=True)
    
    def _log_step(self, step_num: int, title: str, details: dict = None):
        """단계별 로그 출력"""
        print(f'\n📌 Step {step_num}: {title}', flush=True)
        if details:
            for key, value in details.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f'   └─ {key}: {value[:5]}... (총 {len(value)}개)', flush=True)
                elif isinstance(value, str) and len(value) > 200:
                    print(f'   └─ {key}: {value[:200]}... (총 {len(value)}자)', flush=True)
                else:
                    print(f'   └─ {key}: {value}', flush=True)
    
    def process_query(self, query: str, df, full_df) -> dict:
        """전체 질의 처리 파이프라인"""
        import time
        total_start_time = time.time()  # 전체 처리 시간 측정 시작
        
        self._log_separator(f'Agent 질의 처리 시작')
        print(f'📝 사용자 질의: "{query}"', flush=True)
        
        # ========================================
        # Step 1: 메모리 데이터 수집
        # ========================================
        available_data = {
            'available_months': sorted(full_df['snapshot_month'].unique().tolist()) if 'snapshot_month' in full_df.columns else [],
            'available_cpos': full_df['CPO명'].unique().tolist() if 'CPO명' in full_df.columns else [],
            'available_columns': list(full_df.columns)
        }
        
        self._log_step(1, '메모리 데이터 수집 (S3 캐시)', {
            '전체 데이터 행 수': len(full_df),
            '현재 필터 데이터 행 수': len(df) if df is not None else 0,
            '사용 가능한 월': available_data['available_months'],
            '사용 가능한 CPO 수': len(available_data['available_cpos']),
            '컬럼 목록': available_data['available_columns']
        })
        
        # ========================================
        # Step 2: RAG - Knowledge Base 검색
        # ========================================
        self._log_step(2, 'RAG - Knowledge Base 검색', {
            'Knowledge Base ID': Config.KNOWLEDGE_BASE_ID,
            '검색 쿼리': query,
            '검색 결과 수 설정': Config.KB_NUMBER_OF_RESULTS
        })
        
        kb_context = self.retrieve_from_kb(query)
        
        print(f'   └─ KB 검색 결과: {len(kb_context)} 자 컨텍스트 획득', flush=True)
        if kb_context:
            # KB 결과 요약 출력
            kb_preview = kb_context[:300].replace('\n', ' ')
            print(f'   └─ KB 컨텍스트 미리보기: {kb_preview}...', flush=True)
        
        # ========================================
        # Step 3: 프롬프트 엔지니어링 - 질의 의도 분석
        # ========================================
        self._log_step(3, '프롬프트 엔지니어링 - 질의 의도 분석', {
            'LLM 모델': Config.MODEL_ID,
            '분석 목적': '차트 필요 여부, 차트 타입, 데이터 필터 조건 판단'
        })
        
        intent = self.analyze_query_intent(query, available_data)
        
        # Multi-Step Reasoning 분석 결과 출력
        reasoning = intent.get('reasoning', {})
        if reasoning:
            print(f'   └─ 🧠 Multi-Step Reasoning 분석:', flush=True)
            
            step1 = reasoning.get('step1_extraction', {})
            print(f'      ├─ Step1 (요소 추출):', flush=True)
            print(f'         ├─ 대상: {step1.get("target", "N/A")}', flush=True)
            print(f'         ├─ 측정값: {step1.get("metric", "N/A")}', flush=True)
            print(f'         └─ 조건: {step1.get("conditions", "N/A")}', flush=True)
            
            step2 = reasoning.get('step2_column_mapping', {})
            print(f'      ├─ Step2 (컬럼 매핑):', flush=True)
            print(f'         ├─ 사용자 표현: {step2.get("user_expression", "N/A")}', flush=True)
            print(f'         ├─ 매핑 컬럼: {step2.get("mapped_column", "N/A")}', flush=True)
            print(f'         └─ 매핑 이유: {step2.get("mapping_reason", "N/A")}', flush=True)
            
            step3 = reasoning.get('step3_confidence', {})
            print(f'      ├─ Step3 (확신도): {step3.get("level", "N/A")} - {step3.get("reason", "N/A")}', flush=True)
            
            step4 = reasoning.get('step4_decision', {})
            print(f'      └─ Step4 (결정): {step4.get("action", "N/A")} - {step4.get("explanation", "N/A")}', flush=True)
        
        # 계산 필요 여부 출력
        calc_required = intent.get('calculation_required', {})
        if calc_required.get('needed'):
            print(f'   └─ 📐 계산 필요:', flush=True)
            print(f'      ├─ 타입: {calc_required.get("type", "N/A")}', flush=True)
            print(f'      ├─ 기준 컬럼: {calc_required.get("base_column", "N/A")}', flush=True)
            print(f'      └─ 변화 컬럼: {calc_required.get("change_column", "N/A")}', flush=True)
        
        print(f'   └─ 📊 최종 분석 결과:', flush=True)
        print(f'      ├─ 출력 형식: {intent.get("output_format", "table")}', flush=True)
        print(f'      ├─ 차트 필요: {intent.get("needs_chart")}', flush=True)
        print(f'      ├─ 표 표시: {intent.get("show_table", True)}', flush=True)
        print(f'      ├─ 차트 타입: {intent.get("chart_type", "none")}', flush=True)
        print(f'      ├─ 제목: {intent.get("chart_title")}', flush=True)
        print(f'      ├─ 분석 유형: {intent.get("analysis_type")}', flush=True)
        print(f'      ├─ 데이터 필터: {intent.get("data_filter")}', flush=True)
        chart_config = intent.get('chart_config', {})
        if chart_config:
            print(f'      └─ 설정: x축={chart_config.get("x_axis")}, y축={chart_config.get("y_axis")}, y축타입={chart_config.get("y_axis_type")}, y축라벨={chart_config.get("y_axis_label")}', flush=True)
        else:
            print(f'      └─ 설정: 기본값 사용', flush=True)
        
        # ========================================
        # Step 3.5: 명확화 필요 여부 확인
        # ========================================
        if intent.get('needs_clarification'):
            clarification_msg = intent.get('clarification_message', '질의를 더 구체적으로 해주세요.')
            reasoning = intent.get('reasoning', {})
            confidence = reasoning.get('step3_confidence', {})
            
            print(f'   └─ ⚠️ 명확화 필요: {confidence.get("level", "UNKNOWN")}', flush=True)
            print(f'   └─ 이유: {confidence.get("reason", "N/A")}', flush=True)
            print(f'   └─ 메시지: {clarification_msg}', flush=True)
            
            total_time = time.time() - total_start_time
            self._log_separator('Agent 처리 완료 (명확화 요청)')
            
            return {
                'success': True,
                'query': query,
                'answer': clarification_msg,
                'has_chart': False,
                'needs_clarification': True,
                'bedrock_time': 0,
                'total_time': round(total_time, 2)
            }
        
        # ========================================
        # Step 4: 도구 선택 및 실행
        # ========================================
        if intent.get('needs_chart'):
            self._log_step(4, '도구 선택: 코드 인터프리터 (차트 생성)', {
                '선택된 도구': 'ChartGenerator (matplotlib 기반)',
                '차트 타입': intent.get('chart_type'),
                '실행 방식': 'Python 코드 동적 생성 → subprocess 실행 → Base64 이미지 반환'
            })
            
            # 차트 데이터 추출
            print(f'\n   📊 데이터 추출 중...', flush=True)
            chart_data = self.extract_chart_data(full_df, intent)
            
            if chart_data.get('error'):
                total_time = time.time() - total_start_time
                print(f'   └─ ❌ 데이터 추출 실패: {chart_data["error"]}', flush=True)
                return {
                    'success': False,
                    'error': chart_data['error'],
                    'has_chart': False,
                    'bedrock_time': 0,
                    'total_time': round(total_time, 2)
                }
            
            # 데이터 추출 결과 로깅
            is_multi = chart_data.get('multi_series', False)
            if is_multi:
                series_count = len(chart_data.get('series', []))
                data_points = len(chart_data.get('labels', []))
                print(f'   └─ ✅ 다중 시리즈 데이터 추출 완료', flush=True)
                print(f'      ├─ 시리즈 수: {series_count}개', flush=True)
                print(f'      ├─ 데이터 포인트: {data_points}개', flush=True)
                for s in chart_data.get('series', []):
                    print(f'      ├─ {s["name"]}: {s["values"][:3]}...', flush=True)
            else:
                data_points = len(chart_data.get('values', []))
                print(f'   └─ ✅ 단일 시리즈 데이터 추출 완료', flush=True)
                print(f'      ├─ 데이터 포인트: {data_points}개', flush=True)
                print(f'      ├─ 라벨: {chart_data.get("labels", [])[:5]}...', flush=True)
                print(f'      └─ 값: {chart_data.get("values", [])[:5]}...', flush=True)
            
            # ========================================
            # Step 5: 코드 인터프리터 실행 (차트 생성)
            # ========================================
            self._log_step(5, '코드 인터프리터 실행 - 차트 생성', {
                '실행 방식': 'matplotlib Python 코드 생성 → subprocess 실행',
                '출력 형식': 'Base64 인코딩 PNG 이미지'
            })
            
            chart_result = self.generate_chart(intent, chart_data)
            
            if not chart_result.get('success'):
                print(f'   └─ ⚠️ 차트 생성 실패: {chart_result.get("error")}', flush=True)
                chart_result = {'success': False, 'image': None}
            else:
                img_size = len(chart_result.get('image', '')) if chart_result.get('image') else 0
                print(f'   └─ ✅ 차트 생성 성공 (이미지 크기: {img_size:,} bytes)', flush=True)
            
            # ========================================
            # Step 6: LLM 답변 생성
            # ========================================
            self._log_step(6, 'LLM 답변 생성', {
                'LLM 모델': Config.MODEL_ID,
                '입력 데이터': f'차트 데이터 + KB 컨텍스트 ({len(kb_context)}자)',
                '답변 유형': '차트 분석 + 인사이트'
            })
            
            answer, bedrock_time = self.generate_answer_with_chart(
                query, df, kb_context, intent, chart_data, chart_result
            )
            
            print(f'   └─ ✅ 답변 생성 완료 ({len(answer)}자)', flush=True)
            
            # ========================================
            # 처리 완료 요약
            # ========================================
            total_time = time.time() - total_start_time
            self._log_separator('Agent 처리 완료')
            print(f'📊 처리 요약:', flush=True)
            print(f'   ├─ 질의: {query[:50]}...', flush=True)
            print(f'   ├─ 차트 생성: {"성공" if chart_result.get("success") else "실패"}', flush=True)
            print(f'   ├─ 차트 타입: {intent.get("chart_type")}', flush=True)
            print(f'   ├─ 데이터 소스: S3 캐시 (메모리)', flush=True)
            print(f'   ├─ RAG 사용: {"예" if kb_context else "아니오"} ({len(kb_context)}자)', flush=True)
            print(f'   ├─ Bedrock 응답 시간: {bedrock_time:.2f}초', flush=True)
            print(f'   ├─ 전체 처리 시간: {total_time:.2f}초', flush=True)
            print(f'   └─ 답변 길이: {len(answer)}자', flush=True)
            
            return {
                'success': True,
                'query': query,
                'answer': answer,
                'has_chart': chart_result.get('success', False),
                'chart_image': chart_result.get('image'),
                'chart_type': intent.get('chart_type'),
                'chart_title': intent.get('chart_title'),
                'bedrock_time': round(bedrock_time, 2),
                'total_time': round(total_time, 2),
                'data_summary': {
                    'labels': chart_data.get('labels', []),
                    'values': chart_data.get('values', []),
                    'series': chart_data.get('series', []),
                    'count': data_points
                }
            }
        
        else:
            # ========================================
            # Step 4: 표 기반 답변 생성 (차트 불필요)
            # ========================================
            output_format = intent.get('output_format', 'table')
            show_table = intent.get('show_table', True)
            
            self._log_step(4, '도구 선택: 표 기반 답변', {
                '출력 형식': output_format,
                '표 표시': show_table,
                '이유': '시각화 키워드 없음 - 표 형식으로 답변'
            })
            
            # 데이터 추출 (차트와 동일한 로직 사용)
            print(f'\n   📊 표 데이터 추출 중...', flush=True)
            table_data = self.extract_chart_data(full_df, intent)
            
            if table_data.get('error'):
                total_time = time.time() - total_start_time
                print(f'   └─ ❌ 데이터 추출 실패: {table_data["error"]}', flush=True)
                return {
                    'success': False,
                    'error': table_data['error'],
                    'has_chart': False,
                    'bedrock_time': 0,
                    'total_time': round(total_time, 2)
                }
            
            # 다중 시리즈 여부 확인
            is_multi = table_data.get('multi_series', False)
            if is_multi:
                series_count = len(table_data.get('series', []))
                data_points = len(table_data.get('labels', []))
                print(f'   └─ ✅ 다중 시리즈 표 데이터 추출 완료', flush=True)
                print(f'      ├─ 시리즈 수: {series_count}개', flush=True)
                print(f'      ├─ 데이터 포인트: {data_points}개', flush=True)
                for s in table_data.get('series', []):
                    print(f'      ├─ {s["name"]}: {s["values"][:3]}...', flush=True)
            else:
                data_points = len(table_data.get('values', []))
                print(f'   └─ ✅ 표 데이터 추출 완료 ({data_points}개 항목)', flush=True)
            
            # ========================================
            # Step 5: 표 기반 LLM 답변 생성
            # ========================================
            self._log_step(5, 'LLM 답변 생성 (표 형식)', {
                'LLM 모델': Config.MODEL_ID,
                '출력 형식': output_format,
                '표 표시': show_table
            })
            
            answer, bedrock_time = self.generate_table_answer(
                query, df, kb_context, intent, table_data, show_table
            )
            
            print(f'   └─ ✅ 답변 생성 완료 ({len(answer)}자)', flush=True)
            
            # ========================================
            # 처리 완료 요약
            # ========================================
            total_time = time.time() - total_start_time
            self._log_separator('Agent 처리 완료 (표 모드)')
            print(f'📊 처리 요약:', flush=True)
            print(f'   ├─ 질의: {query[:50]}...', flush=True)
            print(f'   ├─ 출력 형식: {output_format}', flush=True)
            print(f'   ├─ 표 표시: {show_table}', flush=True)
            print(f'   ├─ 데이터 소스: S3 캐시 (메모리)', flush=True)
            print(f'   ├─ RAG 사용: {"예" if kb_context else "아니오"} ({len(kb_context)}자)', flush=True)
            print(f'   ├─ Bedrock 응답 시간: {bedrock_time:.2f}초', flush=True)
            print(f'   ├─ 전체 처리 시간: {total_time:.2f}초', flush=True)
            print(f'   └─ 답변 길이: {len(answer)}자', flush=True)
            
            return {
                'success': True,
                'query': query,
                'answer': answer,
                'has_chart': False,
                'show_table': show_table,
                'bedrock_time': round(bedrock_time, 2),
                'total_time': round(total_time, 2),
                'output_format': output_format,
                'data_summary': {
                    'labels': table_data.get('labels', []),
                    'values': table_data.get('values', []),
                    'count': data_points
                }
            }
