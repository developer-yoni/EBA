"""
Bedrock을 활용한 AI 분석 리포트 생성
"""
import boto3
import json
from config import Config

class AIReportGenerator:
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
    
    def retrieve_from_kb(self, query):
        """Knowledge Base에서 관련 정보 검색"""
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
            context = '\n\n'.join([
                f"[참고자료 {i+1}]\n{r.get('content', {}).get('text', '')}"
                for i, r in enumerate(results)
            ])
            
            return context
        
        except Exception as e:
            print(f'⚠️ Knowledge Base 검색 오류: {e}')
            return ''
    
    def invoke_bedrock(self, prompt, context=''):
        """Bedrock 모델 호출"""
        try:
            system_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            payload = {
                'anthropic_version': Config.ANTHROPIC_VERSION,
                'max_tokens': Config.MAX_TOKENS,
                'temperature': Config.TEMPERATURE,
                'messages': [
                    {
                        'role': 'user',
                        'content': system_prompt
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=Config.MODEL_ID,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        
        except Exception as e:
            print(f'❌ Bedrock 호출 오류: {e}')
            return None
    
    def generate_executive_summary(self, insights):
        """경영진 요약 리포트 생성"""
        # insights를 JSON 직렬화 가능한 형태로 변환
        insights_str = str(insights)
        
        prompt = f"""
다음은 한국 전기차 충전 인프라 현황 데이터 분석 결과입니다.

{insights_str}

위 데이터를 바탕으로 경영진을 위한 핵심 요약 리포트를 작성해주세요.

**중요: 제공된 데이터만 사용하세요**
- 위에 제공된 실제 데이터만 분석하고 인용하세요
- 추측하거나 가상의 데이터를 만들지 마세요
- 데이터에 없는 내용은 "데이터 없음" 또는 "분석 불가"로 표시하세요
- 구체적인 숫자를 언급할 때는 반드시 제공된 데이터에서 확인된 값만 사용하세요

**작성 형식:**
- 맨 처음에 핵심 요약 2문장을 작성해주세요 (가장 중요한 인사이트)
- 그 다음 상세 내용을 작성해주세요

**포함 내용 (제목에 넘버링 필수):**
1. 전체 현황 요약 (3-4문장)
2. 주요 발견사항 (Top 3-5)
3. 시장 트렌드 분석
4. 주요 사업자 현황
5. 권장사항

**Markdown 포맷팅 규칙 (반드시 준수):**
1. H1 대제목 구분 규칙
   - H1: "# 1. 섹션명"
   - H1의 위와 아래에 구분선(---) 추가
   - H1 아래에는 반드시 한 줄을 띄우고 내용 작성
   - **중요**: H2, H3에는 구분선을 넣지 않습니다 (오직 H1만)
   
   올바른 H1 예시:
   ```
   ---
   
   # 1. 경영진 요약
   
   현재 충전 인프라는...
   
   ---
   ```

2. 제목 넘버링 및 빈 줄 규칙
   - H2: "## 1. 전체 현황 요약"
   - H3: "### 1.1 주요 지표"
   - **필수**: 제목 다음 줄은 반드시 빈 줄로 두세요
   
   올바른 예시:
   ```
   ## 1. 전체 현황 요약
   
   현재 충전 인프라는...
   
   ### 1.1 주요 지표
   
   - 총 충전소: 92,021개
   ```

2. 들여쓰기 규칙
   - H2 아래 → 글머리 기호 1단계 ("- ")
   - H3 아래 → 글머리 기호 2단계 ("  - ")

3. 표 작성 규칙
   - GitHub-style Markdown table 사용
   - 예시:
     | 항목 | 값 |
     |------|-----|
     | A | 123 |
   - 표 아래에는 한 줄 빈 줄 추가
   - escape 문자 사용 금지 (\\|, \\- 금지)

4. 순수 Markdown만 사용
   - HTML, LaTeX, 코드블록 사용 금지

한국어로 작성하고, 비즈니스 의사결정에 도움이 되도록 명확하고 간결하게 작성해주세요.
"""
        
        # Knowledge Base에서 추가 컨텍스트 검색
        context = self.retrieve_from_kb('충전 인프라 현황 분석 트렌드')
        
        return self.invoke_bedrock(prompt, context)
    
    def generate_cpo_analysis(self, cpo_data):
        """CPO별 상세 분석"""
        cpo_str = str(cpo_data)
        
        prompt = f"""
다음은 충전사업자(CPO)별 데이터입니다:

{cpo_str}

각 사업자의 특징과 시장 포지션을 분석하고, 다음을 포함한 리포트를 작성해주세요:

**중요: 제공된 데이터만 사용하세요**
- 위에 제공된 실제 CPO 데이터만 분석하고 인용하세요
- 추측하거나 가상의 사업자 정보를 만들지 마세요
- 데이터에 없는 사업자는 언급하지 마세요
- 구체적인 숫자(충전소 수, 충전기 수 등)는 반드시 제공된 데이터에서 확인된 값만 사용하세요

**작성 형식:**
- 맨 처음에 핵심 요약 2문장을 작성해주세요 (가장 중요한 인사이트)
- 그 다음 상세 내용을 작성해주세요

**포함 내용 (제목에 넘버링 필수):**
1. 시장 점유율 분석
2. 주요 사업자 프로필
3. 경쟁 구도 분석
4. 성장 가능성 평가

**Markdown 포맷팅 규칙 (반드시 준수):**
1. H1 대제목 구분 규칙
   - H1: "# 1. 섹션명"
   - H1의 위와 아래에 구분선(---) 추가
   - H1 아래에는 반드시 한 줄을 띄우고 내용 작성
   - **중요**: H2, H3에는 구분선을 넣지 않습니다 (오직 H1만)
   
   올바른 H1 예시:
   ```
   ---
   
   # 1. CPO 분석
   
   충전사업자 시장은...
   
   ---
   ```

2. 제목 넘버링 및 빈 줄 규칙
   - H2: "## 1. 시장 점유율 분석"
   - H3: "### 1.1 상위 사업자"
   - **필수**: 제목 다음 줄은 반드시 빈 줄로 두세요
   
   올바른 예시:
   ```
   ## 1. 시장 점유율 분석
   
   현재 충전 인프라 시장은...
   
   ### 1.1 상위 사업자
   
   - 한국전력공사: 충전소 15,234개
   ```

2. 들여쓰기: H2 아래는 "- ", H3 아래는 "  - "
3. 표: GitHub-style Markdown table만 사용 (| 항목 | 값 |), escape 문자 금지
4. 순수 Markdown만 사용 (HTML, LaTeX, 코드블록 금지)

한국어로 작성해주세요.
"""
        
        context = self.retrieve_from_kb('충전사업자 CPO 분석')
        return self.invoke_bedrock(prompt, context)
    
    def generate_regional_analysis(self, region_data):
        """지역별 분석"""
        region_str = str(region_data)
        
        prompt = f"""
다음은 지역별 충전 인프라 데이터입니다:

{region_str}

지역별 특성과 인프라 분포를 분석하고, 다음을 포함한 리포트를 작성해주세요:

1. 지역별 인프라 현황
2. 수도권 vs 지방 비교
3. 인프라 격차 분석
4. 지역별 개선 방안

한국어로 작성해주세요.
"""
        
        context = self.retrieve_from_kb('지역별 충전 인프라 분석')
        return self.invoke_bedrock(prompt, context)
    
    def generate_trend_forecast(self, trend_data):
        """트렌드 및 예측"""
        trend_str = str(trend_data)
        
        prompt = f"""
다음은 시계열 트렌드 데이터입니다:

{trend_str}

과거 데이터를 바탕으로 다음을 분석해주세요:

**중요: 제공된 데이터만 사용하세요**
- 위에 제공된 실제 트렌드 데이터만 분석하세요
- 과거 데이터에 기반한 합리적인 추론만 하세요
- 확인되지 않은 미래 예측이나 가상의 시나리오를 만들지 마세요
- 구체적인 증감률이나 수치는 반드시 제공된 데이터에서 계산된 값만 사용하세요
- 예측은 "~로 예상됩니다" 대신 "~의 가능성이 있습니다" 같은 신중한 표현을 사용하세요

**작성 형식:**
- 맨 처음에 핵심 요약 2문장을 작성해주세요 (가장 중요한 인사이트)
- 그 다음 상세 내용을 작성해주세요

**포함 내용 (제목에 넘버링 필수):**
1. 성장 추세 분석 (실제 데이터 기반)
2. 계절성 패턴 (데이터에서 관찰된 패턴만)
3. 향후 3-6개월 전망 (신중한 표현 사용)
4. 주요 성장 동인 (데이터에서 확인 가능한 요인만)

**Markdown 포맷팅 규칙 (반드시 준수):**
1. H1 대제목 구분 규칙
   - H1: "# 1. 섹션명"
   - H1의 위와 아래에 구분선(---) 추가
   - H1 아래에는 반드시 한 줄을 띄우고 내용 작성
   - **중요**: H2, H3에는 구분선을 넣지 않습니다 (오직 H1만)
   
   올바른 H1 예시:
   ```
   ---
   
   # 1. 트렌드 및 예측
   
   최근 6개월간 충전기는...
   
   ---
   ```

2. 제목 넘버링 및 빈 줄 규칙
   - H2: "## 1. 성장 추세 분석"
   - H3: "### 1.1 월별 증감 추이"
   - **필수**: 제목 다음 줄은 반드시 빈 줄로 두세요
   
   올바른 예시:
   ```
   ## 1. 성장 추세 분석
   
   최근 6개월간 충전기는...
   
   ### 1.1 월별 증감 추이
   
   - 2025-10: +597기 증가
   ```

2. 들여쓰기: H2 아래는 "- ", H3 아래는 "  - "
3. 표: GitHub-style Markdown table만 사용 (| 항목 | 값 |), escape 문자 금지
4. 순수 Markdown만 사용 (HTML, LaTeX, 코드블록 금지)

한국어로 작성해주세요.
"""
        
        context = self.retrieve_from_kb('충전 인프라 성장 트렌드 예측')
        return self.invoke_bedrock(prompt, context)
    
    def generate_full_report(self, insights):
        """전체 종합 리포트 생성"""
        print('🤖 AI 분석 리포트 생성 중...\n')
        
        report = {
            'executive_summary': None,
            'cpo_analysis': None,
            'regional_analysis': None,
            'trend_forecast': None
        }
        
        # 경영진 요약
        print('📝 경영진 요약 생성 중...')
        report['executive_summary'] = self.generate_executive_summary(insights)
        
        # CPO 분석
        if insights.get('cpo_analysis') is not None:
            print('📝 CPO 분석 생성 중...')
            report['cpo_analysis'] = self.generate_cpo_analysis(insights['cpo_analysis'])
        
        # 트렌드 예측
        if insights.get('trend') is not None:
            print('📝 트렌드 예측 생성 중...')
            report['trend_forecast'] = self.generate_trend_forecast(insights['trend'])
        
        print('✅ AI 리포트 생성 완료\n')
        return report
