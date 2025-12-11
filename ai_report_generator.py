"""
Bedrock을 활용한 AI 분석 리포트 생성
"""
import boto3
import json
import pandas as pd
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
            
            if len(results) == 0:
                return ''
            
            context = '\n\n'.join([
                f"[참고자료 {i+1}]\n{r.get('content', {}).get('text', '')}"
                for i, r in enumerate(results)
            ])
            
            return context
        
        except Exception as e:
            print(f'❌ Knowledge Base 검색 오류: {e}', flush=True)
            return ''
    
    def invoke_bedrock(self, prompt, context=''):
        """Bedrock 모델 호출 (리포트 생성용)"""
        import time
        try:
            start_time = time.time()
            
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
            result = response_body['content'][0]['text']
            
            elapsed_time = time.time() - start_time
            print(f'✅ 리포트 생성 완료 (⏱️ {elapsed_time:.1f}초)', flush=True)
            return result
        
        except Exception as e:
            print(f'❌ Bedrock 호출 오류: {e}', flush=True)
            return f"리포트 생성 중 오류가 발생했습니다: {str(e)}"
    
    def invoke_bedrock_for_query(self, structured_prompt):
        """Bedrock 모델 호출 (커스텀 질의용 - 구조화된 프롬프트 사용)"""
        import time
        try:
            start_time = time.time()
            
            payload = {
                'anthropic_version': Config.ANTHROPIC_VERSION,
                'max_tokens': Config.MAX_TOKENS,
                'temperature': 0.3,  # 더 정확한 답변을 위해 낮은 temperature
                'messages': [
                    {
                        'role': 'user',
                        'content': structured_prompt
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
            result = response_body['content'][0]['text']
            
            elapsed_time = time.time() - start_time
            print(f'✅ Bedrock 응답 완료 (⏱️ {elapsed_time:.2f}초)', flush=True)
            
            return result, elapsed_time
        
        except Exception as e:
            print(f'❌ Bedrock 호출 오류: {e}')
            return None, 0
    
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
    
    def generate_gs_chargebee_report(self, target_month, target_insights, range_insights, target_data, range_data, available_months):
        """GS차지비 관점 AI 리포트 생성"""
        import time
        print(f'🤖 GS차지비 관점 AI 리포트 생성 중... (기준월: {target_month})\n')
        
        report = {
            'executive_summary': None,
            'cpo_analysis': None,
            'trend_forecast': None,
            'response_times': {}
        }
        
        # GS차지비 데이터 추출
        gs_target = target_data[target_data['CPO명'] == 'GS차지비'] if 'CPO명' in target_data.columns else None
        gs_range = range_data[range_data['CPO명'] == 'GS차지비'] if 'CPO명' in range_data.columns else None
        
        # GS차지비 정보 문자열 생성
        gs_info = ""
        if gs_target is not None and len(gs_target) > 0:
            gs_row = gs_target.iloc[0]
            gs_info = f"""
GS차지비 {target_month} 현황:
- 순위: {gs_row.get('순위', 'N/A')}위
- 충전소 수: {gs_row.get('충전소수', 'N/A')}개
- 완속충전기: {gs_row.get('완속충전기', 'N/A')}기
- 급속충전기: {gs_row.get('급속충전기', 'N/A')}기
- 총충전기: {gs_row.get('총충전기', 'N/A')}기
- 시장점유율: {gs_row.get('시장점유율', 'N/A')}
- 순위변동: {gs_row.get('순위변동', 'N/A')}
- 충전소증감: {gs_row.get('충전소증감', 'N/A')}
- 완속증감: {gs_row.get('완속증감', 'N/A')}
- 급속증감: {gs_row.get('급속증감', 'N/A')}
- 총증감: {gs_row.get('총증감', 'N/A')}
"""
        
        # GS차지비 월별 추이
        gs_trend = ""
        if gs_range is not None and len(gs_range) > 0:
            gs_trend = "\nGS차지비 월별 추이:\n"
            for _, row in gs_range.sort_values('snapshot_month').iterrows():
                gs_trend += f"- {row.get('snapshot_month', 'N/A')}: 순위 {row.get('순위', 'N/A')}위, 총충전기 {row.get('총충전기', 'N/A')}기, 시장점유율 {row.get('시장점유율', 'N/A')}\n"
        
        # 경쟁사 분석 (상위 10개사)
        competitor_info = ""
        if 'CPO명' in target_data.columns:
            top10 = target_data.nlargest(10, '총충전기') if '총충전기' in target_data.columns else target_data.head(10)
            competitor_info = f"\n{target_month} 상위 10개 CPO:\n"
            for _, row in top10.iterrows():
                cpo_name = row.get('CPO명', 'N/A')
                rank = row.get('순위', 'N/A')
                stations = row.get('충전소수')
                total_chargers = row.get('총충전기')
                market_share = row.get('시장점유율')
                total_change = row.get('총증감')
                
                # NaN 처리
                stations = int(stations) if pd.notna(stations) else 'N/A'
                total_chargers = int(total_chargers) if pd.notna(total_chargers) else 'N/A'
                market_share = f"{market_share:.1f}%" if pd.notna(market_share) else 'N/A'
                total_change = f"{int(total_change):+d}" if pd.notna(total_change) else 'N/A'
                
                competitor_info += f"- {cpo_name}: 순위 {rank}위, 충전소 {stations}개, 총충전기 {total_chargers}기, 시장점유율 {market_share}, 총증감 {total_change}기\n"
        
        # 1. 경영진 요약 (GS차지비 관점)
        print('📝 [1/3] 경영진 요약 생성 중...', flush=True)
        start_time = time.time()
        report['executive_summary'] = self._generate_gs_executive_summary(
            target_month, gs_info, gs_trend, competitor_info, target_insights, available_months
        )
        report['response_times']['executive_summary'] = round(time.time() - start_time, 1)
        
        # 2. 경쟁 분석 (GS차지비 관점)
        print('📝 [2/3] 경쟁 분석 생성 중...', flush=True)
        start_time = time.time()
        report['cpo_analysis'] = self._generate_gs_competitor_analysis(
            target_month, gs_info, gs_trend, competitor_info, target_insights, range_insights
        )
        report['response_times']['cpo_analysis'] = round(time.time() - start_time, 1)
        
        # 3. 전략 제안 (GS차지비 관점)
        print('📝 [3/3] 전략 제안 생성 중...', flush=True)
        start_time = time.time()
        report['trend_forecast'] = self._generate_gs_strategy(
            target_month, gs_info, gs_trend, competitor_info, range_insights, available_months
        )
        report['response_times']['trend_forecast'] = round(time.time() - start_time, 1)
        
        total_time = sum(report['response_times'].values())
        print(f'✅ GS차지비 AI 리포트 생성 완료 (총 ⏱️ {total_time:.1f}초)\n', flush=True)
        return report
    
    def _generate_gs_executive_summary(self, target_month, gs_info, gs_trend, competitor_info, insights, available_months):
        """GS차지비 경영진 요약"""
        prompt = f"""
당신은 GS차지비의 전략 컨설턴트입니다. 다음 데이터를 바탕으로 GS차지비 경영진을 위한 핵심 요약 리포트를 작성해주세요.

## 기준월: {target_month}
## 분석 가능 기간: {available_months[0]} ~ {available_months[-1]} ({len(available_months)}개월)

## GS차지비 현황
{gs_info}

## GS차지비 월별 추이
{gs_trend}

## 경쟁사 현황
{competitor_info}

## 전체 시장 인사이트
{str(insights)}

---

**작성 지침:**
1. GS차지비 관점에서 가장 중요한 인사이트 3가지를 먼저 제시
2. 시장 내 GS차지비의 포지션 분석
3. 주요 경쟁사 대비 강점/약점
4. 즉각적인 주의가 필요한 사항

**Markdown 포맷팅 규칙:**
- H2: "## 1. 섹션명" (제목 다음 줄은 빈 줄)
- 글머리 기호: "- " 사용
- 표: GitHub-style Markdown table
- 순수 Markdown만 사용 (HTML, LaTeX 금지)

한국어로 작성해주세요.
"""
        context = self.retrieve_from_kb('GS차지비 충전 인프라 시장 분석')
        return self.invoke_bedrock(prompt, context)
    
    def _generate_gs_competitor_analysis(self, target_month, gs_info, gs_trend, competitor_info, target_insights, range_insights):
        """GS차지비 경쟁 분석"""
        prompt = f"""
당신은 GS차지비의 경쟁 분석 전문가입니다. 다음 데이터를 바탕으로 GS차지비의 경쟁 환경을 분석해주세요.

## 기준월: {target_month}

## GS차지비 현황
{gs_info}

## GS차지비 월별 추이
{gs_trend}

## 경쟁사 현황
{competitor_info}

## 시장 인사이트
{str(target_insights)}

---

**작성 지침:**
1. GS차지비 vs 상위 경쟁사 비교 분석
2. 시장점유율 변화 추이 분석
3. 충전기 증설 속도 비교
4. 경쟁사별 전략 추정 및 GS차지비 대응 방안
5. 벤치마킹 대상 및 포인트

**포함 내용:**
- 경쟁사 대비 GS차지비의 강점/약점 표
- 시장점유율 순위 변동 분석
- 급속/완속 충전기 비율 비교
- 성장률 비교

**Markdown 포맷팅 규칙:**
- H2: "## 1. 섹션명" (제목 다음 줄은 빈 줄)
- 글머리 기호: "- " 사용
- 표: GitHub-style Markdown table
- 순수 Markdown만 사용 (HTML, LaTeX 금지)

한국어로 작성해주세요.
"""
        context = self.retrieve_from_kb('충전사업자 CPO 경쟁 분석')
        return self.invoke_bedrock(prompt, context)
    
    def _generate_gs_strategy(self, target_month, gs_info, gs_trend, competitor_info, range_insights, available_months):
        """GS차지비 전략 제안"""
        prompt = f"""
당신은 GS차지비의 전략 기획 전문가입니다. 다음 데이터를 바탕으로 GS차지비의 성장 전략을 제안해주세요.

## 기준월: {target_month}
## 분석 기간: {available_months[0]} ~ {available_months[-1]}

## GS차지비 현황
{gs_info}

## GS차지비 월별 추이
{gs_trend}

## 경쟁사 현황
{competitor_info}

## 시장 트렌드
{str(range_insights.get('trend', {}))}

---

**작성 지침:**
1. 단기 전략 (3개월 이내)
   - 즉시 실행 가능한 액션 아이템
   - 시장점유율 방어/확대 방안
   
2. 중기 전략 (6개월~1년)
   - 충전기 증설 계획 제안
   - 급속/완속 비율 최적화 방안
   
3. 장기 전략 (1년 이상)
   - 시장 포지셔닝 전략
   - 차별화 전략
   
4. 리스크 요인 및 대응 방안

5. KPI 제안
   - 모니터링해야 할 핵심 지표
   - 목표 수치 제안

**Markdown 포맷팅 규칙:**
- H2: "## 1. 섹션명" (제목 다음 줄은 빈 줄)
- 글머리 기호: "- " 사용
- 표: GitHub-style Markdown table
- 순수 Markdown만 사용 (HTML, LaTeX 금지)

한국어로 작성해주세요.
"""
        context = self.retrieve_from_kb('충전 인프라 성장 전략')
        return self.invoke_bedrock(prompt, context)

    
    def generate_kpi_snapshot_report(self, target_month, target_insights, target_data, available_months):
        """KPI Report 생성 - 현황 중심"""
        print(f'📊 KPI Report 생성 중... (기준월: {target_month})', flush=True)
        
        # GS차지비 데이터 추출
        gs_data = target_data[target_data['CPO명'] == 'GS차지비'] if 'CPO명' in target_data.columns else None
        gs_info = ""
        if gs_data is not None and len(gs_data) > 0:
            gs_row = gs_data.iloc[0]
            gs_info = f"""
GS차지비 {target_month} 현황:
- 순위: {gs_row.get('순위', 'N/A')}위
- 충전소 수: {gs_row.get('충전소수', 'N/A')}개
- 완속충전기: {gs_row.get('완속충전기', 'N/A')}기
- 급속충전기: {gs_row.get('급속충전기', 'N/A')}기
- 총충전기: {gs_row.get('총충전기', 'N/A')}기
- 시장점유율: {gs_row.get('시장점유율', 'N/A')}
- 전월 대비 증감: 충전소 {gs_row.get('충전소증감', 'N/A')}, 완속 {gs_row.get('완속증감', 'N/A')}, 급속 {gs_row.get('급속증감', 'N/A')}, 총 {gs_row.get('총증감', 'N/A')}
"""
        
        # 상위 10개 CPO 정보
        top10_info = ""
        if 'CPO명' in target_data.columns and '총충전기' in target_data.columns:
            top10 = target_data.nlargest(10, '총충전기')
            top10_info = f"\n{target_month} 상위 10개 CPO:\n"
            
            # 간단한 상태 확인
            print(f'🔍 상위 10개 CPO 데이터: {len(top10)}개 준비완료')
            
            for idx, row in top10.iterrows():
                # 안전한 데이터 추출
                rank = row.get('순위', 'N/A')
                cpo_name = row.get('CPO명', 'N/A')
                stations = row.get('충전소수')
                total_chargers = row.get('총충전기')
                slow_chargers = row.get('완속충전기')
                fast_chargers = row.get('급속충전기')
                market_share = row.get('시장점유율')
                total_change = row.get('총증감')
                
                # 강화된 NaN 처리 - 절대 'N/A'나 '-'를 반환하지 않음
                def safe_convert_int(val):
                    try:
                        if pd.isna(val) or val is None:
                            return 0
                        if isinstance(val, str) and (val.strip() == '' or val.strip() == '-'):
                            return 0
                        return int(float(val))
                    except (ValueError, TypeError):
                        return 0
                
                def safe_convert_float(val):
                    try:
                        if pd.isna(val) or val is None:
                            return 0.0
                        if isinstance(val, str) and (val.strip() == '' or val.strip() == '-'):
                            return 0.0
                        return float(val)
                    except (ValueError, TypeError):
                        return 0.0
                
                stations = safe_convert_int(stations)
                total_chargers = safe_convert_int(total_chargers)
                slow_chargers = safe_convert_int(slow_chargers)
                fast_chargers = safe_convert_int(fast_chargers)
                market_share_val = safe_convert_float(market_share)
                total_change_val = safe_convert_int(total_change)
                
                # 포맷팅
                market_share_str = f"{market_share_val:.1f}%"
                total_change_str = f"{total_change_val:+d}" if total_change_val != 0 else "0"
                
                top10_info += f"- {rank}위. {cpo_name}: 충전소 {stations:,}개, 총충전기 {total_chargers:,}기 (완속 {slow_chargers:,}, 급속 {fast_chargers:,}), 점유율 {market_share_str}, 전월 대비 {total_change_str}기\n"
        
        # 전체 시장 요약
        summary = target_insights.get('summary', {})
        market_summary = f"""
전국 충전 인프라 현황 ({target_month}):
- 총 CPO 수: {summary.get('total_cpos', 'N/A')}개
- 총 충전소 수: {summary.get('total_stations', 'N/A')}개
- 총 충전기 수: {summary.get('total_chargers', 'N/A')}기
- 완속충전기: {summary.get('slow_chargers', 'N/A')}기 ({summary.get('slow_ratio', 'N/A')}%)
- 급속충전기: {summary.get('fast_chargers', 'N/A')}기 ({summary.get('fast_ratio', 'N/A')}%)
"""
        
        # 날짜 정보를 명확하게 추출
        report_start_month = available_months[0]
        report_end_month = available_months[-1]
        report_target_month = target_month
        report_period_count = len(available_months)
        
        prompt = f"""
<role>
당신은 한국 EV 충전 인프라 데이터를 기반으로 경영진이 보는 수준의 시각적·정리형 리포트를 작성하는 시니어 데이터 애널리스트이자 리포트 디자이너입니다.
</role>

<context>
- 데이터는 "전국의 CPO(충전사업자)별 월별 충전소 수, 완속/급속/총 충전기 수"에 한정됩니다.
- 이 데이터는 국가 전체 인프라 규모, 사업자별 상대적 규모/순위, 월별 증감/추세 정도까지만 알 수 있습니다.
- 우리 회사는 CPO 중 하나인 "GS차지비"이며, 리포트에서는 GS차지비의 위치와 특징을 항상 별도로 짚어줘야 합니다.
- 차량 수, 매출, 이용률, 충전건수, 고객 세그먼트 등 "지금 주어진 데이터로는 알 수 없는 항목"은 절대로 만들어내지 마십시오.
</context>

## ⚠️⚠️⚠️ 날짜 정보 - 반드시 이 날짜만 사용하세요! ⚠️⚠️⚠️

**분석 기간 시작:** {report_start_month}
**분석 기간 종료:** {report_end_month}
**기준월 (최신 데이터):** {report_target_month}
**분석 개월 수:** {report_period_count}개월

⚠️ 리포트에서 날짜를 언급할 때는 반드시 위의 날짜만 사용하세요.
⚠️ 절대로 2024-10, 2024-09 같은 다른 날짜를 만들거나 추측하지 마세요.

## 기준월: {report_target_month}

## 전국 충전 인프라 현황
{market_summary}

## GS차지비 현황
{gs_info}

## 상위 10개 CPO
{top10_info}

---

**리포트 유형: EV Infra KPI Snapshot Report**

이 리포트는 한 기준월({target_month})을 기준으로 전국 EV 충전 인프라의 "단면(KPI 스냅샷)"을 정리하는 리포트입니다.

**중요: 기준월은 {target_month}입니다. 리포트 내용에서 날짜를 언급할 때는 반드시 {target_month}를 사용하세요.**

**주요 내용:**
- 국가 전체 총 충전소 수, 완속/급속/총 충전기 수 (KPI)
- 전월 대비 증감 수치
- 상위 5~10개 CPO의 충전기/충전소 규모 비교 표
- 상위 CPO 집중도(Top N 비중) 요약
- GS차지비의 순위, 규모, 전체 대비 비중 요약

**날짜 표기 규칙:**
- 기준월을 언급할 때는 반드시 "{target_month}"를 사용하세요.
- 다른 날짜(예: 2024-10)를 임의로 만들지 마세요.
- 위에 제공된 데이터의 날짜만 사용하세요.

**중요: 리포트는 반드시 아래 HTML 타이틀로 정확히 시작해야 합니다:**

<div align="center">
<h1>📊 EV Infra KPI Report</h1>
<p style="color: #666; font-size: 14px;">분석 기간: {report_start_month} ~ {report_end_month} ({report_period_count}개월) | 기준월: {report_target_month}</p>
</div>

**⚠️ 위 HTML을 그대로 복사하세요. 날짜를 절대 변경하지 마세요:**
- {report_start_month} (분석 시작)
- {report_end_month} (분석 종료)  
- {report_target_month} (기준월)

이 날짜들만 사용하고, 다른 날짜(예: 2024-10)를 절대 만들지 마세요.

**레이아웃 규칙:**

1) 제목 계층 (넘버링 포함):
- H1: `# 1. Market-wide EV Infra KPIs`
- H2: `## 1.1 National Infrastructure Overview`
- H3: `### 1.1.1 Total Chargers`

3) 섹션별 콜아웃 (H1 바로 아래):
```
> 💡 **Key Insight**
> 
> 이 섹션의 핵심 메시지를 2문장으로 요약합니다.
```

4) 숫자 표현:
- 여러 수치는 표(Table) 형태로 정리
- 예시:

| Rank | CPO | Chargers | Share | MoM Change |
|------|------------|----------|--------|------------|
| 1 | GS차지비 | 73,290 | 16.2% | +120 |

5) GS차지비 강조:
- 표에서 GS차지비 행에 별도 코멘트
- 별도 bullet에서 GS차지비만 따로 분석

**할루시네이션 방지:**
- 오직 "충전소 수, 완속/급속/총 충전기 수" 데이터에서 계산 가능한 사실만 사용
- 전망/해석 시 반드시 관련 수치를 먼저 제시한 뒤 "이 수치를 기반으로 볼 때 ~로 해석될 수 있습니다."와 같이 추론임을 명시

**금지 사항:**
- 차트 제안 문장을 작성하지 마세요 (예: "📊 차트 제안:", "라인 차트로 시각화하면" 등)
- 시각화 관련 제안을 하지 마세요

**리포트 구조:**

# 1. Market-wide EV Infra KPIs
> 💡 콜아웃

## 1.1 National Infrastructure Overview
- 총 충전소 수
- 총 충전기 수 (완속/급속 비율)
- 전월 대비 증감

## 1.2 Top CPO Market Share
- 상위 5~10개 CPO 표
- 상위 3/5사 집중도

# 2. GS차지비 Position Summary
> 💡 콜아웃

## 2.1 Current Standing
- 순위, 규모, 점유율
- 전월 대비 변화

## 2.2 Competitive Context
- 상위 그룹 내 위치
- 주요 경쟁사 대비 특징

한국어로 작성하되, HTML 타이틀과 표는 위 형식을 정확히 따라주세요.
"""
        
        context = self.retrieve_from_kb('충전 인프라 KPI 현황 분석')
        return self.invoke_bedrock(prompt, context)
    
    def generate_cpo_ranking_report(self, target_month, target_insights, target_data, available_months):
        """CPO Ranking & GS차지비 Positioning Report 생성"""
        print(f'🏢 CPO Ranking Report 생성 중... (기준월: {target_month})', flush=True)
        
        # 날짜 정보를 명확하게 추출
        report_start_month = available_months[0]
        report_end_month = available_months[-1]
        report_target_month = target_month
        report_period_count = len(available_months)
        
        # GS차지비 데이터 추출
        gs_data = target_data[target_data['CPO명'] == 'GS차지비'] if 'CPO명' in target_data.columns else None
        gs_info = ""
        if gs_data is not None and len(gs_data) > 0:
            gs_row = gs_data.iloc[0]
            total = int(gs_row.get('총충전기', 0))
            stations = int(gs_row.get('충전소수', 0))
            avg_per_site = total / stations if stations > 0 else 0
            slow_ratio = (int(gs_row.get('완속충전기', 0)) / total * 100) if total > 0 else 0
            fast_ratio = (int(gs_row.get('급속충전기', 0)) / total * 100) if total > 0 else 0
            
            gs_info = f"""
GS차지비 {target_month} 상세:
- 순위: {gs_row.get('순위', 'N/A')}위
- 충전소 수: {stations:,}개
- 총충전기: {total:,}기
- 완속충전기: {gs_row.get('완속충전기', 'N/A')}기 ({slow_ratio:.1f}%)
- 급속충전기: {gs_row.get('급속충전기', 'N/A')}기 ({fast_ratio:.1f}%)
- 충전소당 평균 충전기: {avg_per_site:.2f}기
- 시장점유율: {gs_row.get('시장점유율', 'N/A')}
- 전월 대비 증감: {gs_row.get('총증감', 'N/A')}기
"""
        
        # 상위 15개 CPO 상세 정보 (충전소당 평균 포함)
        top15_detail = ""
        if 'CPO명' in target_data.columns and '총충전기' in target_data.columns:
            top15 = target_data.nlargest(15, '총충전기')
            top15_detail = f"\n상위 15개 CPO 상세 ({target_month}):\n"
            for _, row in top15.iterrows():
                cpo_name = row.get('CPO명', 'N/A')
                total = int(row.get('총충전기', 0))
                stations = int(row.get('충전소수', 0))
                slow = int(row.get('완속충전기', 0))
                fast = int(row.get('급속충전기', 0))
                avg_per_site = total / stations if stations > 0 else 0
                slow_pct = (slow / total * 100) if total > 0 else 0
                fast_pct = (fast / total * 100) if total > 0 else 0
                
                top15_detail += f"- {row.get('순위', 'N/A')}위. {cpo_name}: 총 {total:,}기, 충전소 {stations:,}개, 충전소당 {avg_per_site:.2f}기, 완속 {slow:,}기({slow_pct:.1f}%), 급속 {fast:,}기({fast_pct:.1f}%), 점유율 {row.get('시장점유율', 'N/A')}, 증감 {row.get('총증감', 'N/A')}기\n"
        
        # 시장 구조
        summary = target_insights.get('summary', {})
        market_structure = f"""
전국 시장 구조 ({target_month}):
- 총 CPO 수: {summary.get('total_cpos', 'N/A')}개
- 총 충전소: {summary.get('total_stations', 'N/A')}개
- 총 충전기: {summary.get('total_chargers', 'N/A')}기
- 완속 비중: {summary.get('slow_ratio', 'N/A')}%
- 급속 비중: {summary.get('fast_ratio', 'N/A')}%
"""
        
        prompt = f"""
<role>
당신은 한국 EV 충전 인프라 데이터를 기반으로 경영진이 보는 수준의 시각적·정리형 리포트를 작성하는 시니어 데이터 애널리스트이자 리포트 디자이너입니다.
</role>

<context>
- 데이터는 "전국의 CPO(충전사업자)별 월별 충전소 수, 완속/급속/총 충전기 수"에 한정됩니다.
- 우리 회사는 CPO 중 하나인 "GS차지비"이며, 리포트에서는 GS차지비의 위치와 특징을 항상 별도로 짚어줘야 합니다.
- 차량 수, 매출, 이용률, 충전건수, 고객 세그먼트 등 "지금 주어진 데이터로는 알 수 없는 항목"은 절대로 만들어내지 마십시오.
</context>

## ⚠️⚠️⚠️ 날짜 정보 - 반드시 이 날짜만 사용하세요! ⚠️⚠️⚠️

**분석 기간 시작:** {report_start_month}
**분석 기간 종료:** {report_end_month}
**기준월 (최신 데이터):** {report_target_month}
**분석 개월 수:** {report_period_count}개월

⚠️ 리포트에서 날짜를 언급할 때는 반드시 위의 날짜만 사용하세요.
⚠️ 절대로 2024-10, 2024-09 같은 다른 날짜를 만들거나 추측하지 마세요.

## 기준월: {report_target_month}

## 전국 시장 구조
{market_structure}

## GS차지비 상세
{gs_info}

## 상위 15개 CPO 상세
{top15_detail}

---

**리포트 유형: EV Infra CPO Ranking & GS차지비 Positioning Report**

이 리포트는 같은 기준월({report_target_month})을 기준으로 CPO별 순위/규모/구조를 상세히 보는 "사업자 관점" 리포트입니다.

**중요: 기준월은 {report_target_month}입니다. 리포트 내용에서 날짜를 언급할 때는 반드시 {report_target_month}를 사용하세요.**

**주요 내용:**
- CPO별 충전기 수 기준 랭킹 표 (Top 10~15)
- 각 CPO별 "충전소당 평균 충전기 수"(total_count / station_count) 비교
- 완속/급속 비중 요약
- GS차지비 중심 분석:
  - 시장 내 순위
  - 상위 그룹과의 격차
  - 완속/급속 구조 특징
  - "Efficiency vs Coverage" 축으로 주요 경쟁사와 GS차지비 포지션 비교

**날짜 표기 규칙:**
- 기준월을 언급할 때는 반드시 "{target_month}"를 사용하세요.
- 다른 날짜를 임의로 만들지 마세요.
- 위에 제공된 데이터의 날짜만 사용하세요.

**중요: 리포트는 반드시 아래 HTML 타이틀로 정확히 시작해야 합니다:**

<div align="center">
<h1>📈 EV Infra CPO Market Report</h1>
<p style="color: #666; font-size: 14px;">분석 기간: {report_start_month} ~ {report_end_month} ({report_period_count}개월) | 기준월: {report_target_month}</p>
</div>

**⚠️ 위 HTML을 그대로 복사하세요. 날짜를 절대 변경하지 마세요:**
- {report_start_month} (분석 시작)
- {report_end_month} (분석 종료)  
- {report_target_month} (기준월)

이 날짜들만 사용하고, 다른 날짜(예: 2024-10)를 절대 만들지 마세요.

**레이아웃 규칙:**

1) 제목 계층 (넘버링):
- H1: `# 1. CPO Ranking Overview`
- H2: `## 1.1 Top 15 CPO Leaderboard`
- H3: `### 1.1.1 Charger Count Ranking`

3) 섹션별 콜아웃 (H1 바로 아래):
```
> 💡 **Key Insight**
> 
> 핵심 메시지 2문장
```

4) 표 형식 (충전소당 평균 포함):

| Rank | CPO | Chargers | Sites | Avg/Site | AC% | DC% | Share |
|------|------------|----------|-------|----------|-----|-----|-------|
| 1 | GS차지비 | 73,290 | 13,173| 5.56 | 85% | 15% | 16.2% |

5) GS차지비 포지셔닝:
- "Efficiency vs Coverage" 분석
- 충전소당 평균 충전기 수로 효율성 평가
- 완속/급속 비중으로 전략 특징 분석

**할루시네이션 방지:**
- 오직 충전소/충전기 수 데이터만 사용
- 해석 시 반드시 수치 먼저 제시 후 추론임을 명시

**금지 사항:**
- 차트 제안 문장을 작성하지 마세요 (예: "📊 차트 제안:", "라인 차트로 시각화하면" 등)
- 시각화 관련 제안을 하지 마세요

**리포트 구조:**

# 1. CPO Ranking Overview
> 💡 콜아웃

## 1.1 Top 15 CPO Leaderboard
- 랭킹 표 (충전소당 평균 포함)

## 1.2 Market Structure Analysis
- 상위 3/5사 집중도
- 롱테일 구조

# 2. GS차지비 Detailed Positioning
> 💡 콜아웃

## 2.1 Ranking & Scale
- 순위, 규모, 점유율

## 2.2 Efficiency vs Coverage Analysis

### 2.2.1 Operational Efficiency Matrix
- 충전소당 평균 충전기 수 비교 (상위 10개사만 표로 작성)
- **중요**: 표는 반드시 완전한 형태로 작성하세요. 모든 행과 열을 포함해야 합니다.
- 표 예시:

| Rank | CPO | Avg Chargers/Site |
|------|-----|-------------------|
| 1 | CPO명 | 5.56 |

### 2.2.2 Strategic Positioning
- 완속/급속 비중 특징
- 주요 경쟁사 대비 포지션

## 2.3 Strategic Implications
- GS차지비의 구조적 특징
- 경쟁 우위/약점

한국어로 작성하되, HTML 타이틀과 표는 위 형식을 정확히 따라주세요.
"""
        
        context = self.retrieve_from_kb('CPO 순위 경쟁 분석')
        return self.invoke_bedrock(prompt, context)
    
    def generate_monthly_trend_report(self, target_month, range_insights, range_data, available_months):
        """Monthly Trend Report 생성 - 시계열 분석 중심"""
        print(f'📈 Monthly Trend Report 생성 중... (기준월: {target_month})', flush=True)
        
        # 날짜 정보를 명확하게 추출
        report_start_month = available_months[0]
        report_end_month = available_months[-1]
        report_target_month = target_month
        report_period_count = len(available_months)
        
        # 월별 추이 데이터 추출
        monthly_trend = ""
        if 'snapshot_month' in range_data.columns:
            monthly_summary = range_data.groupby('snapshot_month').agg({
                '충전소수': 'sum',
                '완속충전기': 'sum',
                '급속충전기': 'sum',
                '총충전기': 'sum'
            }).reset_index()
            
            monthly_trend = "\n월별 충전 인프라 추이:\n"
            for _, row in monthly_summary.iterrows():
                monthly_trend += f"- {row['snapshot_month']}: 충전소 {row['충전소수']}개, 완속 {row['완속충전기']}기, 급속 {row['급속충전기']}기, 총 {row['총충전기']}기\n"
        
        # GS차지비 월별 추이
        gs_trend = ""
        if 'CPO명' in range_data.columns:
            gs_monthly = range_data[range_data['CPO명'] == 'GS차지비'].sort_values('snapshot_month')
            print(f'🔍 GS차지비 월별 데이터: {len(gs_monthly)}개월 준비완료')
            
            if len(gs_monthly) > 0:
                gs_trend = "\nGS차지비 월별 추이:\n"
                for _, row in gs_monthly.iterrows():
                    # 안전한 데이터 추출
                    month = row.get('snapshot_month', 'N/A')
                    rank = row.get('순위', 'N/A')
                    stations = row.get('충전소수')
                    slow_chargers = row.get('완속충전기')
                    fast_chargers = row.get('급속충전기')
                    total_chargers = row.get('총충전기')
                    market_share = row.get('시장점유율')
                    
                    # 강화된 데이터 변환 함수
                    def safe_convert_int(val):
                        try:
                            if pd.isna(val) or val is None:
                                return 0
                            if isinstance(val, str) and (val.strip() == '' or val.strip() == '-' or val.strip().lower() == 'n/a'):
                                return 0
                            return int(float(val))
                        except (ValueError, TypeError):
                            return 0
                    
                    def safe_convert_float(val):
                        try:
                            if pd.isna(val) or val is None:
                                return 0.0
                            if isinstance(val, str) and (val.strip() == '' or val.strip() == '-' or val.strip().lower() == 'n/a'):
                                return 0.0
                            return float(val)
                        except (ValueError, TypeError):
                            return 0.0
                    
                    # 안전한 변환 적용
                    stations = safe_convert_int(stations)
                    slow_chargers = safe_convert_int(slow_chargers)
                    fast_chargers = safe_convert_int(fast_chargers)
                    total_chargers = safe_convert_int(total_chargers)
                    market_share_val = safe_convert_float(market_share)
                    market_share = f"{market_share_val:.1f}%"
                    
                    gs_trend += f"- {month}: 순위 {rank}위, 충전소 {stations}개, 완속충전기 {slow_chargers}기, 급속충전기 {fast_chargers}기, 총충전기 {total_chargers}기, 점유율 {market_share}\n"
            else:
                print('⚠️ GS차지비 데이터를 찾을 수 없습니다!')
                # CPO명 목록 확인
                if 'CPO명' in range_data.columns:
                    unique_cpos = range_data['CPO명'].unique()
                    print(f'   - 사용 가능한 CPO명: {list(unique_cpos)[:10]}...')  # 처음 10개만 표시
        
        # 상위 10개 CPO 월별 추이
        top10_trend = ""
        if 'CPO명' in range_data.columns and '총충전기' in range_data.columns:
            # 최신 월 기준 상위 10개 CPO 선정
            latest_month = available_months[-1] if available_months else target_month
            latest_data = range_data[range_data['snapshot_month'] == latest_month]
            top10_cpos = latest_data.nlargest(10, '총충전기')['CPO명'].tolist()
            
            top10_trend = "\n상위 10개 CPO 월별 추이:\n"
            for cpo in top10_cpos:
                cpo_data = range_data[range_data['CPO명'] == cpo].sort_values('snapshot_month')
                if len(cpo_data) > 0:
                    first_month = cpo_data.iloc[0]
                    last_month = cpo_data.iloc[-1]
                    growth = int(last_month.get('총충전기', 0)) - int(first_month.get('총충전기', 0))
                    top10_trend += f"- {cpo}: {first_month.get('snapshot_month', 'N/A')} {first_month.get('총충전기', 'N/A')}기 → {last_month.get('snapshot_month', 'N/A')} {last_month.get('총충전기', 'N/A')}기 (증감: {growth:+d}기)\n"
        
        prompt = f"""
<role>
당신은 한국 EV 충전 인프라 데이터를 기반으로 경영진이 보는 수준의 시각적·정리형 리포트를 작성하는 시니어 데이터 애널리스트이자 리포트 디자이너입니다.
</role>

<context>
- 데이터는 "전국의 CPO(충전사업자)별 월별 충전소 수, 완속/급속/총 충전기 수"에 한정됩니다.
- 우리 회사는 CPO 중 하나인 "GS차지비"이며, 리포트에서는 GS차지비의 위치와 특징을 항상 별도로 짚어줘야 합니다.
- 차량 수, 매출, 이용률, 충전건수, 고객 세그먼트 등 "지금 주어진 데이터로는 알 수 없는 항목"은 절대로 만들어내지 마십시오.
</context>

## ⚠️⚠️⚠️ 날짜 정보 - 반드시 이 날짜만 사용하세요! ⚠️⚠️⚠️

**분석 기간 시작:** {report_start_month}
**분석 기간 종료:** {report_end_month}
**기준월 (최신 데이터):** {report_target_month}
**분석 개월 수:** {report_period_count}개월

⚠️ 리포트에서 날짜를 언급할 때는 반드시 위의 날짜만 사용하세요.
⚠️ 절대로 2024-10, 2024-09 같은 다른 날짜를 만들거나 추측하지 마세요.

## 분석 기간: {report_start_month} ~ {report_end_month} ({report_period_count}개월)
## 기준월: {report_target_month}

## 월별 전국 충전 인프라 추이
{monthly_trend}

## GS차지비 월별 추이
{gs_trend}

## 상위 10개 CPO 월별 추이
{top10_trend}

---

**리포트 유형: EV Infra Monthly Trend Report**

이 리포트는 여러 달에 걸친 "월별 데이터"를 사용하여 국가 전체 및 GS차지비의 트렌드를 보는 시계열 리포트입니다.

**중요: 기준월은 {report_target_month}입니다. 최신 데이터를 언급할 때는 반드시 {report_target_month}를 사용하세요.**

**주요 내용:**
- 국가 전체 총 충전기/충전소의 월별 증가 추이 (표 + 추세 설명)
- 완속/급속 충전기의 월별 증감·성장률 요약
- GS차지비의 월별:
  - 총 충전기 수 / 충전소 수
  - 전체 시장 내 비중(점유율) 변화
- "전국 vs GS차지비" 성장 속도 비교

**날짜 표기 규칙:**
- 기준월(최신 데이터)을 언급할 때는 반드시 "{report_target_month}"를 사용하세요.
- 다른 날짜를 임의로 만들지 마세요.
- 위에 제공된 월별 데이터의 날짜만 사용하세요.

**중요: 리포트는 반드시 아래 HTML 타이틀로 정확히 시작해야 합니다:**

<div align="center">
<h1>🔍 EV Infra Trend Report</h1>
<p style="color: #666; font-size: 14px;">분석 기간: {report_start_month} ~ {report_end_month} ({report_period_count}개월) | 기준월: {report_target_month}</p>
</div>

**⚠️ 위 HTML을 그대로 복사하세요. 날짜를 절대 변경하지 마세요:**
- {report_start_month} (분석 시작)
- {report_end_month} (분석 종료)  
- {report_target_month} (기준월)

이 날짜들만 사용하고, 다른 날짜(예: 2024-10)를 절대 만들지 마세요.

**레이아웃 규칙:**

1) 제목 계층 (넘버링):
- H1: `# 1. National Infrastructure Trend`
- H2: `## 1.1 Monthly Charger Growth`
- H3: `### 1.1.1 Total Chargers`

3) 섹션별 콜아웃 (H1 바로 아래):
```
> 💡 **Key Insight**
> 
> 핵심 메시지 2문장
```

4) 월별 추이 표 형식:

| Month | Total Chargers | AC | DC | MoM Change | Growth Rate |
|-------|----------------|----|----|------------|-------------|
| 2024-12 | 450,000 | 380,000 | 70,000 | +5,000 | +1.1% |

5) GS차지비 트렌드 분석:
- 월별 점유율 변화
- 전국 대비 성장 속도 비교

**금지 사항:**
- 차트 제안 문장을 작성하지 마세요 (예: "📊 차트 제안:", "라인 차트로 시각화하면" 등)
- 시각화 관련 제안을 하지 마세요

**할루시네이션 방지:**
- 오직 충전소/충전기 수 데이터만 사용
- 전망 시 "~의 가능성이 있습니다" 같은 신중한 표현 사용
- 해석 시 반드시 수치 먼저 제시

**리포트 구조:**

# 1. National Infrastructure Trend
> 💡 콜아웃

## 1.1 Monthly Charger Growth
- 월별 총 충전기 추이 표
- 완속/급속 변화 분석
- 성장률 추이

## 1.2 Structural Changes
- 완속 vs 급속 비중 변화
- 계절성 패턴 (관찰된 패턴만)

# 2. GS차지비 Trend Analysis
> 💡 콜아웃

## 2.1 Monthly Performance
### 2.1.1 GS차지비 Infrastructure Status

**중요: 아래 표를 정확히 작성하세요. 모든 데이터는 위에 제공된 "GS차지비 월별 추이" 데이터에서 가져와야 합니다.**

| Month | Charging Stations | AC (완속) | DC (급속) | Total Chargers | Market Share | Rank |
|-------|------------------|-----------|-----------|----------------|--------------|------|
| 각 월별로 위 데이터에서 정확한 수치 입력 |

**데이터 입력 규칙:**
- Charging Stations: 충전소 수 (위 데이터의 "충전소 X개"에서 추출)
- AC (완속): 완속충전기 수 (위 데이터의 "완속충전기 X기"에서 추출)  
- DC (급속): 급속충전기 수 (위 데이터의 "급속충전기 X기"에서 추출)
- Total Chargers: 총충전기 수 (위 데이터의 "총충전기 X기"에서 추출)
- Market Share: 시장점유율 (위 데이터의 "점유율 X%"에서 추출)
- Rank: 순위 (위 데이터의 "순위 X위"에서 추출)

**⚠️⚠️⚠️ 중요: 데이터 표시 규칙 ⚠️⚠️⚠️**

1. **절대 "-", "N/A", "데이터 없음"을 표에 사용하지 마세요**
2. **위에 제공된 실제 숫자 데이터만 사용하세요**
3. **데이터가 0인 경우 "0"으로 표시하세요**
4. **모든 숫자는 쉼표로 구분하세요 (예: 1,234)**

**예시:**
위 데이터에 "- 2025-11: 순위 2위, 충전소 1234개, 완속충전기 5678기, 급속충전기 2345기, 총충전기 8023기, 점유율 15.2%"가 있다면:
| 2025-11 | 1,234 | 5,678 | 2,345 | 8,023 | 15.2% | 2 |

**데이터 추출 실패 시에도 "0"으로 표시하고, 절대 "-"나 "N/A"를 사용하지 마세요.**

- 점유율 변화 추이

## 2.2 Growth Comparison
- 전국 vs GS차지비 성장 속도
- 상위 경쟁사 대비 성장률

## 2.3 Outlook
- 데이터 기반 단기 전망 (신중한 표현)

한국어로 작성하되, HTML 타이틀과 표는 위 형식을 정확히 따라주세요.
"""
        
        context = self.retrieve_from_kb('충전 인프라 트렌드 시계열 분석')
        return self.invoke_bedrock(prompt, context)
    
    def generate_strategy_report(self, target_month, target_insights, range_insights, target_data, range_data, available_months):
        """Strategy Report 생성 - 경쟁력·전략 분석 중심"""
        print(f'🎯 Strategy Report 생성 중... (기준월: {target_month})', flush=True)
        
        # GS차지비 데이터 추출
        gs_target = target_data[target_data['CPO명'] == 'GS차지비'] if 'CPO명' in target_data.columns else None
        gs_info = ""
        if gs_target is not None and len(gs_target) > 0:
            gs_row = gs_target.iloc[0]
            slow_ratio = (int(gs_row.get('완속충전기', 0)) / int(gs_row.get('총충전기', 1)) * 100) if gs_row.get('총충전기', 0) > 0 else 0
            fast_ratio = (int(gs_row.get('급속충전기', 0)) / int(gs_row.get('총충전기', 1)) * 100) if gs_row.get('총충전기', 0) > 0 else 0
            
            gs_info = f"""
GS차지비 {target_month} 포지션:
- 순위: {gs_row.get('순위', 'N/A')}위
- 충전소 수: {gs_row.get('충전소수', 'N/A')}개
- 총충전기: {gs_row.get('총충전기', 'N/A')}기
- 완속충전기: {gs_row.get('완속충전기', 'N/A')}기 ({slow_ratio:.1f}%)
- 급속충전기: {gs_row.get('급속충전기', 'N/A')}기 ({fast_ratio:.1f}%)
- 시장점유율: {gs_row.get('시장점유율', 'N/A')}
- 전월 대비 증감: {gs_row.get('총증감', 'N/A')}기
"""
        
        # 경쟁사 벤치마킹 (상위 5개사)
        competitor_benchmark = ""
        if 'CPO명' in target_data.columns and '총충전기' in target_data.columns:
            top5 = target_data.nlargest(5, '총충전기')
            competitor_benchmark = f"\n상위 5개사 벤치마킹 ({target_month}):\n"
            for _, row in top5.iterrows():
                cpo_name = row.get('CPO명', 'N/A')
                total = int(row.get('총충전기', 0))
                slow = int(row.get('완속충전기', 0))
                fast = int(row.get('급속충전기', 0))
                slow_pct = (slow / total * 100) if total > 0 else 0
                fast_pct = (fast / total * 100) if total > 0 else 0
                
                competitor_benchmark += f"- {row.get('순위', 'N/A')}위. {cpo_name}: 총 {total}기 (완속 {slow}기 {slow_pct:.1f}%, 급속 {fast}기 {fast_pct:.1f}%), 점유율 {row.get('시장점유율', 'N/A')}, 증감 {row.get('총증감', 'N/A')}기\n"
        
        # GS차지비 월별 추이
        gs_trend = ""
        if 'CPO명' in range_data.columns:
            gs_monthly = range_data[range_data['CPO명'] == 'GS차지비'].sort_values('snapshot_month')
            if len(gs_monthly) > 0:
                gs_trend = "\nGS차지비 월별 추이:\n"
                for _, row in gs_monthly.iterrows():
                    gs_trend += f"- {row.get('snapshot_month', 'N/A')}: 순위 {row.get('순위', 'N/A')}위, 총충전기 {row.get('총충전기', 'N/A')}기, 점유율 {row.get('시장점유율', 'N/A')}, 증감 {row.get('총증감', 'N/A')}기\n"
        
        # 시장 구조 분석
        market_structure = ""
        summary = target_insights.get('summary', {})
        market_structure = f"""
전국 시장 구조 ({target_month}):
- 총 CPO 수: {summary.get('total_cpos', 'N/A')}개
- 총 충전기: {summary.get('total_chargers', 'N/A')}기
- 완속 비중: {summary.get('slow_ratio', 'N/A')}%
- 급속 비중: {summary.get('fast_ratio', 'N/A')}%
"""
        
        prompt = f"""
당신은 GS차지비의 전략 컨설턴트입니다. 다음 데이터를 바탕으로 **EV Infra Competitiveness & Positioning Report**를 작성해주세요.

이 리포트는 **경쟁력·전략 분석 중심**으로, GS차지비가 전국 시장에서 어디에 서 있으며, 어떤 전략적 방향성이 필요한지 제시하는 리포트입니다.

## 기준월: {target_month}
## 분석 기간: {available_months[0]} ~ {available_months[-1]}

## GS차지비 포지션
{gs_info}

## GS차지비 월별 추이
{gs_trend}

## 경쟁사 벤치마킹
{competitor_benchmark}

## 시장 구조
{market_structure}

---

**중요: 할루시네이션 방지 규칙**
1. 절대로 제공되지 않은 지표(매출, 이용건수, kWh, 충전시간, 고객 수 등)를 언급하지 마세요
2. "시장 구조", "성장성" 등 정성적 코멘트는 반드시 **충전소/충전기 수와 성장률, 비중**만을 근거로 해석하세요
3. 데이터에 없는 내용은 "데이터 상 확인 불가" 또는 아예 언급하지 마세요
4. 전략 제안은 데이터 기반의 합리적 추론만 하세요

**리포트 구성:**

## 1. GS차지비 Position Overview
- 충전기 총량 & 충전소 수
- 완속·급속 비율
- 시장점유율(MS)
- 업계 내 순위

## 2. Benchmarking Against Competitors (경쟁사 벤치마킹)
- 상위 5개사 비교 (표 형식)
  - 충전기 수
  - 성장률
  - 급속 비중
  - 충전소당 기기수
- GS차지비 차별화 포지션

## 3. Market Share Competitiveness
- 점유율 추이
- Top 3 대비 격차
- 경쟁 압력 분석

## 4. Strategic Risk & Opportunity Analysis
**핵심 분석:**
- 완속 중심 인프라의 구조조정 리스크
- 급속 충전 시장 성장 대비 대응도
- 경쟁사의 급속 확대 정책이 MS에 미치는 영향
- 전국 완속 감소세와 GS차지비 전략 정합성

## 5. Strategic Recommendation (전략 제안)
**데이터 기반 전략:**
- 급속 확대 vs 선택적 투자
- 완속 운영 효율화
- 경쟁사 대비 차별화 포인트 강화
- 내부 KPI 제안 (예: 급속 비중, 설치/철거 기준 등)

**Markdown 포맷팅 규칙:**
- H2: "## 1. GS차지비 Position Overview" (제목 다음 줄은 빈 줄)
- H3: "### 1.1 현재 포지션" (제목 다음 줄은 빈 줄)
- 글머리 기호: "- " 사용
- 표: GitHub-style Markdown table (| 항목 | 값 |)
- 순수 Markdown만 사용 (HTML, LaTeX 금지)

한국어로 작성해주세요.
"""
        
        context = self.retrieve_from_kb('GS차지비 경쟁력 전략 분석')
        return self.invoke_bedrock(prompt, context)

    def generate_ai_simulation(self, base_month, simulation_months, additional_chargers, full_data, target_data):
        """AI 기반 시장점유율 시뮬레이션 예측 - RAG 데이터 기반"""
        import time
        print(f'\n🎯 AI 시뮬레이션 예측 시작 (RAG 기반)', flush=True)
        print(f'   ├─ 기준월: {base_month}', flush=True)
        print(f'   ├─ 예측 기간: {simulation_months}개월', flush=True)
        print(f'   └─ 추가 충전기: {additional_chargers:,}대', flush=True)
        
        start_time = time.time()
        
        # 1. RAG - Knowledge Base에서 모든 과거 데이터 검색
        print(f'   📚 RAG: Knowledge Base에서 과거 데이터 검색 중...', flush=True)
        
        # 여러 쿼리로 RAG 데이터 수집
        rag_queries = [
            f'충전인프라 현황 {base_month} GS차지비 시장점유율 충전기',
            f'전기차 충전사업자 순위 충전소 현황 {base_month}',
            f'GS차지비 충전기 증감 추이 시장점유율 변화',
            f'충전인프라 시장 성장률 경쟁사 분석'
        ]
        
        rag_context_parts = []
        for query in rag_queries:
            ctx = self.retrieve_from_kb(query)
            if ctx:
                rag_context_parts.append(ctx)
        
        rag_context = "\n\n---\n\n".join(rag_context_parts) if rag_context_parts else ""
        print(f'   📚 RAG 컨텍스트 수집 완료: {len(rag_context):,}자', flush=True)
        
        # 2. 메모리 데이터에서 모든 과거 데이터 수집 (기준월 이전 모든 데이터)
        all_months = sorted(full_data['snapshot_month'].unique().tolist())
        available_months = [m for m in all_months if m <= base_month]
        
        print(f'   📅 분석 기간: {available_months[0]} ~ {available_months[-1]} ({len(available_months)}개월)', flush=True)
        
        # 3. GS차지비 전체 히스토리 데이터 추출
        gs_data = full_data[full_data['CPO명'] == 'GS차지비'].copy()
        gs_history = gs_data[gs_data['snapshot_month'].isin(available_months)].sort_values('snapshot_month')
        
        # GS차지비 월별 추이 데이터
        gs_trend_data = []
        for _, row in gs_history.iterrows():
            gs_trend_data.append({
                'month': row.get('snapshot_month'),
                'rank': int(row.get('순위', 0)) if pd.notna(row.get('순위')) else None,
                'stations': int(row.get('충전소수', 0)) if pd.notna(row.get('충전소수')) else None,
                'slow_chargers': int(row.get('완속충전기', 0)) if pd.notna(row.get('완속충전기')) else None,
                'fast_chargers': int(row.get('급속충전기', 0)) if pd.notna(row.get('급속충전기')) else None,
                'total_chargers': int(row.get('총충전기', 0)) if pd.notna(row.get('총충전기')) else None,
                'market_share': float(row.get('시장점유율', 0)) * 100 if pd.notna(row.get('시장점유율')) and row.get('시장점유율') < 1 else float(row.get('시장점유율', 0)) if pd.notna(row.get('시장점유율')) else None,
                'total_change': int(row.get('총증감', 0)) if pd.notna(row.get('총증감')) else None
            })
        
        # 4. 전체 시장 데이터 추출
        market_data = []
        for month in available_months:
            month_data = full_data[full_data['snapshot_month'] == month]
            if len(month_data) > 0:
                total_chargers = month_data['총충전기'].sum()
                total_cpos = len(month_data[month_data['총충전기'] > 0])
                market_data.append({
                    'month': month,
                    'total_chargers': int(total_chargers),
                    'total_cpos': int(total_cpos)
                })
        
        # 5. 경쟁사 현황 (상위 10개사)
        current_data = full_data[full_data['snapshot_month'] == base_month]
        top10 = current_data.nlargest(10, '총충전기') if '총충전기' in current_data.columns else current_data.head(10)
        
        competitor_info = []
        for _, row in top10.iterrows():
            competitor_info.append({
                'name': row.get('CPO명', 'N/A'),
                'rank': int(row.get('순위', 0)) if pd.notna(row.get('순위')) else None,
                'total_chargers': int(row.get('총충전기', 0)) if pd.notna(row.get('총충전기')) else None,
                'market_share': float(row.get('시장점유율', 0)) * 100 if pd.notna(row.get('시장점유율')) and row.get('시장점유율') < 1 else float(row.get('시장점유율', 0)) if pd.notna(row.get('시장점유율')) else None,
                'total_change': int(row.get('총증감', 0)) if pd.notna(row.get('총증감')) else None
            })
        
        # 6. 미래 월 계산
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        base_date = datetime.strptime(base_month, '%Y-%m')
        future_months = []
        for i in range(1, simulation_months + 1):
            future_date = base_date + relativedelta(months=i)
            future_months.append(future_date.strftime('%Y-%m'))
        
        # 7. AI 프롬프트 생성
        gs_trend_str = "\n".join([
            f"- {d['month']}: 순위 {d['rank']}위, 총충전기 {d['total_chargers']:,}기, 시장점유율 {d['market_share']:.2f}%, 월증감 {d['total_change']:+,}기"
            for d in gs_trend_data if d['total_chargers']
        ])
        
        market_trend_str = "\n".join([
            f"- {d['month']}: 전체 충전기 {d['total_chargers']:,}기, CPO 수 {d['total_cpos']}개"
            for d in market_data
        ])
        
        competitor_str = "\n".join([
            f"- {c['name']}: 순위 {c['rank']}위, 총충전기 {c['total_chargers']:,}기, 시장점유율 {c['market_share']:.2f}%, 월증감 {c['total_change']:+,}기"
            for c in competitor_info if c['total_chargers']
        ])
        
        # 현재 GS차지비 상태
        current_gs = gs_trend_data[-1] if gs_trend_data else {}
        
        # 미래 월 목록
        future_months_str = ", ".join(future_months)
        
        prompt = f"""당신은 한국 전기차 충전 인프라 시장 분석 전문가입니다.
아래 제공된 RAG(검색 증강 생성) 데이터와 과거 실적 데이터를 기반으로 GS차지비의 미래 시장점유율을 예측해주세요.

## 📊 RAG 참조 데이터 (Knowledge Base에서 검색된 실제 데이터)
{rag_context if rag_context else "RAG 데이터 없음 - 아래 과거 실적 데이터만 사용"}

---

## 🎯 시뮬레이션 조건
- 기준월: {base_month}
- 예측 기간: {simulation_months}개월
- 예측 대상 월: {future_months_str}
- GS차지비 추가 설치 계획: {additional_chargers:,}대 (예측 기간 동안 균등 배분, 월 {additional_chargers // simulation_months if simulation_months > 0 else 0:,}대)

## 📈 GS차지비 현재 상태 ({base_month})
- 순위: {current_gs.get('rank', 'N/A')}위
- 총충전기: {current_gs.get('total_chargers', 0):,}기
- 시장점유율: {current_gs.get('market_share', 0):.2f}%

## 📅 GS차지비 전체 과거 실적 ({len(gs_trend_data)}개월)
{gs_trend_str}

## 🌐 전체 시장 추이
{market_trend_str}

## 🏆 경쟁사 현황 (상위 10개사, {base_month} 기준)
{competitor_str}

---

## 🤖 AI 분석 요청

위의 RAG 데이터와 과거 실적을 종합 분석하여 다음을 예측해주세요:

1. **과거 데이터 패턴 분석**
   - 시장 전체 월평균 성장률 계산 (과거 데이터 기반)
   - GS차지비 월평균 성장률 계산 (과거 데이터 기반)
   - 계절성, 트렌드 등 패턴 식별

2. **미래 예측 ({simulation_months}개월)**
   - 기준선 시나리오: 현재 추세 유지 시 각 월별 시장점유율 예측
   - 투자 시나리오: {additional_chargers:,}대 추가 설치 시 각 월별 시장점유율 예측

3. **전략적 인사이트**
   - 투자 효과 분석
   - 리스크 요인
   - 권고사항

## 📋 응답 형식 (반드시 아래 JSON 형식으로만 응답)

```json
{{
    "analysis": {{
        "market_monthly_growth_rate": 시장 월평균 성장률 숫자 (예: 1.5),
        "gs_monthly_growth_rate": GS차지비 월평균 성장률 숫자 (예: 0.8),
        "market_trend": "시장 트렌드 분석 요약 (2-3문장)",
        "competition_analysis": "경쟁 환경 분석 (2-3문장)"
    }},
    "current_status": {{
        "market_share": {current_gs.get('market_share', 0):.2f},
        "total_chargers": {current_gs.get('total_chargers', 0)},
        "rank": {current_gs.get('rank', 0)}
    }},
    "baseline_prediction": {{
        "final_market_share": 최종 시장점유율 숫자,
        "final_total_chargers": 최종 충전기 수 숫자,
        "monthly_predictions": [
            {{"month": "{future_months[0] if future_months else 'YYYY-MM'}", "market_share": 숫자, "total_chargers": 숫자}},
            ... (총 {simulation_months}개월 모두 포함)
        ]
    }},
    "investment_prediction": {{
        "final_market_share": 최종 시장점유율 숫자,
        "final_total_chargers": 최종 충전기 수 숫자,
        "market_share_increase": 기준선 대비 증가분 숫자,
        "monthly_predictions": [
            {{"month": "{future_months[0] if future_months else 'YYYY-MM'}", "market_share": 숫자, "total_chargers": 숫자}},
            ... (총 {simulation_months}개월 모두 포함)
        ]
    }},
    "insights": {{
        "key_findings": ["주요 발견 1", "주요 발견 2", "주요 발견 3"],
        "risks": ["리스크 1", "리스크 2"],
        "recommendations": ["권고사항 1", "권고사항 2", "권고사항 3"]
    }},
    "confidence_level": "HIGH 또는 MEDIUM 또는 LOW",
    "confidence_reason": "신뢰도 판단 근거 (1-2문장)"
}}
```

**⚠️ 중요 지침:**
1. 반드시 제공된 RAG 데이터와 과거 실적만 사용하여 분석하세요
2. monthly_predictions는 정확히 {simulation_months}개월 모두 포함해야 합니다
3. 시장점유율은 소수점 2자리까지 표시 (예: 16.25)
4. JSON 형식 외의 텍스트는 절대 포함하지 마세요
5. 모든 숫자는 따옴표 없이 숫자 타입으로 작성하세요
"""
        
        # 8. Bedrock 호출
        print(f'   🤖 AI 예측 모델 호출 중... (Bedrock Claude Sonnet 4.5)', flush=True)
        
        try:
            payload = {
                'anthropic_version': Config.ANTHROPIC_VERSION,
                'max_tokens': 8192,
                'temperature': 0.2,
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
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
            result_text = response_body['content'][0]['text']
            
            elapsed_time = time.time() - start_time
            print(f'   ✅ AI 예측 완료 (⏱️ {elapsed_time:.2f}초)', flush=True)
            
            # 8. JSON 파싱
            # JSON 블록 추출
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON 블록이 없으면 전체 텍스트를 JSON으로 파싱 시도
                json_str = result_text
            
            prediction_result = json.loads(json_str)
            prediction_result['bedrock_time'] = round(elapsed_time, 2)
            prediction_result['total_time'] = round(elapsed_time, 2)
            prediction_result['simulation_months'] = simulation_months
            prediction_result['additional_chargers'] = additional_chargers
            prediction_result['base_month'] = base_month
            
            # 히스토리 데이터 추가 (차트용)
            prediction_result['history'] = gs_trend_data
            
            return {
                'success': True,
                'prediction': prediction_result
            }
            
        except json.JSONDecodeError as e:
            print(f'   ❌ JSON 파싱 오류: {e}', flush=True)
            print(f'   📝 원본 응답: {result_text[:500]}...', flush=True)
            return {
                'success': False,
                'error': f'AI 응답 파싱 오류: {str(e)}',
                'raw_response': result_text
            }
        except Exception as e:
            print(f'   ❌ AI 예측 오류: {e}', flush=True)
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
