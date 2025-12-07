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
            print(f'🔍 Knowledge Base 검색 시작: "{query}"', flush=True)
            
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
            print(f'📊 검색 결과 개수: {len(results)}', flush=True)
            
            if len(results) == 0:
                print('⚠️ Knowledge Base에서 검색 결과가 없습니다', flush=True)
                return ''
            
            # 각 결과의 메타데이터와 내용 로깅
            for i, r in enumerate(results):
                score = r.get('score', 0)
                location = r.get('location', {})
                s3_location = location.get('s3Location', {})
                uri = s3_location.get('uri', 'N/A')
                content_preview = r.get('content', {}).get('text', '')[:200]
                print(f'  [{i+1}] Score: {score:.4f}, URI: {uri}', flush=True)
                print(f'      Preview: {content_preview}...', flush=True)
            
            context = '\n\n'.join([
                f"[참고자료 {i+1}] (관련도: {r.get('score', 0):.2f})\n출처: {r.get('location', {}).get('s3Location', {}).get('uri', 'N/A')}\n\n{r.get('content', {}).get('text', '')}"
                for i, r in enumerate(results)
            ])
            
            print(f'✅ Knowledge Base 컨텍스트 생성 완료 (총 {len(context)} 자)', flush=True)
            return context
        
        except Exception as e:
            print(f'❌ Knowledge Base 검색 오류: {e}', flush=True)
            import traceback
            traceback.print_exc()
            return ''
    
    def invoke_bedrock(self, prompt, context=''):
        """Bedrock 모델 호출 (리포트 생성용)"""
        import time
        try:
            print(f'🔄 Bedrock 모델 호출 시작...', flush=True)
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
            print(f'✅ Bedrock 응답 완료 ({len(result)} 자, ⏱️ {elapsed_time:.2f}초)', flush=True)
            return result
        
        except Exception as e:
            print(f'❌ Bedrock 호출 오류: {e}', flush=True)
            import traceback
            traceback.print_exc()
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
                competitor_info += f"- {row.get('CPO명', 'N/A')}: 순위 {row.get('순위', 'N/A')}위, 총충전기 {row.get('총충전기', 'N/A')}기, 시장점유율 {row.get('시장점유율', 'N/A')}, 총증감 {row.get('총증감', 'N/A')}\n"
        
        # 1. 경영진 요약 (GS차지비 관점)
        print('📝 [1/3] GS차지비 경영진 요약 생성 중...', flush=True)
        start_time = time.time()
        report['executive_summary'] = self._generate_gs_executive_summary(
            target_month, gs_info, gs_trend, competitor_info, target_insights, available_months
        )
        report['response_times']['executive_summary'] = round(time.time() - start_time, 2)
        print(f'✅ [1/3] 경영진 요약 완료 (⏱️ {report["response_times"]["executive_summary"]}초)', flush=True)
        
        # 2. 경쟁 분석 (GS차지비 관점)
        print('📝 [2/3] GS차지비 경쟁 분석 생성 중...', flush=True)
        start_time = time.time()
        report['cpo_analysis'] = self._generate_gs_competitor_analysis(
            target_month, gs_info, gs_trend, competitor_info, target_insights, range_insights
        )
        report['response_times']['cpo_analysis'] = round(time.time() - start_time, 2)
        print(f'✅ [2/3] 경쟁 분석 완료 (⏱️ {report["response_times"]["cpo_analysis"]}초)', flush=True)
        
        # 3. 전략 제안 (GS차지비 관점)
        print('📝 [3/3] GS차지비 전략 제안 생성 중...', flush=True)
        start_time = time.time()
        report['trend_forecast'] = self._generate_gs_strategy(
            target_month, gs_info, gs_trend, competitor_info, range_insights, available_months
        )
        report['response_times']['trend_forecast'] = round(time.time() - start_time, 2)
        print(f'✅ [3/3] 전략 제안 완료 (⏱️ {report["response_times"]["trend_forecast"]}초)', flush=True)
        
        total_time = sum(report['response_times'].values())
        print(f'✅ GS차지비 AI 리포트 생성 완료 (총 ⏱️ {total_time:.2f}초)\n', flush=True)
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
            for idx, row in top10.iterrows():
                top10_info += f"- {row.get('순위', 'N/A')}위. {row.get('CPO명', 'N/A')}: 총충전기 {row.get('총충전기', 'N/A')}기 (완속 {row.get('완속충전기', 'N/A')}, 급속 {row.get('급속충전기', 'N/A')}), 점유율 {row.get('시장점유율', 'N/A')}, 전월 대비 {row.get('총증감', 'N/A')}기\n"
        
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

## 기준월: {target_month}

## 전국 충전 인프라 현황
{market_summary}

## GS차지비 현황
{gs_info}

## 상위 10개 CPO
{top10_info}

---

**리포트 유형: EV Infra KPI Snapshot Report**

이 리포트는 한 기준월({target_month})을 기준으로 전국 EV 충전 인프라의 "단면(KPI 스냅샷)"을 정리하는 리포트입니다.

**주요 내용:**
- 국가 전체 총 충전소 수, 완속/급속/총 충전기 수 (KPI)
- 전월 대비 증감 수치
- 상위 5~10개 CPO의 충전기/충전소 규모 비교 표
- 상위 CPO 집중도(Top N 비중) 요약
- GS차지비의 순위, 규모, 전체 대비 비중 요약

**중요: 리포트는 반드시 아래 HTML 타이틀로 시작해야 합니다. 절대로 수정하지 마세요:**

<div align="center">
<h1>📊 EV Infra KPI Report</h1>
<p style="color: #666; font-size: 14px;">분석 기간: {available_months[0]} ~ {available_months[-1]} ({len(available_months)}개월) | 기준월: {target_month}</p>
</div>

**경고: 위 HTML 타이틀의 분석 기간을 절대로 변경하지 마세요. 그대로 복사해서 사용하세요.**

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

| Rank | CPO | Chargers | Sites | Share | MoM Change |
|------|------------|----------|-------|--------|------------|
| 1 | GS차지비 | 73,290 | 13,173| 16.2% | +120 |

5) GS차지비 강조:
- 표에서 GS차지비 행에 별도 코멘트
- 별도 bullet에서 GS차지비만 따로 분석

**할루시네이션 방지:**
- 오직 "충전소 수, 완속/급속/총 충전기 수" 데이터에서 계산 가능한 사실만 사용
- 전망/해석 시 반드시 관련 수치를 먼저 제시한 뒤 "이 수치를 기반으로 볼 때 ~로 해석될 수 있습니다."와 같이 추론임을 명시

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

## 기준월: {target_month}

## 전국 시장 구조
{market_structure}

## GS차지비 상세
{gs_info}

## 상위 15개 CPO 상세
{top15_detail}

---

**리포트 유형: EV Infra CPO Ranking & GS차지비 Positioning Report**

이 리포트는 같은 기준월({target_month})을 기준으로 CPO별 순위/규모/구조를 상세히 보는 "사업자 관점" 리포트입니다.

**주요 내용:**
- CPO별 충전기 수 기준 랭킹 표 (Top 10~15)
- 각 CPO별 "충전소당 평균 충전기 수"(total_count / station_count) 비교
- 완속/급속 비중 요약
- GS차지비 중심 분석:
  - 시장 내 순위
  - 상위 그룹과의 격차
  - 완속/급속 구조 특징
  - "Efficiency vs Coverage" 축으로 주요 경쟁사와 GS차지비 포지션 비교

**중요: 리포트는 반드시 아래 HTML 타이틀로 시작해야 합니다. 절대로 수정하지 마세요:**

<div align="center">
<h1>📈 EV Infra CPO Market Report</h1>
<p style="color: #666; font-size: 14px;">분석 기간: {available_months[0]} ~ {available_months[-1]} ({len(available_months)}개월) | 기준월: {target_month}</p>
</div>

**경고: 위 HTML 타이틀의 분석 기간을 절대로 변경하지 마세요. 그대로 복사해서 사용하세요.**

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
            if len(gs_monthly) > 0:
                gs_trend = "\nGS차지비 월별 추이:\n"
                for _, row in gs_monthly.iterrows():
                    gs_trend += f"- {row.get('snapshot_month', 'N/A')}: 순위 {row.get('순위', 'N/A')}위, 총충전기 {row.get('총충전기', 'N/A')}기, 점유율 {row.get('시장점유율', 'N/A')}\n"
        
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

## 분석 기간: {available_months[0]} ~ {available_months[-1]} ({len(available_months)}개월)
## 기준월: {target_month}

## 월별 전국 충전 인프라 추이
{monthly_trend}

## GS차지비 월별 추이
{gs_trend}

## 상위 10개 CPO 월별 추이
{top10_trend}

---

**리포트 유형: EV Infra Monthly Trend Report**

이 리포트는 여러 달에 걸친 "월별 데이터"를 사용하여 국가 전체 및 GS차지비의 트렌드를 보는 시계열 리포트입니다.

**주요 내용:**
- 국가 전체 총 충전기/충전소의 월별 증가 추이 (표 + 추세 설명)
- 완속/급속 충전기의 월별 증감·성장률 요약
- GS차지비의 월별:
  - 총 충전기 수 / 충전소 수
  - 전체 시장 내 비중(점유율) 변화
- "전국 vs GS차지비" 성장 속도 비교

**중요: 리포트는 반드시 아래 HTML 타이틀로 시작해야 합니다. 절대로 수정하지 마세요:**

<div align="center">
<h1>🔍 EV Infra Trend Report</h1>
<p style="color: #666; font-size: 14px;">분석 기간: {available_months[0]} ~ {available_months[-1]} ({len(available_months)}개월) | 기준월: {target_month}</p>
</div>

**경고: 위 HTML 타이틀의 분석 기간을 절대로 변경하지 마세요. 그대로 복사해서 사용하세요.**

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

5) 차트 제안:
- "월별 total chargers 추이를 라인 차트로 시각화하면 감소 전환 시점을 쉽게 확인할 수 있습니다."

6) GS차지비 트렌드 분석:
- 월별 점유율 변화
- 전국 대비 성장 속도 비교

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
- GS차지비 월별 충전기/충전소 수 표
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
