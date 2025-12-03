"""
커스텀 질의 분석 및 차트 생성 모듈
- RAG 연동
- 프롬프트 엔지니어링
- 코드 인터프리터 연동
"""
import json
import re
import boto3
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
            if not results:
                return ''
            
            context = '\n\n'.join([
                f"[참고자료 {i+1}] (관련도: {r.get('score', 0):.2f})\n{r.get('content', {}).get('text', '')}"
                for i, r in enumerate(results)
            ])
            
            return context
        except Exception as e:
            print(f'❌ KB 검색 오류: {e}')
            return ''
    
    def analyze_query_intent(self, query: str, available_data: dict) -> dict:
        """질의 의도 분석 - 차트 필요 여부 판단"""
        
        analysis_prompt = f"""
당신은 데이터 분석 질의를 분석하는 전문가입니다.

## 사용자 질의
{query}

## 사용 가능한 데이터
- 사용 가능한 월: {available_data.get('available_months', [])}
- 사용 가능한 CPO: {available_data.get('available_cpos', [])[:20]}... (상위 20개)
- 사용 가능한 컬럼: {available_data.get('available_columns', [])}

## 분석 작업
다음 JSON 형식으로 질의를 분석해주세요:

```json
{{
    "needs_chart": true/false,
    "chart_type": "line" | "bar" | "pie" | "area" | null,
    "chart_title": "차트 제목",
    "data_filter": {{
        "cpo_name": "CPO명 또는 null",
        "start_month": "YYYY-MM 또는 null",
        "end_month": "YYYY-MM 또는 null",
        "column": "조회할 컬럼명 (단일) 또는 [컬럼1, 컬럼2] (다중 비교)"
    }},
    "analysis_type": "trend" | "comparison" | "ranking" | "single",
    "explanation": "분석 설명"
}}
```

## 중요: 다중 컬럼 비교
- 사용자가 "완속충전기와 급속충전기를 비교", "두 가지를 하나의 그래프로" 등 요청 시
- column을 배열로 지정: ["완속증감", "급속증감"] 또는 ["완속충전기", "급속충전기"]

## 차트 타입 결정 기준
- line: 시간에 따른 추이, 트렌드 분석, 다중 시리즈 비교
- bar: 항목별 비교, 순위
- pie: 비율, 점유율
- area: 누적 추이

## 컬럼명 매핑
- 완속충전기, 완속: "완속충전기"
- 급속충전기, 급속: "급속충전기"
- 총충전기, 전체충전기, TTL: "총충전기"
- 충전소, 충전소수: "충전소수"
- 시장점유율, 점유율: "시장점유율"
- 완속증감, 완속 증감량: "완속증감"
- 급속증감, 급속 증감량: "급속증감"
- 총증감: "총증감"

JSON만 출력하세요.
"""
        
        try:
            payload = {
                'anthropic_version': Config.ANTHROPIC_VERSION,
                'max_tokens': 1024,
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
    
    def extract_chart_data(self, df, intent: dict) -> dict:
        """DataFrame에서 차트 데이터 추출"""
        try:
            data_filter = intent.get('data_filter', {})
            cpo_name = data_filter.get('cpo_name')
            start_month = data_filter.get('start_month')
            end_month = data_filter.get('end_month')
            column = data_filter.get('column', '총충전기')
            
            # 컬럼명 정규화 함수
            def normalize_column(col):
                if col is None:
                    return '총충전기'
                column_mapping = {
                    '완속': '완속충전기',
                    '완속증감': '완속증감',
                    '급속': '급속충전기',
                    '급속증감': '급속증감',
                    '총': '총충전기',
                    '총증감': '총증감',
                    '충전소': '충전소수',
                    '충전소증감': '충전소증감',
                    '점유율': '시장점유율'
                }
                for key, val in column_mapping.items():
                    if key in str(col):
                        return val
                return col
            
            # 다중 컬럼 지원 (리스트인 경우)
            columns = []
            if isinstance(column, list):
                columns = [normalize_column(c) for c in column]
            else:
                columns = [normalize_column(column)]
            
            # 데이터 필터링
            filtered_df = df.copy()
            
            # CPO 필터
            if cpo_name and 'CPO명' in filtered_df.columns:
                # 부분 매칭 지원
                mask = filtered_df['CPO명'].str.contains(cpo_name, case=False, na=False)
                filtered_df = filtered_df[mask]
            
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
            
            # 다중 시리즈 지원 (여러 컬럼 비교)
            if len(columns) > 1:
                # 다중 컬럼 시계열 차트
                if 'snapshot_month' in filtered_df.columns:
                    result = {'labels': [], 'series': [], 'multi_series': True}
                    
                    for col in columns:
                        if col in filtered_df.columns:
                            grouped = filtered_df.groupby('snapshot_month')[col].sum().reset_index()
                            grouped = grouped.sort_values('snapshot_month')
                            
                            if not result['labels']:
                                result['labels'] = grouped['snapshot_month'].tolist()
                            
                            result['series'].append({
                                'name': col,
                                'values': grouped[col].tolist()
                            })
                    
                    return result
            
            # 단일 컬럼
            col = columns[0]
            
            if analysis_type == 'trend':
                # 시간별 추이
                if 'snapshot_month' in filtered_df.columns and col in filtered_df.columns:
                    grouped = filtered_df.groupby('snapshot_month')[col].sum().reset_index()
                    grouped = grouped.sort_values('snapshot_month')
                    return {
                        'labels': grouped['snapshot_month'].tolist(),
                        'values': grouped[col].tolist()
                    }
            
            elif analysis_type == 'comparison':
                # 항목별 비교
                if 'CPO명' in filtered_df.columns and col in filtered_df.columns:
                    latest_month = filtered_df['snapshot_month'].max()
                    latest_df = filtered_df[filtered_df['snapshot_month'] == latest_month]
                    top_df = latest_df.nlargest(10, col)
                    return {
                        'labels': top_df['CPO명'].tolist(),
                        'values': top_df[col].tolist()
                    }
            
            elif analysis_type == 'ranking':
                # 순위
                if 'CPO명' in filtered_df.columns and col in filtered_df.columns:
                    latest_month = filtered_df['snapshot_month'].max()
                    latest_df = filtered_df[filtered_df['snapshot_month'] == latest_month]
                    top_df = latest_df.nlargest(10, col)
                    return {
                        'labels': top_df['CPO명'].tolist(),
                        'values': top_df[col].tolist()
                    }
            
            # 기본: 시간별 추이
            if 'snapshot_month' in filtered_df.columns and col in filtered_df.columns:
                grouped = filtered_df.groupby('snapshot_month')[col].sum().reset_index()
                grouped = grouped.sort_values('snapshot_month')
                return {
                    'labels': grouped['snapshot_month'].tolist(),
                    'values': grouped[col].tolist()
                }
            
            return {'labels': [], 'values': [], 'error': '데이터 추출 실패'}
            
        except Exception as e:
            print(f'❌ 데이터 추출 오류: {e}')
            return {'labels': [], 'values': [], 'error': str(e)}
    
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
                                    chart_data: dict, chart_result: dict) -> str:
        """차트와 함께 답변 생성"""
        
        # 다중 시리즈 여부 확인
        is_multi_series = chart_data.get('multi_series', False)
        
        if is_multi_series:
            # 다중 시리즈 데이터 요약
            series_info = []
            for s in chart_data.get('series', []):
                values = s.get('values', [])
                if values:
                    series_info.append(f"- {s['name']}: 최소 {min(values):,}, 최대 {max(values):,}")
            
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
                row_values = [label] + [f"{s['values'][i]:,}" for s in series_list]
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
            return response_body['content'][0]['text']
            
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def process_query(self, query: str, df, full_df) -> dict:
        """전체 질의 처리 파이프라인"""
        print(f'\n🔍 질의 처리 시작: "{query}"', flush=True)
        
        # 1. 사용 가능한 데이터 정보 수집
        available_data = {
            'available_months': sorted(full_df['snapshot_month'].unique().tolist()) if 'snapshot_month' in full_df.columns else [],
            'available_cpos': full_df['CPO명'].unique().tolist() if 'CPO명' in full_df.columns else [],
            'available_columns': list(full_df.columns)
        }
        print(f'📊 사용 가능한 월: {len(available_data["available_months"])}개', flush=True)
        
        # 2. RAG - Knowledge Base 검색
        print(f'📚 Knowledge Base 검색 중...', flush=True)
        kb_context = self.retrieve_from_kb(query)
        print(f'📊 KB 컨텍스트: {len(kb_context)} 자', flush=True)
        
        # 3. 질의 의도 분석 (프롬프트 엔지니어링)
        print(f'🧠 질의 의도 분석 중...', flush=True)
        intent = self.analyze_query_intent(query, available_data)
        print(f'📊 분석 결과: needs_chart={intent.get("needs_chart")}, type={intent.get("chart_type")}', flush=True)
        
        # 4. 차트 필요 여부에 따른 처리
        if intent.get('needs_chart'):
            print(f'📈 차트 데이터 추출 중...', flush=True)
            
            # 차트 데이터 추출
            chart_data = self.extract_chart_data(full_df, intent)
            
            if chart_data.get('error'):
                return {
                    'success': False,
                    'error': chart_data['error'],
                    'has_chart': False
                }
            
            print(f'📊 데이터 포인트: {len(chart_data.get("values", []))}개', flush=True)
            
            # 5. 코드 인터프리터로 차트 생성
            print(f'🎨 차트 생성 중...', flush=True)
            chart_result = self.generate_chart(intent, chart_data)
            
            if not chart_result.get('success'):
                print(f'⚠️ 차트 생성 실패: {chart_result.get("error")}', flush=True)
                # 차트 실패해도 텍스트 답변은 생성
                chart_result = {'success': False, 'image': None}
            else:
                print(f'✅ 차트 생성 완료', flush=True)
            
            # 6. 답변 생성
            print(f'💬 답변 생성 중...', flush=True)
            answer = self.generate_answer_with_chart(
                query, df, kb_context, intent, chart_data, chart_result
            )
            
            return {
                'success': True,
                'query': query,
                'answer': answer,
                'has_chart': chart_result.get('success', False),
                'chart_image': chart_result.get('image'),
                'chart_type': intent.get('chart_type'),
                'chart_title': intent.get('chart_title'),
                'data_summary': {
                    'labels': chart_data.get('labels', []),
                    'values': chart_data.get('values', []),
                    'count': len(chart_data.get('values', []))
                }
            }
        
        else:
            # 차트 불필요 - 기존 텍스트 답변만
            print(f'💬 텍스트 답변 생성 중...', flush=True)
            return {
                'success': True,
                'query': query,
                'answer': None,  # 기존 로직 사용
                'has_chart': False,
                'use_legacy': True  # 기존 custom_query 로직 사용 플래그
            }
