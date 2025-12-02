"""
충전 인프라 분석 리포트 웹앱
"""
from flask import Flask, render_template, jsonify, request
import json
from data_loader import ChargingDataLoader
from data_analyzer import ChargingDataAnalyzer
from ai_report_generator import AIReportGenerator

app = Flask(__name__)

# 전역 캐시
cache = {
    'data': None,
    'insights': None,
    'report': None
}

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/months')
def get_available_months():
    """S3 파일 목록에서 사용 가능한 기준월 빠르게 조회"""
    try:
        loader = ChargingDataLoader()
        files = loader.list_available_files()
        
        # 파일명에서 기준월 추출 (데이터 로드 없이)
        months = []
        for f in files:
            filename = f['filename']
            snapshot_date, snapshot_month = loader.parse_snapshot_date_from_filename(filename)
            if snapshot_month:
                months.append(snapshot_month)
        
        # 중복 제거 및 정렬 (최신순)
        unique_months = sorted(list(set(months)), reverse=True)
        latest_month = unique_months[0] if unique_months else None
        
        return jsonify({
            'success': True,
            'months': unique_months,
            'latest_month': latest_month,
            'total_months': len(unique_months)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/load', methods=['POST'])
def load_data():
    """데이터 로드 (전체 월 데이터)"""
    try:
        import sys
        sys.stdout.flush()  # 출력 버퍼 플러시
        
        loader = ChargingDataLoader()
        
        # 항상 전체 월 데이터 로드
        print('🔄 전체 월 데이터 로드 시작...', flush=True)
        df = loader.load_multiple()
        print('✅ 데이터 로드 완료, 응답 생성 중...', flush=True)
        
        if df is None:
            return jsonify({
                'success': False,
                'error': '데이터 로드 실패'
            }), 500
        
        # 캐시 저장
        cache['data'] = df
        cache['full_data'] = df.copy()  # 전체 데이터 백업 (복사본)
        
        # 기본 정보 반환
        unique_months = []
        latest_month = None
        
        if 'snapshot_month' in df.columns:
            unique_months = sorted(df['snapshot_month'].unique().tolist(), reverse=True)
            latest_month = unique_months[0] if unique_months else None
        
        print(f'💾 캐시 저장: data={len(cache["data"])} 행, full_data={len(cache["full_data"])} 행', flush=True)
        print(f'📅 포함된 월: {unique_months}', flush=True)
        
        # 데이터 로드 후 자동으로 분석 실행
        print('📊 데이터 분석 시작...', flush=True)
        analyzer = ChargingDataAnalyzer(df)
        insights = analyzer.generate_insights()
        cache['insights'] = insights
        print('✅ 데이터 분석 완료', flush=True)
        
        response_data = {
            'success': True,
            'rows': int(len(df)),
            'total_months': len(unique_months),
            'unique_months': unique_months,
            'latest_month': latest_month,
            'columns': [str(col) for col in df.columns],
            'analyzed': True  # 분석 완료 플래그
        }
        
        print(f'📤 응답 전송: {len(df)} 행, {len(unique_months)} 개월', flush=True)
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f'❌ 오류 발생: {error_msg}', flush=True)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/api/filter', methods=['POST'])
def filter_by_month():
    """기준월로 데이터 필터링"""
    try:
        data = request.json
        selected_month = data.get('month')
        
        if not selected_month:
            return jsonify({
                'success': False,
                'error': '기준월을 선택해주세요'
            }), 400
        
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        # 전체 데이터에서 선택된 월만 필터링
        df_full = cache['full_data']
        df_filtered = df_full[df_full['snapshot_month'] == selected_month].copy()
        
        if len(df_filtered) == 0:
            return jsonify({
                'success': False,
                'error': f'{selected_month} 데이터가 없습니다'
            }), 404
        
        # 필터링된 데이터를 캐시에 저장 (full_data는 유지)
        cache['data'] = df_filtered
        
        # 필터링 후 자동으로 분석 실행 (필터링된 데이터로)
        print(f'📊 {selected_month} 데이터 분석 시작...', flush=True)
        analyzer = ChargingDataAnalyzer(df_filtered)
        insights = analyzer.generate_insights()
        cache['insights'] = insights
        print('✅ 데이터 분석 완료', flush=True)
        print(f'💾 full_data 보존: {len(cache["full_data"])} 행', flush=True)
        
        # 정보 반환
        snapshot_date = str(df_filtered['snapshot_date'].iloc[0]) if 'snapshot_date' in df_filtered.columns else None
        
        return jsonify({
            'success': True,
            'rows': int(len(df_filtered)),
            'snapshot_month': selected_month,
            'snapshot_date': snapshot_date,
            'columns': [str(col) for col in df_filtered.columns],
            'filtered': True,
            'analyzed': True  # 분석 완료 플래그
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dashboard')
def get_dashboard():
    """대시보드 데이터 조회 (차트 + 요약)"""
    try:
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        # 전체 데이터로 최근 6개월 차트 생성
        from data_analyzer import ChargingDataAnalyzer
        full_analyzer = ChargingDataAnalyzer(cache['full_data'])
        
        # 현재 필터링된 데이터의 요약 정보
        current_insights = cache.get('insights', {})
        current_data = cache.get('data')
        
        # 선택된 기준월 확인
        target_month = None
        if current_data is not None and 'snapshot_month' in current_data.columns:
            # 현재 선택된 월 (필터링된 데이터의 월)
            target_month = current_data['snapshot_month'].iloc[0] if len(current_data) > 0 else None
        
        print(f'📊 대시보드 생성: 기준월={target_month}', flush=True)
        
        # 현재 선택된 월의 요약 테이블 - 엑셀 K2:P4에서 직접 추출
        summary_table = None
        if current_data is not None and len(current_data) > 0:
            # 현재 선택된 월의 파일 경로 찾기
            data_source = current_data['data_source'].iloc[0] if 'data_source' in current_data.columns else None
            if data_source:
                loader = ChargingDataLoader()
                summary_table = loader.extract_summary_data(data_source)
                print(f'📊 요약 테이블 추출: {summary_table}', flush=True)
        
        # 대시보드 데이터 구성 (선택한 월 기준 최근 6개월)
        dashboard = {
            'summary': current_insights.get('summary'),
            'summary_table': summary_table,
            'top_performers': current_insights.get('top_performers'),
            'target_month': target_month,
            'charts': {
                'total_trend': full_analyzer.get_recent_6months_trend(target_month),
                'gs_trend': full_analyzer.get_gs_chargebee_trend(target_month),
                'top5_market_share': full_analyzer.get_top5_market_share_trend(target_month),
                'cumulative_chargers': full_analyzer.get_cumulative_chargers_trend(target_month)
            }
        }
        
        return jsonify({
            'success': True,
            'dashboard': dashboard
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/generate-report')
def generate_report():
    """AI 리포트 생성"""
    try:
        if cache['insights'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 분석해주세요'
            }), 400
        
        generator = AIReportGenerator()
        report = generator.generate_full_report(cache['insights'])
        
        # 캐시 저장
        cache['report'] = report
        
        return jsonify({
            'success': True,
            'report': report
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/query', methods=['POST'])
def custom_query():
    """커스텀 질의"""
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({
                'success': False,
                'error': '질의를 입력해주세요'
            }), 400
        
        print(f'\n🔍 커스텀 질의 시작: "{query}"', flush=True)
        
        generator = AIReportGenerator()
        
        # Knowledge Base 검색 (배경 지식)
        print(f'📚 Knowledge Base 검색 중...', flush=True)
        kb_context = generator.retrieve_from_kb(query)
        print(f'📊 KB 컨텍스트 길이: {len(kb_context)} 자', flush=True)
        
        # 선택된 기준월 정보
        selected_month = "전체"
        if cache.get('data') is not None and 'snapshot_month' in cache['data'].columns:
            selected_month = cache['data']['snapshot_month'].iloc[0] if len(cache['data']) > 0 else "전체"
        
        print(f'📅 선택된 기준월: {selected_month}', flush=True)
        
        # 현재 선택된 월의 DataFrame을 테이블 형태로 변환
        current_month_table = ""
        if cache.get('data') is not None:
            df = cache['data']
            # 주요 컬럼만 선택하여 테이블 생성
            relevant_cols = ['CPO명', '순위', '충전소수', '완속충전기', '급속충전기', '총충전기', '시장점유율', '순위변동', '충전소증감', '완속증감', '급속증감', '총증감']
            available_cols = [col for col in relevant_cols if col in df.columns]
            
            if len(available_cols) > 0:
                # NaN 값 제거하고 유효한 데이터만 추출
                df_clean = df[available_cols].dropna(subset=['CPO명'])
                # 상위 50개만 (너무 많으면 토큰 초과)
                df_top = df_clean.head(50)
                # 테이블 형태로 변환
                current_month_table = df_top.to_string(index=False)
                print(f'📊 현재 월 테이블: {len(df_top)} 행, {len(available_cols)} 컬럼', flush=True)
        
        # 전체 기간 데이터 (기간별 비교용)
        all_months_summary = ""
        available_months = []
        if cache.get('full_data') is not None:
            df_full = cache['full_data']
            if 'snapshot_month' in df_full.columns:
                available_months = sorted(df_full['snapshot_month'].unique().tolist())
                print(f'📅 사용 가능한 월: {available_months}', flush=True)
                
                # 각 월별로 주요 CPO의 데이터 요약 (상위 20개만)
                relevant_cols_with_month = ['snapshot_month', 'CPO명', '충전소수', '완속충전기', '급속충전기', '총충전기', '시장점유율']
                available_cols_full = [col for col in relevant_cols_with_month if col in df_full.columns]
                
                if len(available_cols_full) > 0:
                    df_full_clean = df_full[available_cols_full].dropna(subset=['CPO명'])
                    # 각 월별 상위 20개 CPO만 추출
                    df_summary = df_full_clean.groupby('snapshot_month').head(20)
                    all_months_summary = df_summary.to_string(index=False, max_rows=200)
                    print(f'📊 전체 기간 요약: {len(df_summary)} 행', flush=True)
        
        # 현재 분석된 인사이트 데이터
        insights_data = ""
        if cache['insights']:
            insights_data = json.dumps(cache['insights'], ensure_ascii=False, indent=2)
            print(f'📊 인사이트 데이터 길이: {len(insights_data)} 자', flush=True)
        
        # 구조화된 프롬프트 생성
        structured_prompt = f"""
당신은 한국 전기차 충전 인프라 데이터 분석 전문가 Agent입니다.

## 사용자 질문
{query}

## 사용 가능한 데이터

### 1. 현재 선택된 월 데이터 ({selected_month})
**단일 월 조회 시 사용**

```
{current_month_table}
```

### 2. 전체 기간 데이터 (기간별 비교용)
**사용 가능한 월: {', '.join(available_months)}**
**기간별 비교 조회 시 사용**

```
{all_months_summary}
```

**컬럼 설명:**
- snapshot_month: 기준 연월 (YYYY-MM 형식)
- CPO명: 충전사업자 이름
- 순위: 시장점유율 기반 순위
- 충전소수: 운영 중인 충전소 개수
- 완속충전기: 완속 충전기 개수
- 급속충전기: 급속 충전기 개수
- 총충전기: 총 충전기 개수 (TTL)
- 시장점유율: 시장점유율 (%)
- 순위변동: 전월 대비 순위 변동
- 충전소증감: 전월 대비 충전소 증감량
- 완속증감: 전월 대비 완속 충전기 증감량
- 급속증감: 전월 대비 급속 충전기 증감량
- 총증감: 전월 대비 총 충전기 증감량

## 질의 처리 방식 (단계별 사고 - Chain of Thought)

**먼저 질의 유형을 판단하세요:**

### 질의 유형 A: 단일 월 조회
- "2025년 10월에 급속충전기가 많은 top 3"
- "2025년 9월 한국전력공사의 충전소 수"
→ **"현재 선택된 월 데이터" 사용**

### 질의 유형 B: 기간별 비교 조회
- "2025년 1월부터 10월까지 완속충전기 증가량이 많은 top 5"
- "2024년 12월과 2025년 10월 비교"
→ **"전체 기간 데이터" 사용**

---

### 유형 A: 단일 월 조회 처리

**Step 1: 질의 분석**
- 요청 월: [YYYY-MM]
- 요청 항목: [무엇을 찾는가?]
- 필요 컬럼: [어떤 컬럼?]
- 정렬 기준: [어떤 순서?]
- 개수: [몇 개?]

**Step 2: 현재 월 테이블 조회**
- 해당 컬럼 값 확인
- 정렬 후 상위 N개 선택

**Step 3: 검증 및 답변**

---

### 유형 B: 기간별 비교 조회 처리

**Step 1: 질의 분석**
- 시작 월: [YYYY-MM]
- 종료 월: [YYYY-MM]
- 비교 항목: [어떤 컬럼?]
- 계산 방식: [증가량 = 종료월 값 - 시작월 값]

**Step 2: 전체 기간 데이터에서 두 시점 조회**

예: "2025년 1월부터 10월까지 완속충전기 증가량 top 5"

1. 전체 기간 데이터에서 snapshot_month = "2025-01" 필터링
   - 각 CPO의 완속충전기 값 추출 → 1월_값
   
2. 전체 기간 데이터에서 snapshot_month = "2025-10" 필터링
   - 각 CPO의 완속충전기 값 추출 → 10월_값
   
3. 각 CPO별로 증가량 계산
   - 증가량 = 10월_값 - 1월_값
   
4. 증가량 기준으로 내림차순 정렬
   
5. 상위 5개 선택

**Step 3: 검증**
- 각 CPO의 1월 값, 10월 값, 증가량을 명시
- 예: "한국전력공사: 1월 10,000기 → 10월 12,000기 (증가량: +2,000기)"

**Step 4: 답변 작성**
- 표 형식으로 정리

---

### 예시 1: 단일 월 조회

질문: "2025년 10월에 급속충전기가 많은 top 3"

Step 1: 단일 월 조회 → 현재 월 테이블 사용
Step 2: 급속충전기 컬럼 확인, 내림차순 정렬, 상위 3개
Step 3: 답변 작성

---

### 예시 2: 기간별 비교

질문: "2025년 1월부터 10월까지 완속충전기 증가량 top 5"

Step 1: 기간별 비교 → 전체 기간 데이터 사용
Step 2: 
- 2025-01 데이터에서 각 CPO의 완속충전기 값
- 2025-10 데이터에서 각 CPO의 완속충전기 값
- 증가량 = 10월 - 1월
Step 3: 증가량 기준 정렬, 상위 5개
Step 4: 답변 작성
- 2위: [CPO명], 급속충전기: [정확한 숫자]
- 3위: [CPO명], 급속충전기: [정확한 숫자]

Step 4 답변:
[표 형식으로 정리된 답변]

## 추가 참고 데이터

**분석 인사이트:**
{insights_data}

**Knowledge Base 참고 (보조 자료):**
{kb_context}

## 답변 작성 규칙

**중요: 반드시 위의 "단계별 사고" 과정을 따라 답변하세요!**

1. **데이터 소스 우선순위**
   - **최우선**: "실제 데이터 테이블" - 이 테이블의 값이 절대적으로 정확합니다
   - Knowledge Base는 참고만 하고, 구체적인 숫자는 테이블에서 가져오세요

2. **정확한 값 추출 방법**
   - 테이블을 한 줄씩 읽으면서 해당 컬럼 값 확인
   - 숫자는 테이블에 표시된 그대로 사용 (쉼표 포함)
   - 절대 추측하거나 계산하지 말 것
   - 테이블에 없는 데이터는 "확인할 수 없습니다" 명시

3. **답변 형식**
   
   반드시 다음 형식으로 답변:
   
   ```
   ## [질문 요약]
   
   [핵심 답변 1-2문장]
   
   | 순위 | CPO명 | [요청 컬럼] | 기타 정보 |
   |------|-------|------------|----------|
   | 1 | [정확한 이름] | [정확한 숫자] | [추가 정보] |
   | 2 | [정확한 이름] | [정확한 숫자] | [추가 정보] |
   | 3 | [정확한 이름] | [정확한 숫자] | [추가 정보] |
   
   **데이터 출처**: {selected_month} 실제 분석 데이터
   ```

4. **금지 사항**
   - Knowledge Base의 다른 월 데이터 사용 금지
   - 테이블에 없는 CPO 언급 금지
   - 숫자 반올림, 근사값 사용 금지
   - HTML, LaTeX, 코드블록 사용 금지

5. **답변 예시**
   - 핵심 답변 (1-2문장, 정확한 수치 포함)
   - 상세 데이터 (표 형식)
   - 추가 인사이트 (있는 경우)

5. **답변 예시**

질문: "2025년 10월 급속 충전기를 많이 운영하는 충전사업자 top 3 알려줘"

올바른 답변:
```
## 2025년 10월 급속충전기 보유 상위 3개 CPO

2025년 10월 기준, 급속충전기를 가장 많이 운영하는 충전사업자는 한국전력공사(15,234기), 환경부(12,567기), SK시그넷(8,901기) 순입니다.

| 순위 | CPO명 | 급속충전기 | 시장점유율 |
|------|-------|-----------|-----------|
| 1 | 한국전력공사 | 15,234 | 31.2% |
| 2 | 환경부 | 12,567 | 25.8% |
| 3 | SK시그넷 | 8,901 | 18.3% |

**데이터 출처**: 2025-10 실제 분석 데이터
```

**중요**: 위 예시의 숫자는 가상입니다. 반드시 실제 테이블에서 정확한 값을 찾아 사용하세요!

한국어로 명확하고 간결하게 답변해주세요.
"""
        
        # Bedrock 응답 생성 (컨텍스트 없이 구조화된 프롬프트만 전달)
        answer = generator.invoke_bedrock_for_query(structured_prompt)
        
        return jsonify({
            'success': True,
            'query': query,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # use_reloader=False로 설정하여 파일 변경 시 자동 재시작 방지
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False, threaded=True)
