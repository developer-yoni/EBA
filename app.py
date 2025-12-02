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
        
        generator = AIReportGenerator()
        
        # Knowledge Base 검색
        context = generator.retrieve_from_kb(query)
        
        # 현재 인사이트 추가
        if cache['insights']:
            context += f"\n\n현재 분석 데이터:\n{json.dumps(cache['insights'], ensure_ascii=False, indent=2)}"
        
        # Bedrock 응답 생성
        answer = generator.invoke_bedrock(query, context)
        
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
