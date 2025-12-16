"""
충전 인프라 분석 리포트 웹앱
"""
from flask import Flask, render_template, jsonify, request
import json
import os
import pandas as pd
from dotenv import load_dotenv
from data_loader import ChargingDataLoader
from data_analyzer import ChargingDataAnalyzer
from ai_report_generator import AIReportGenerator
from query_analyzer import QueryAnalyzer

# 환경변수 로드
load_dotenv()

app = Flask(__name__)

# 전역 캐시
cache = {
    'data': None,
    'full_data': None,
    'insights': None,
    'report': None
}

@app.route('/')
def intro():
    """인트로 페이지"""
    return render_template('intro.html')

@app.route('/dashboard')
def dashboard():
    """대시보드 페이지"""
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

@app.route('/api/dashboard', methods=['GET', 'POST'])
def get_dashboard():
    """대시보드 데이터 조회 (차트 + 요약)"""
    try:
        # GET 요청인 경우 데이터 상태만 확인 (간단한 체크)
        if request.method == 'GET':
            if cache.get('full_data') is not None:
                return jsonify({
                    'success': True,
                    'data_loaded': True,
                    'message': '데이터가 로드되어 있습니다'
                })
            else:
                return jsonify({
                    'success': False,
                    'data_loaded': False,
                    'error': '데이터가 로드되지 않았습니다'
                }), 400
        
        # POST 요청인 경우 실제 대시보드 데이터 반환
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        # 선택된 월 목록 가져오기 (POST 요청인 경우)
        selected_months = []
        start_month = None
        end_month = None
        if request.method == 'POST':
            data = request.json
            selected_months = data.get('months', [])
            start_month = data.get('startMonth')
            end_month = data.get('endMonth')
            print(f'📅 선택된 기간: {start_month} ~ {end_month}', flush=True)
            print(f'📅 선택된 월: {selected_months}', flush=True)
        
        # 전체 데이터로 차트 생성
        from data_analyzer import ChargingDataAnalyzer
        full_analyzer = ChargingDataAnalyzer(cache['full_data'])
        
        # 선택된 월이 있으면 해당 월들로 필터링
        current_data = None
        period_summary = None
        if selected_months:
            filtered_data = cache['full_data'][cache['full_data']['snapshot_month'].isin(selected_months)]
            if len(filtered_data) > 0:
                # 필터링된 데이터로 분석
                analyzer = ChargingDataAnalyzer(filtered_data)
                current_insights = analyzer.generate_insights()
                cache['data'] = filtered_data
                cache['insights'] = current_insights
                current_data = filtered_data
                
                # 기간 표시
                if len(selected_months) == 1:
                    target_month = selected_months[0]
                else:
                    target_month = f"{selected_months[0]}~{selected_months[-1]}"
                
                # 기간 요약 데이터 생성 (시작월~종료월 증감량)
                if start_month and end_month:
                    period_summary = full_analyzer.get_period_summary(start_month, end_month)
                
                print(f'📊 선택된 기간: {len(selected_months)}개월 ({target_month})', flush=True)
            else:
                current_insights = cache.get('insights', {})
                target_month = None
        else:
            # 현재 필터링된 데이터의 요약 정보
            current_insights = cache.get('insights', {})
            current_data = cache.get('data')
            
            # 선택된 기준월 확인
            target_month = None
            if current_data is not None and 'snapshot_month' in current_data.columns:
                target_month = current_data['snapshot_month'].iloc[0] if len(current_data) > 0 else None
        
        print(f'📊 대시보드 생성: 기준월={target_month}', flush=True)
        
        # 현재 선택된 월의 요약 테이블 - 엑셀 K2:P4에서 직접 추출
        summary_table = None
        if current_data is not None and len(current_data) > 0:
            # end_month에 해당하는 파일에서 요약 데이터 추출 (정렬 문제 해결)
            data_source = None
            if end_month and 'snapshot_month' in current_data.columns:
                # end_month에 해당하는 데이터에서 파일 경로 찾기
                end_month_data = current_data[current_data['snapshot_month'] == end_month]
                if len(end_month_data) > 0 and 'data_source' in end_month_data.columns:
                    data_source = end_month_data['data_source'].iloc[0]
                    print(f'📊 end_month({end_month})에서 data_source 찾음: {data_source}', flush=True)
            
            # end_month로 찾지 못한 경우 snapshot_month 기준 최신 데이터 사용
            if not data_source and 'data_source' in current_data.columns:
                sorted_data = current_data.sort_values('snapshot_month', ascending=False)
                data_source = sorted_data['data_source'].iloc[0]
                print(f'📊 정렬 후 최신 data_source: {data_source}', flush=True)
            
            if data_source:
                loader = ChargingDataLoader()
                summary_table = loader.extract_summary_data(data_source)
                print(f'📊 요약 테이블 추출: {summary_table}', flush=True)
        
        # period_summary가 있으면 summary_table 대신 사용
        if period_summary:
            summary_table = period_summary
        
        # 엑셀 N4, O4에서 직접 충전기 증감값 추출
        loader = ChargingDataLoader()
        excel_changes = loader.get_all_months_charger_changes()
        print(f'📊 엑셀에서 추출한 증감값: {len(excel_changes)}개월', flush=True)
        
        # GS차지비 KPI 데이터 생성
        gs_kpi = None
        if current_data is not None and len(current_data) > 0 and end_month:
            gs_data = current_data[current_data['CPO명'] == 'GS차지비']
            if len(gs_data) > 0:
                # 종료월 데이터
                end_data = gs_data[gs_data['snapshot_month'] == end_month]
                if len(end_data) > 0:
                    end_row = end_data.iloc[0]
                    
                    # 현재 시장점유율 파싱
                    try:
                        current_share_raw = end_row.get('시장점유율', '0%')
                        print(f'📊 GS차지비 시장점유율 원본값: {current_share_raw}, 타입: {type(current_share_raw)}', flush=True)
                        
                        if isinstance(current_share_raw, str):
                            current_share = float(current_share_raw.replace('%', '').strip())
                        else:
                            # 이미 숫자인 경우 (0.168 같은 형식)
                            current_share = float(current_share_raw) * 100 if current_share_raw < 1 else float(current_share_raw)
                        
                        print(f'📊 GS차지비 시장점유율 파싱 결과: {current_share}%', flush=True)
                    except Exception as e:
                        print(f'⚠️ 시장점유율 파싱 오류: {e}, 원본값: {end_row.get("시장점유율")}', flush=True)
                        current_share = 0.0
                    
                    # 현재 값
                    current_kpi = {
                        'market_share': round(current_share, 1),
                        'stations': int(end_row.get('충전소수', 0)),
                        'slow_chargers': int(end_row.get('완속충전기', 0)),
                        'fast_chargers': int(end_row.get('급속충전기', 0)),
                        'total_chargers': int(end_row.get('총충전기', 0))
                    }
                    
                    # 전월 대비 증감량 계산
                    # 전월 찾기
                    all_months = sorted(gs_data['snapshot_month'].unique().tolist())
                    monthly_change = {
                        'prev_month': None,
                        'current_month': end_month,
                        'market_share_change': 0,
                        'stations': int(end_row.get('충전소증감', 0)),
                        'slow_chargers': int(end_row.get('완속증감', 0)),
                        'fast_chargers': int(end_row.get('급속증감', 0)),
                        'total_chargers': int(end_row.get('총증감', 0))
                    }
                    
                    if end_month in all_months:
                        current_idx = all_months.index(end_month)
                        if current_idx > 0:
                            prev_month = all_months[current_idx - 1]
                            prev_data = gs_data[gs_data['snapshot_month'] == prev_month]
                            if len(prev_data) > 0:
                                prev_row = prev_data.iloc[0]
                                monthly_change['prev_month'] = prev_month
                                
                                # 전월 시장점유율
                                try:
                                    prev_share_raw = prev_row.get('시장점유율', '0%')
                                    print(f'📊 전월({prev_month}) 시장점유율 원본값: {prev_share_raw}, 타입: {type(prev_share_raw)}', flush=True)
                                    
                                    if isinstance(prev_share_raw, str):
                                        prev_share = float(prev_share_raw.replace('%', '').strip())
                                    else:
                                        prev_share = float(prev_share_raw) * 100 if prev_share_raw < 1 else float(prev_share_raw)
                                    
                                    print(f'📊 전월({prev_month}) 시장점유율 파싱 결과: {prev_share}%', flush=True)
                                    print(f'📊 현재월({end_month}) 시장점유율: {current_share}%', flush=True)
                                    
                                    share_change = round(current_share - prev_share, 1)
                                    print(f'📊 시장점유율 증감량: {current_share}% - {prev_share}% = {share_change}%p', flush=True)
                                    
                                    monthly_change['market_share_change'] = share_change
                                except Exception as e:
                                    print(f'⚠️ 전월 시장점유율 파싱 오류: {e}', flush=True)
                                    monthly_change['market_share_change'] = 0
                    
                    # 기간 증감량
                    period_change = None
                    if start_month:
                        start_data = gs_data[gs_data['snapshot_month'] == start_month]
                        if len(start_data) > 0:
                            start_row = start_data.iloc[0]
                            
                            # 시장점유율 변화
                            try:
                                start_share_raw = start_row.get('시장점유율', '0%')
                                if isinstance(start_share_raw, str):
                                    start_share = float(start_share_raw.replace('%', '').strip())
                                else:
                                    start_share = float(start_share_raw) * 100 if start_share_raw < 1 else float(start_share_raw)
                                
                                end_share_raw = end_row.get('시장점유율', '0%')
                                if isinstance(end_share_raw, str):
                                    end_share = float(end_share_raw.replace('%', '').strip())
                                else:
                                    end_share = float(end_share_raw) * 100 if end_share_raw < 1 else float(end_share_raw)
                                
                                share_change = round(end_share - start_share, 1)
                            except Exception as e:
                                print(f'⚠️ 기간 시장점유율 변화 계산 오류: {e}', flush=True)
                                share_change = 0
                            
                            period_change = {
                                'market_share_change': share_change,
                                'stations': int(end_row.get('충전소수', 0)) - int(start_row.get('충전소수', 0)),
                                'slow_chargers': int(end_row.get('완속충전기', 0)) - int(start_row.get('완속충전기', 0)),
                                'fast_chargers': int(end_row.get('급속충전기', 0)) - int(start_row.get('급속충전기', 0)),
                                'total_chargers': int(end_row.get('총충전기', 0)) - int(start_row.get('총충전기', 0))
                            }
                    
                    gs_kpi = {
                        'current': current_kpi,
                        'monthly_change': monthly_change,
                        'period_change': period_change
                    }
                    print(f'📊 GS차지비 KPI 생성 완료', flush=True)
        
        # 최신 월(end_month) 기준 TOP10 CPO 생성
        top_performers_data = None
        latest_month_for_top = end_month or (selected_months[0] if selected_months else None)
        if latest_month_for_top and cache['full_data'] is not None:
            latest_month_data = cache['full_data'][cache['full_data']['snapshot_month'] == latest_month_for_top]
            if len(latest_month_data) > 0:
                latest_analyzer = ChargingDataAnalyzer(latest_month_data)
                top_performers_data = latest_analyzer.top_performers(10)
                print(f'📊 TOP10 CPO 생성 완료 (기준월: {latest_month_for_top})', flush=True)
        
        # top_performers_data가 없으면 current_insights에서 가져오기
        if not top_performers_data:
            top_performers_data = current_insights.get('top_performers')
        
        # 대시보드 데이터 구성 (선택한 기간 기준)
        dashboard = {
            'summary': current_insights.get('summary'),
            'summary_table': summary_table,
            'gs_kpi': gs_kpi,
            'top_performers': top_performers_data,
            'target_month': target_month,
            'start_month': start_month,
            'end_month': end_month,
            'charts': {
                'total_trend': full_analyzer.get_recent_6months_trend(target_month, start_month, end_month, excel_changes),
                'gs_trend': full_analyzer.get_gs_chargebee_trend(target_month, start_month, end_month),
                'top5_market_share': full_analyzer.get_top5_market_share_trend(target_month, start_month, end_month),
                'cumulative_chargers': full_analyzer.get_cumulative_chargers_trend(target_month, start_month, end_month)
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

# 리포트 생성 진행 상태 저장 (세션별)
report_progress = {}

@app.route('/api/report-progress/<session_id>')
def get_report_progress(session_id):
    """리포트 생성 진행 상태 조회"""
    progress = report_progress.get(session_id, {
        'completed': [],
        'total': 3,
        'status': 'pending'
    })
    return jsonify(progress)

@app.route('/api/generate-all-reports', methods=['POST'])
def generate_all_reports():
    """AI 리포트 3종 병렬 생성 (KPI + CPO + Trend) - 실시간 진행률 지원"""
    import time
    import uuid
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    try:
        data = request.json
        target_month = data.get('targetMonth')
        
        if not target_month:
            return jsonify({
                'success': False,
                'error': '기준월을 선택해주세요'
            }), 400
        
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        # 세션 ID 생성 (진행률 추적용)
        session_id = str(uuid.uuid4())
        report_progress[session_id] = {
            'completed': [],
            'total': 3,
            'status': 'running',
            'report_times': {}
        }
        
        print(f'\n🚀 병렬 리포트 생성 시작 - 기준월: {target_month} (세션: {session_id[:8]})', flush=True)
        total_start = time.time()
        
        # 실제 데이터에서 사용 가능한 모든 월 가져오기
        all_months = sorted(cache['full_data']['snapshot_month'].unique().tolist())
        
        # 기준월 기준 최근 12개월 계산
        from datetime import datetime
        target_date = datetime.strptime(target_month, '%Y-%m')
        
        months_back = 11
        start_year = target_date.year
        start_month_num = target_date.month - months_back
        
        while start_month_num <= 0:
            start_month_num += 12
            start_year -= 1
        
        start_month = f'{start_year}-{start_month_num:02d}'
        available_months = [m for m in all_months if start_month <= m <= target_month]
        
        if len(available_months) < 12:
            available_months = [m for m in all_months if m <= target_month]
        
        print(f'📅 분석 범위: {available_months[0]} ~ {available_months[-1]} ({len(available_months)}개월)', flush=True)
        
        # 기준월 데이터
        target_data = cache['full_data'][cache['full_data']['snapshot_month'] == target_month]
        range_data = cache['full_data'][cache['full_data']['snapshot_month'].isin(available_months)]
        
        if len(target_data) == 0:
            return jsonify({
                'success': False,
                'error': f'{target_month} 데이터가 없습니다'
            }), 404
        
        # 분석 실행
        from data_analyzer import ChargingDataAnalyzer
        from data_loader import ChargingDataLoader
        target_analyzer = ChargingDataAnalyzer(target_data)
        range_analyzer = ChargingDataAnalyzer(range_data)
        
        target_insights = target_analyzer.generate_insights()
        range_insights = range_analyzer.generate_insights()
        
        # 엑셀 요약 데이터로 summary 덮어쓰기 (RAG 데이터 정합성)
        if 'data_source' in target_data.columns and len(target_data) > 0:
            data_source = target_data['data_source'].iloc[0]
            if data_source:
                loader = ChargingDataLoader()
                excel_summary = loader.extract_summary_data(data_source)
                if excel_summary and 'total' in excel_summary:
                    total = excel_summary['total']
                    target_insights['summary']['total_cpos'] = total.get('cpos', 0)
                    target_insights['summary']['total_stations'] = total.get('stations', 0)
                    target_insights['summary']['total_chargers'] = total.get('total_chargers', 0)
                    target_insights['summary']['slow_chargers'] = total.get('slow_chargers', 0)
                    target_insights['summary']['fast_chargers'] = total.get('fast_chargers', 0)
                    total_chargers = total.get('total_chargers', 0)
                    if total_chargers > 0:
                        target_insights['summary']['slow_ratio'] = round(total.get('slow_chargers', 0) / total_chargers * 100, 1)
                        target_insights['summary']['fast_ratio'] = round(total.get('fast_chargers', 0) / total_chargers * 100, 1)
                    print(f'✅ AI 리포트용 엑셀 요약 데이터 주입 완료: CPO={total.get("cpos")}, 충전소={total.get("stations")}, 충전기={total.get("total_chargers")}', flush=True)
        
        # 병렬 실행을 위한 함수 정의 (각 스레드에서 별도 generator 인스턴스 생성)
        def generate_kpi():
            local_generator = AIReportGenerator()
            start = time.time()
            content = local_generator.generate_kpi_snapshot_report(
                target_month=target_month,
                target_insights=target_insights,
                target_data=target_data,
                available_months=available_months
            )
            elapsed = time.time() - start
            # 진행 상태 업데이트
            report_progress[session_id]['completed'].append('kpi')
            report_progress[session_id]['report_times']['kpi'] = round(elapsed, 2)
            print(f'✅ KPI Report 완료 (⏱️ {elapsed:.2f}초) - 진행률: {len(report_progress[session_id]["completed"])}/3', flush=True)
            return ('kpi', content, elapsed)
        
        def generate_cpo():
            local_generator = AIReportGenerator()
            start = time.time()
            content = local_generator.generate_cpo_ranking_report(
                target_month=target_month,
                target_insights=target_insights,
                target_data=target_data,
                available_months=available_months
            )
            elapsed = time.time() - start
            # 진행 상태 업데이트
            report_progress[session_id]['completed'].append('cpo')
            report_progress[session_id]['report_times']['cpo'] = round(elapsed, 2)
            print(f'✅ CPO Report 완료 (⏱️ {elapsed:.2f}초) - 진행률: {len(report_progress[session_id]["completed"])}/3', flush=True)
            return ('cpo', content, elapsed)
        
        def generate_trend():
            local_generator = AIReportGenerator()
            start = time.time()
            content = local_generator.generate_monthly_trend_report(
                target_month=target_month,
                range_insights=range_insights,
                range_data=range_data,
                available_months=available_months
            )
            elapsed = time.time() - start
            # 진행 상태 업데이트
            report_progress[session_id]['completed'].append('trend')
            report_progress[session_id]['report_times']['trend'] = round(elapsed, 2)
            print(f'✅ Trend Report 완료 (⏱️ {elapsed:.2f}초) - 진행률: {len(report_progress[session_id]["completed"])}/3', flush=True)
            return ('trend', content, elapsed)
        
        # ThreadPoolExecutor로 병렬 실행
        reports = {}
        report_times = {}
        
        print(f'🔄 3개 리포트 병렬 생성 시작...', flush=True)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(generate_kpi),
                executor.submit(generate_cpo),
                executor.submit(generate_trend)
            ]
            
            for future in as_completed(futures):
                report_type, content, elapsed = future.result()
                reports[report_type] = content
                report_times[report_type] = round(elapsed, 2)
        
        total_elapsed = time.time() - total_start
        
        # 진행 상태 완료로 업데이트
        report_progress[session_id]['status'] = 'completed'
        
        # 오래된 세션 정리 (메모리 관리)
        if len(report_progress) > 100:
            oldest_keys = list(report_progress.keys())[:50]
            for key in oldest_keys:
                del report_progress[key]
        
        print(f'\n✅ 병렬 리포트 생성 완료!', flush=True)
        print(f'   - KPI: {report_times.get("kpi", 0)}초', flush=True)
        print(f'   - CPO: {report_times.get("cpo", 0)}초', flush=True)
        print(f'   - Trend: {report_times.get("trend", 0)}초', flush=True)
        print(f'   - 총 소요: {total_elapsed:.2f}초 (순차 실행 대비 약 {sum(report_times.values()) / total_elapsed:.1f}배 빠름)', flush=True)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'reports': {
                'kpi': {'type': 'kpi', 'content': reports.get('kpi', '')},
                'cpo': {'type': 'cpo', 'content': reports.get('cpo', '')},
                'trend': {'type': 'trend', 'content': reports.get('trend', '')}
            },
            'report_times': report_times,
            'total_time': round(total_elapsed, 2)
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate-all-reports-stream', methods=['POST'])
def generate_all_reports_stream():
    """AI 리포트 3종 병렬 생성 - SSE 스트리밍 방식"""
    from flask import Response, stream_with_context
    import time
    import uuid
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import queue
    import threading
    
    data = request.json
    target_month = data.get('targetMonth')
    
    if not target_month:
        return jsonify({'success': False, 'error': '기준월을 선택해주세요'}), 400
    
    if cache['full_data'] is None:
        return jsonify({'success': False, 'error': '먼저 데이터를 로드해주세요'}), 400
    
    def generate():
        import json
        
        try:
            total_start = time.time()
            
            # 초기 상태 전송
            yield f"data: {json.dumps({'type': 'start', 'message': '리포트 생성 시작', 'progress': 0})}\n\n"
            
            # 데이터 준비
            all_months = sorted(cache['full_data']['snapshot_month'].unique().tolist())
            from datetime import datetime
            target_date = datetime.strptime(target_month, '%Y-%m')
            
            months_back = 11
            start_year = target_date.year
            start_month_num = target_date.month - months_back
            
            while start_month_num <= 0:
                start_month_num += 12
                start_year -= 1
            
            start_month_str = f'{start_year}-{start_month_num:02d}'
            available_months = [m for m in all_months if start_month_str <= m <= target_month]
            
            if len(available_months) < 12:
                available_months = [m for m in all_months if m <= target_month]
            
            target_data = cache['full_data'][cache['full_data']['snapshot_month'] == target_month]
            range_data = cache['full_data'][cache['full_data']['snapshot_month'].isin(available_months)]
            
            from data_analyzer import ChargingDataAnalyzer
            from data_loader import ChargingDataLoader
            target_analyzer = ChargingDataAnalyzer(target_data)
            range_analyzer = ChargingDataAnalyzer(range_data)
            
            target_insights = target_analyzer.generate_insights()
            range_insights = range_analyzer.generate_insights()
            
            # 엑셀 요약 데이터로 summary 덮어쓰기 (RAG 데이터 정합성)
            if 'data_source' in target_data.columns and len(target_data) > 0:
                data_source = target_data['data_source'].iloc[0]
                if data_source:
                    loader = ChargingDataLoader()
                    excel_summary = loader.extract_summary_data(data_source)
                    if excel_summary and 'total' in excel_summary:
                        total = excel_summary['total']
                        target_insights['summary']['total_cpos'] = total.get('cpos', 0)
                        target_insights['summary']['total_stations'] = total.get('stations', 0)
                        target_insights['summary']['total_chargers'] = total.get('total_chargers', 0)
                        target_insights['summary']['slow_chargers'] = total.get('slow_chargers', 0)
                        target_insights['summary']['fast_chargers'] = total.get('fast_chargers', 0)
                        total_chargers = total.get('total_chargers', 0)
                        if total_chargers > 0:
                            target_insights['summary']['slow_ratio'] = round(total.get('slow_chargers', 0) / total_chargers * 100, 1)
                            target_insights['summary']['fast_ratio'] = round(total.get('fast_chargers', 0) / total_chargers * 100, 1)
                        print(f'✅ AI 리포트용 엑셀 요약 데이터 주입 완료: CPO={total.get("cpos")}, 충전소={total.get("stations")}, 충전기={total.get("total_chargers")}', flush=True)
            
            # 결과 저장용 큐
            result_queue = queue.Queue()
            reports = {}
            report_times = {}
            completed_count = 0
            
            def generate_kpi():
                local_generator = AIReportGenerator()
                start = time.time()
                content = local_generator.generate_kpi_snapshot_report(
                    target_month=target_month,
                    target_insights=target_insights,
                    target_data=target_data,
                    available_months=available_months
                )
                elapsed = time.time() - start
                result_queue.put(('kpi', content, elapsed))
            
            def generate_cpo():
                local_generator = AIReportGenerator()
                start = time.time()
                content = local_generator.generate_cpo_ranking_report(
                    target_month=target_month,
                    target_insights=target_insights,
                    target_data=target_data,
                    available_months=available_months
                )
                elapsed = time.time() - start
                result_queue.put(('cpo', content, elapsed))
            
            def generate_trend():
                local_generator = AIReportGenerator()
                start = time.time()
                content = local_generator.generate_monthly_trend_report(
                    target_month=target_month,
                    range_insights=range_insights,
                    range_data=range_data,
                    available_months=available_months
                )
                elapsed = time.time() - start
                result_queue.put(('trend', content, elapsed))
            
            # 스레드 시작
            threads = [
                threading.Thread(target=generate_kpi),
                threading.Thread(target=generate_cpo),
                threading.Thread(target=generate_trend)
            ]
            
            for t in threads:
                t.start()
            
            # 결과 수집 및 진행률 전송
            while completed_count < 3:
                try:
                    report_type, content, elapsed = result_queue.get(timeout=1)
                    reports[report_type] = content
                    report_times[report_type] = round(elapsed, 2)
                    completed_count += 1
                    progress = int((completed_count / 3) * 100)
                    
                    yield f"data: {json.dumps({'type': 'progress', 'report': report_type, 'time': round(elapsed, 2), 'progress': progress, 'completed': completed_count})}\n\n"
                except queue.Empty:
                    continue
            
            # 모든 스레드 종료 대기
            for t in threads:
                t.join()
            
            total_elapsed = time.time() - total_start
            
            # 최종 결과 전송
            yield f"data: {json.dumps({'type': 'complete', 'reports': {'kpi': {'content': reports.get('kpi', '')}, 'cpo': {'content': reports.get('cpo', '')}, 'trend': {'content': reports.get('trend', '')}}, 'report_times': report_times, 'total_time': round(total_elapsed, 2), 'progress': 100})}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/generate-report', methods=['GET', 'POST'])
def generate_report():
    """AI 리포트 생성 (3가지 유형)"""
    try:
        target_month = None
        report_type = 'kpi'  # 기본값
        
        if request.method == 'POST':
            data = request.json
            target_month = data.get('targetMonth')
            report_type = data.get('reportType', 'kpi')
            print(f'📅 리포트 생성 - 기준월: {target_month}, 유형: {report_type}', flush=True)
        
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        if not target_month:
            return jsonify({
                'success': False,
                'error': '기준월을 선택해주세요'
            }), 400
        
        # 실제 데이터에서 사용 가능한 모든 월 가져오기
        all_months = sorted(cache['full_data']['snapshot_month'].unique().tolist())
        
        # 기준월 기준 최근 12개월 계산
        from datetime import datetime
        
        target_date = datetime.strptime(target_month, '%Y-%m')
        
        # 12개월 전 계산 (기준월 포함)
        year = target_date.year
        month = target_date.month
        
        start_year = year - 1 if month == 12 else year - (12 - month) // 12 - 1
        start_month_num = month if month == 12 else (month - 12) % 12 if month <= 12 else month - 11
        
        # 더 간단한 방법: 11개월 전 계산
        months_back = 11
        start_year = year
        start_month_num = month - months_back
        
        while start_month_num <= 0:
            start_month_num += 12
            start_year -= 1
        
        start_month = f'{start_year}-{start_month_num:02d}'
        
        # 기준월까지의 최근 12개월 필터링 (실제 데이터 범위 내에서)
        available_months = [m for m in all_months if start_month <= m <= target_month]
        
        # 데이터가 12개월 미만인 경우 사용 가능한 모든 월 사용
        if len(available_months) < 12:
            available_months = [m for m in all_months if m <= target_month]
        
        print(f'📅 기준월: {target_month}', flush=True)
        print(f'📅 분석 범위: {available_months[0]} ~ {available_months[-1]} ({len(available_months)}개월)', flush=True)
        print(f'📅 사용 월: {available_months}', flush=True)
        
        # 기준월 데이터 (메인)
        target_data = cache['full_data'][cache['full_data']['snapshot_month'] == target_month]
        
        # 분석 범위 데이터 (최근 12개월)
        range_data = cache['full_data'][cache['full_data']['snapshot_month'].isin(available_months)]
        
        if len(target_data) == 0:
            return jsonify({
                'success': False,
                'error': f'{target_month} 데이터가 없습니다'
            }), 404
        
        # 분석 실행
        from data_analyzer import ChargingDataAnalyzer
        from data_loader import ChargingDataLoader
        target_analyzer = ChargingDataAnalyzer(target_data)
        range_analyzer = ChargingDataAnalyzer(range_data)
        
        target_insights = target_analyzer.generate_insights()
        range_insights = range_analyzer.generate_insights()
        
        # 엑셀 요약 데이터로 summary 덮어쓰기 (RAG 데이터 정합성)
        if 'data_source' in target_data.columns and len(target_data) > 0:
            data_source = target_data['data_source'].iloc[0]
            if data_source:
                loader = ChargingDataLoader()
                excel_summary = loader.extract_summary_data(data_source)
                if excel_summary and 'total' in excel_summary:
                    total = excel_summary['total']
                    target_insights['summary']['total_cpos'] = total.get('cpos', 0)
                    target_insights['summary']['total_stations'] = total.get('stations', 0)
                    target_insights['summary']['total_chargers'] = total.get('total_chargers', 0)
                    target_insights['summary']['slow_chargers'] = total.get('slow_chargers', 0)
                    target_insights['summary']['fast_chargers'] = total.get('fast_chargers', 0)
                    total_chargers = total.get('total_chargers', 0)
                    if total_chargers > 0:
                        target_insights['summary']['slow_ratio'] = round(total.get('slow_chargers', 0) / total_chargers * 100, 1)
                        target_insights['summary']['fast_ratio'] = round(total.get('fast_chargers', 0) / total_chargers * 100, 1)
                    print(f'✅ AI 리포트용 엑셀 요약 데이터 주입 완료: CPO={total.get("cpos")}, 충전소={total.get("stations")}, 충전기={total.get("total_chargers")}', flush=True)
        
        # 리포트 유형별 생성
        generator = AIReportGenerator()
        
        if report_type == 'kpi':
            report_content = generator.generate_kpi_snapshot_report(
                target_month=target_month,
                target_insights=target_insights,
                target_data=target_data,
                available_months=available_months
            )
        elif report_type == 'cpo':
            report_content = generator.generate_cpo_ranking_report(
                target_month=target_month,
                target_insights=target_insights,
                target_data=target_data,
                available_months=available_months
            )
        elif report_type == 'trend':
            report_content = generator.generate_monthly_trend_report(
                target_month=target_month,
                range_insights=range_insights,
                range_data=range_data,
                available_months=available_months
            )
        else:
            return jsonify({
                'success': False,
                'error': f'알 수 없는 리포트 유형: {report_type}'
            }), 400
        
        report = {
            'type': report_type,
            'content': report_content
        }
        
        # 캐시 저장
        cache['report'] = report
        
        return jsonify({
            'success': True,
            'report': report
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/query', methods=['POST'])
def custom_query():
    """커스텀 질의 - 차트 생성 기능 포함"""
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({
                'success': False,
                'error': '질의를 입력해주세요'
            }), 400
        
        print(f'\n🔍 커스텀 질의 시작: "{query}"', flush=True)
        
        # 데이터 확인
        if cache.get('full_data') is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        # QueryAnalyzer로 질의 처리 (RAG + 프롬프트 엔지니어링 + 코드 인터프리터)
        analyzer = QueryAnalyzer()
        result = analyzer.process_query(
            query=query,
            df=cache.get('data'),
            full_df=cache.get('full_data')
        )
        
        # 차트 생성이 필요 없거나 기존 로직 사용 플래그가 있는 경우
        if result.get('use_legacy'):
            print(f'📝 기존 텍스트 답변 로직 사용', flush=True)
            return _legacy_query_handler(query)
        
        # 차트 포함 응답
        if result.get('success'):
            response_data = {
                'success': True,
                'query': query,
                'answer': result.get('answer'),
                'has_chart': result.get('has_chart', False),
                'bedrock_time': result.get('bedrock_time', 0),
                'total_time': result.get('total_time', 0)
            }
            
            # 차트 이미지가 있으면 추가
            if result.get('has_chart') and result.get('chart_image'):
                response_data['chart'] = {
                    'image': result.get('chart_image'),
                    'type': result.get('chart_type'),
                    'title': result.get('chart_title')
                }
                response_data['data_summary'] = result.get('data_summary')
            
            return jsonify(response_data)
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '질의 처리 실패')
            }), 500
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _legacy_query_handler(query):
    """기존 텍스트 기반 질의 처리 (차트 불필요 시)"""
    import time
    start_time = time.time()
    
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
        relevant_cols = ['CPO명', '순위', '충전소수', '완속충전기', '급속충전기', '총충전기', '시장점유율', '순위변동', '충전소증감', '완속증감', '급속증감', '총증감']
        available_cols = [col for col in relevant_cols if col in df.columns]
        
        if len(available_cols) > 0:
            df_clean = df[available_cols].dropna(subset=['CPO명'])
            df_top = df_clean.head(50)
            current_month_table = df_top.to_string(index=False)
            print(f'📊 현재 월 테이블: {len(df_top)} 행, {len(available_cols)} 컬럼', flush=True)
    
    # 전체 기간 데이터
    all_months_summary = ""
    available_months = []
    if cache.get('full_data') is not None:
        df_full = cache['full_data']
        if 'snapshot_month' in df_full.columns:
            available_months = sorted(df_full['snapshot_month'].unique().tolist())
            print(f'📅 사용 가능한 월: {available_months}', flush=True)
            
            relevant_cols_with_month = ['snapshot_month', 'CPO명', '충전소수', '완속충전기', '급속충전기', '총충전기', '시장점유율']
            available_cols_full = [col for col in relevant_cols_with_month if col in df_full.columns]
            
            if len(available_cols_full) > 0:
                df_full_clean = df_full[available_cols_full].dropna(subset=['CPO명'])
                df_summary = df_full_clean.groupby('snapshot_month').head(20)
                all_months_summary = df_summary.to_string(index=False, max_rows=200)
                print(f'📊 전체 기간 요약: {len(df_summary)} 행', flush=True)
    
    # 인사이트 데이터
    insights_data = ""
    if cache['insights']:
        insights_data = json.dumps(cache['insights'], ensure_ascii=False, indent=2)
    
    # 구조화된 프롬프트
    structured_prompt = f"""
당신은 한국 전기차 충전 인프라 데이터 분석 전문가입니다.

## 사용자 질문
{query}

## 현재 선택된 월 데이터 ({selected_month})
```
{current_month_table}
```

## 전체 기간 데이터
사용 가능한 월: {', '.join(available_months)}
```
{all_months_summary}
```

## Knowledge Base 참고
{kb_context}

## 답변 규칙
1. 실제 데이터 테이블의 값을 최우선으로 사용
2. 정확한 숫자만 사용 (추측 금지)
3. 표 형식으로 명확하게 답변
4. 한국어로 간결하게 답변

한국어로 답변해주세요.
"""
    
    answer, bedrock_time = generator.invoke_bedrock_for_query(structured_prompt)
    total_time = time.time() - start_time
    
    return jsonify({
        'success': True,
        'query': query,
        'answer': answer,
        'has_chart': False,
        'response_time': round(total_time, 2),
        'bedrock_time': round(bedrock_time, 2)
    })

@app.route('/api/gs-kpi', methods=['POST'])
def get_gs_kpi():
    """GS차지비 KPI 데이터 조회"""
    try:
        data = request.json
        start_month = data.get('startMonth')
        end_month = data.get('endMonth')
        target_month = data.get('targetMonth', end_month)
        
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        df = cache['full_data']
        
        print(f'📊 GS-KPI: 전체 데이터 행 수: {len(df)}', flush=True)
        print(f'📊 GS-KPI: CPO명 컬럼 존재: {"CPO명" in df.columns}', flush=True)
        
        # GS차지비 데이터 필터링
        gs_data = df[df['CPO명'] == 'GS차지비'].copy()
        
        print(f'📊 GS-KPI: GS차지비 데이터 행 수: {len(gs_data)}', flush=True)
        
        if len(gs_data) == 0:
            # CPO명 샘플 출력
            sample_cpos = df['CPO명'].dropna().unique()[:10]
            print(f'📊 GS-KPI: CPO명 샘플: {sample_cpos}', flush=True)
            return jsonify({
                'success': False,
                'error': 'GS차지비 데이터를 찾을 수 없습니다'
            }), 404
        
        # 기준월 데이터
        current_data = gs_data[gs_data['snapshot_month'] == target_month]
        if len(current_data) == 0:
            return jsonify({
                'success': False,
                'error': f'{target_month} GS차지비 데이터를 찾을 수 없습니다'
            }), 404
        
        current_row = current_data.iloc[0]
        
        # 현재 값
        current_kpi = {
            'market_share': current_row.get('시장점유율', 'N/A'),
            'stations': int(current_row.get('충전소수', 0)),
            'slow_chargers': int(current_row.get('완속충전기', 0)),
            'fast_chargers': int(current_row.get('급속충전기', 0)),
            'total_chargers': int(current_row.get('총충전기', 0))
        }
        
        # 전월 대비 증감량
        monthly_change = None
        all_months = sorted(gs_data['snapshot_month'].unique().tolist())
        if target_month in all_months:
            current_idx = all_months.index(target_month)
            if current_idx > 0:
                prev_month = all_months[current_idx - 1]
                prev_data = gs_data[gs_data['snapshot_month'] == prev_month]
                if len(prev_data) > 0:
                    prev_row = prev_data.iloc[0]
                    
                    # 시장점유율 변화 계산
                    current_share = current_row.get('시장점유율', '0%')
                    prev_share = prev_row.get('시장점유율', '0%')
                    
                    # 퍼센트 문자열을 숫자로 변환
                    try:
                        current_share_num = float(str(current_share).replace('%', ''))
                        prev_share_num = float(str(prev_share).replace('%', ''))
                        share_change = round(current_share_num - prev_share_num, 1)
                    except:
                        share_change = 0
                    
                    monthly_change = {
                        'prev_month': prev_month,
                        'current_month': target_month,
                        'market_share_change': share_change,
                        'stations': int(current_row.get('충전소증감', 0)),
                        'slow_chargers': int(current_row.get('완속증감', 0)),
                        'fast_chargers': int(current_row.get('급속증감', 0)),
                        'total_chargers': int(current_row.get('총증감', 0))
                    }
        
        # 기간 증감량 (시작월 ~ 종료월)
        period_change = None
        if start_month and end_month:
            start_data = gs_data[gs_data['snapshot_month'] == start_month]
            end_data = gs_data[gs_data['snapshot_month'] == end_month]
            
            if len(start_data) > 0 and len(end_data) > 0:
                start_row = start_data.iloc[0]
                end_row = end_data.iloc[0]
                
                # 시장점유율 변화
                try:
                    start_share = float(str(start_row.get('시장점유율', '0%')).replace('%', ''))
                    end_share = float(str(end_row.get('시장점유율', '0%')).replace('%', ''))
                    share_change = round(end_share - start_share, 1)
                except:
                    share_change = 0
                
                period_change = {
                    'market_share_change': share_change,
                    'stations': int(end_row.get('충전소수', 0)) - int(start_row.get('충전소수', 0)),
                    'slow_chargers': int(end_row.get('완속충전기', 0)) - int(start_row.get('완속충전기', 0)),
                    'fast_chargers': int(end_row.get('급속충전기', 0)) - int(start_row.get('급속충전기', 0)),
                    'total_chargers': int(end_row.get('총충전기', 0)) - int(start_row.get('총충전기', 0))
                }
        
        return jsonify({
            'success': True,
            'gs_kpi': {
                'current': current_kpi,
                'monthly_change': monthly_change,
                'period_change': period_change
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/simulation/predict', methods=['POST'])
def predict_market_share():
    """AI 기반 시장점유율 시뮬레이션 예측 (레거시 호환)"""
    try:
        data = request.json
        base_month = data.get('baseMonth')
        simulation_months = data.get('simulationMonths', 12)
        additional_chargers = data.get('additionalChargers', 0)
        
        if not base_month:
            return jsonify({
                'success': False,
                'error': '기준월을 선택해주세요'
            }), 400
        
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        print(f'\n🎯 AI 시뮬레이션 시작: 기준월={base_month}, 기간={simulation_months}개월, 추가충전기={additional_chargers}대', flush=True)
        
        # 새로운 ScenarioSimulator 사용
        from scenario_simulator import ScenarioSimulator
        simulator = ScenarioSimulator()
        
        # AI 예측 실행
        result = simulator.run_simulation(
            base_month=base_month,
            sim_period_months=simulation_months,
            extra_chargers=additional_chargers,
            full_data=cache['full_data']
        )
        
        if result.get('success'):
            prediction = result.get('prediction')
            # 차트 데이터 생성
            chart_data = simulator.generate_chart_data(prediction)
            prediction['chart_data'] = chart_data
            
            return jsonify({
                'success': True,
                'prediction': prediction
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'AI 예측 실패')
            }), 500
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/scenario-simulator', methods=['POST'])
def scenario_simulator():
    """
    AI Scenario Simulator API
    
    RAG 데이터 기반 미래 시장점유율 시뮬레이션
    
    모드 1 (charger_to_share): 충전기 추가 → 점유율 예측
    모드 2 (share_to_charger): 목표 점유율 → 필요 충전기 계산
    
    Request Body:
        - mode: 시뮬레이션 모드 ('charger_to_share' | 'share_to_charger')
        - baseMonth: 기준월 (YYYY-MM)
        - simPeriodMonths: 예측 기간 (개월)
        - extraChargers: 추가 설치 충전기 수 (모드 1)
        - targetShare: 목표 시장점유율 (모드 2)
    """
    try:
        data = request.json
        mode = data.get('mode', 'charger_to_share')
        base_month = data.get('baseMonth')
        sim_period_months = data.get('simPeriodMonths', 6)
        
        # 유효성 검사
        if not base_month:
            return jsonify({
                'success': False,
                'error': '기준월(baseMonth)을 선택해주세요'
            }), 400
        
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        # RAG 데이터 범위 확인
        all_months = sorted(cache['full_data']['snapshot_month'].unique().tolist())
        earliest_month = all_months[0]
        rag_latest_month = all_months[-1]
        
        # 기준월 유효성 검사
        if base_month < earliest_month or base_month > rag_latest_month:
            return jsonify({
                'success': False,
                'error': f'기준월은 RAG 데이터 범위 내여야 합니다 ({earliest_month} ~ {rag_latest_month})'
            }), 400
        
        from scenario_simulator import ScenarioSimulator
        simulator = ScenarioSimulator()
        
        # 모드에 따른 처리
        if mode == 'share_to_charger':
            # 모드 2: 목표 점유율 → 필요 충전기 계산
            target_share = data.get('targetShare', 17.0)
            
            print(f'\n🎯 목표 점유율 역계산 API 호출', flush=True)
            print(f'   ├─ baseMonth: {base_month}', flush=True)
            print(f'   ├─ simPeriodMonths: {sim_period_months}', flush=True)
            print(f'   ├─ targetShare: {target_share}%', flush=True)
            print(f'   └─ RAG 범위: {earliest_month} ~ {rag_latest_month}', flush=True)
            
            result = simulator.calculate_required_chargers(
                base_month=base_month,
                sim_period_months=sim_period_months,
                target_share=target_share,
                full_data=cache['full_data']
            )
            
            if result.get('success'):
                return jsonify(result)
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', '역계산 실패')
                }), 500
        
        else:
            # 모드 1: 충전기 추가 → 점유율 예측 (기존 로직)
            extra_chargers = data.get('extraChargers', 0)
            
            print(f'\n🎯 AI Scenario Simulator API 호출', flush=True)
            print(f'   ├─ baseMonth: {base_month}', flush=True)
            print(f'   ├─ simPeriodMonths: {sim_period_months}', flush=True)
            print(f'   ├─ extraChargers: {extra_chargers:,}', flush=True)
            print(f'   └─ RAG 범위: {earliest_month} ~ {rag_latest_month}', flush=True)
            
            result = simulator.run_simulation(
                base_month=base_month,
                sim_period_months=sim_period_months,
                extra_chargers=extra_chargers,
                full_data=cache['full_data']
            )
            
            if result.get('success'):
                prediction = result.get('prediction')
                
                # 차트 데이터 생성
                chart_data = simulator.generate_chart_data(prediction)
                
                # 응답 구성
                # ML 분석 결과에서 신뢰도 정보 추출 (R² 기반)
                ml_analysis = prediction.get('ml_analysis', {})
                ml_confidence = ml_analysis.get('confidence', {})
                intuitive_confidence = ml_confidence.get('intuitive', {})
                
                response = {
                    'success': True,
                    'meta': prediction.get('meta', {}),
                    'analysis': prediction.get('analysis', {}),
                    'baseline_prediction': prediction.get('baseline_prediction', {}),
                    'scenario_prediction': prediction.get('scenario_prediction', {}),
                    'insights': prediction.get('insights', {}),
                    'chart_data': chart_data,
                    'history': prediction.get('history', []),
                    'actual_future_data': prediction.get('actual_future_data', []),
                    'confidence': {
                        'level': ml_confidence.get('level', 'MEDIUM'),
                        'reason': prediction.get('confidence_reason', ''),
                        'intuitive': intuitive_confidence  # ML R² 기반 신뢰도 정보
                    }
                }
                
                return jsonify(response)
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'AI 시뮬레이션 실패')
                }), 500
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/scenario-simulator/ml-only', methods=['POST'])
def scenario_simulator_ml_only():
    """
    ML 계산만 수행하는 빠른 API (Bedrock 호출 없음)
    
    프론트엔드에서 순차적 렌더링을 위해 사용:
    1. 먼저 이 API로 ML 결과(숫자, 차트)를 빠르게 표시
    2. 이후 /api/scenario-simulator로 전체 인사이트 로드
    """
    try:
        data = request.json
        mode = data.get('mode', 'charger_to_share')
        base_month = data.get('baseMonth')
        sim_period_months = data.get('simPeriodMonths', 6)
        
        if not base_month:
            return jsonify({'success': False, 'error': '기준월(baseMonth)을 선택해주세요'}), 400
        
        if cache['full_data'] is None:
            return jsonify({'success': False, 'error': '먼저 데이터를 로드해주세요'}), 400
        
        all_months = sorted(cache['full_data']['snapshot_month'].unique().tolist())
        earliest_month = all_months[0]
        rag_latest_month = all_months[-1]
        
        if base_month < earliest_month or base_month > rag_latest_month:
            return jsonify({
                'success': False,
                'error': f'기준월은 RAG 데이터 범위 내여야 합니다 ({earliest_month} ~ {rag_latest_month})'
            }), 400
        
        from scenario_simulator import ScenarioSimulator
        simulator = ScenarioSimulator()
        
        if mode == 'share_to_charger':
            # 모드 2: 목표 점유율 → 필요 충전기 (ML만)
            target_share = data.get('targetShare', 17.0)
            
            print(f'\n⚡ ML-Only API 호출 (모드 2)', flush=True)
            result = simulator.calculate_required_chargers_ml_only(
                base_month=base_month,
                sim_period_months=sim_period_months,
                target_share=target_share,
                full_data=cache['full_data']
            )
        else:
            # 모드 1: 충전기 → 점유율 (ML만)
            extra_chargers = data.get('extraChargers', 0)
            
            print(f'\n⚡ ML-Only API 호출 (모드 1)', flush=True)
            result = simulator.run_simulation_ml_only(
                base_month=base_month,
                sim_period_months=sim_period_months,
                extra_chargers=extra_chargers,
                full_data=cache['full_data']
            )
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify({'success': False, 'error': result.get('error', 'ML 계산 실패')}), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/scenario-simulator/months', methods=['GET'])
def get_simulator_months():
    """시뮬레이터에서 사용 가능한 기준월 목록 조회"""
    try:
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        all_months = sorted(cache['full_data']['snapshot_month'].unique().tolist())
        
        return jsonify({
            'success': True,
            'months': all_months,
            'earliest_month': all_months[0] if all_months else None,
            'latest_month': all_months[-1] if all_months else None,
            'total_months': len(all_months)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/gs-chargebi-history', methods=['GET'])
def get_gs_chargebi_history():
    """GS차지비 월별 점유율 히스토리 조회 (시뮬레이터 모드 2용)"""
    try:
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '데이터가 로드되지 않았습니다'
            }), 400
        
        # GS차지비 데이터 추출
        gs_data = cache['full_data'][cache['full_data']['CPO명'] == 'GS차지비'].copy()
        gs_data = gs_data.sort_values('snapshot_month')
        
        history = []
        for _, row in gs_data.iterrows():
            market_share = row.get('시장점유율', 0)
            if pd.notna(market_share) and market_share < 1:
                market_share = market_share * 100
            
            history.append({
                'month': row.get('snapshot_month'),
                'market_share': round(float(market_share), 2) if pd.notna(market_share) else 0,
                'total_chargers': int(row.get('총충전기', 0)) if pd.notna(row.get('총충전기')) else 0,
                'rank': int(row.get('순위', 0)) if pd.notna(row.get('순위')) else None
            })
        
        return jsonify({
            'success': True,
            'history': history
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/scenario-simulator/reliability-config', methods=['GET'])
def get_simulator_reliability_config():
    """
    시뮬레이터 신뢰도 기반 설정 조회
    
    백테스트 결과를 기반으로 신뢰도 있는 예측 범위를 반환합니다.
    - 기준 시점: RAG 최신 데이터 월로 고정
    - 최대 예측 기간: 백테스트 결과 기반 (현재 6개월)
    - 기간별 오차 통계 제공
    - 목표 점유율 범위 (신뢰도 기반)
    """
    try:
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        from scenario_simulator import ScenarioSimulator
        
        # 신뢰도 설정 조회 (GS차지비 정보 포함)
        reliability_config = ScenarioSimulator.get_reliability_config(cache['full_data'])
        
        return jsonify({
            'success': True,
            **reliability_config
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def initialize_data():
    """앱 시작 시 자동으로 모든 데이터 로드"""
    try:
        print('\n' + '='*80)
        print('🚀 데이터 자동 로드 시작...')
        print('='*80 + '\n')
        
        import sys
        sys.stdout.flush()
        
        loader = ChargingDataLoader()
        
        # 전체 월 데이터 로드
        print('🔄 전체 월 데이터 로드 중... (약 1-2분 소요)')
        df = loader.load_multiple()
        
        if df is None:
            print('❌ 데이터 로드 실패')
            return False
        
        # 캐시 저장
        cache['data'] = df
        cache['full_data'] = df.copy()
        
        # 기본 정보
        unique_months = []
        latest_month = None
        
        if 'snapshot_month' in df.columns:
            unique_months = sorted(df['snapshot_month'].unique().tolist(), reverse=True)
            latest_month = unique_months[0] if unique_months else None
        
        print(f'\n✅ 데이터 로드 완료!')
        print(f'   - 총 행 수: {len(df):,}')
        print(f'   - 포함 월: {len(unique_months)}개월')
        print(f'   - 기간: {unique_months[-1] if unique_months else "N/A"} ~ {unique_months[0] if unique_months else "N/A"}')
        print(f'   - 최신 월: {latest_month}')
        
        # 최신 월로 필터링
        if latest_month:
            df_latest = df[df['snapshot_month'] == latest_month].copy()
            cache['data'] = df_latest
            print(f'   - 기본 선택 월: {latest_month} ({len(df_latest)} 행)')
        
        # 데이터 분석 실행
        print('\n📊 데이터 분석 중...')
        analyzer = ChargingDataAnalyzer(cache['data'])
        insights = analyzer.generate_insights()
        cache['insights'] = insights
        print('✅ 데이터 분석 완료')
        
        print('\n' + '='*80)
        print('🎉 초기화 완료! 서비스 준비됨')
        print('='*80 + '\n')
        
        return True
        
    except Exception as e:
        import traceback
        print(f'\n❌ 초기화 오류: {e}')
        traceback.print_exc()
        return False

@app.route('/api/send-to-slack', methods=['POST'])
def send_dashboard_to_slack():
    """대시보드를 슬랙으로 전송"""
    try:
        data = request.json
        start_month = data.get('startMonth')
        end_month = data.get('endMonth')
        selected_months = data.get('months', [])
        
        if not end_month:
            return jsonify({
                'success': False,
                'error': '종료월을 선택해주세요'
            }), 400
        
        if cache['full_data'] is None:
            return jsonify({
                'success': False,
                'error': '먼저 데이터를 로드해주세요'
            }), 400
        
        print(f'\n📤 슬랙 전송 시작: {start_month} ~ {end_month}', flush=True)
        
        # 대시보드 데이터 생성 (기존 로직 재사용)
        from data_analyzer import ChargingDataAnalyzer
        from data_loader import ChargingDataLoader
        
        full_analyzer = ChargingDataAnalyzer(cache['full_data'])
        
        # 선택된 월들로 필터링
        current_data = None
        if selected_months:
            filtered_data = cache['full_data'][cache['full_data']['snapshot_month'].isin(selected_months)]
            if len(filtered_data) > 0:
                analyzer = ChargingDataAnalyzer(filtered_data)
                current_insights = analyzer.generate_insights()
                current_data = filtered_data
        else:
            current_insights = cache.get('insights', {})
            current_data = cache.get('data')
        
        # GS차지비 KPI 데이터 생성
        gs_kpi = None
        if current_data is not None and len(current_data) > 0 and end_month:
            gs_data = current_data[current_data['CPO명'] == 'GS차지비']
            if len(gs_data) > 0:
                end_data = gs_data[gs_data['snapshot_month'] == end_month]
                if len(end_data) > 0:
                    end_row = end_data.iloc[0]
                    
                    # 현재 시장점유율 파싱
                    try:
                        current_share_raw = end_row.get('시장점유율', '0%')
                        if isinstance(current_share_raw, str):
                            current_share = float(current_share_raw.replace('%', '').strip())
                        else:
                            current_share = float(current_share_raw) * 100 if current_share_raw < 1 else float(current_share_raw)
                    except Exception as e:
                        print(f'⚠️ 시장점유율 파싱 오류: {e}', flush=True)
                        current_share = 0.0
                    
                    # 현재 값
                    current_kpi = {
                        'market_share': round(current_share, 1),
                        'stations': int(end_row.get('충전소수', 0)),
                        'slow_chargers': int(end_row.get('완속충전기', 0)),
                        'fast_chargers': int(end_row.get('급속충전기', 0)),
                        'total_chargers': int(end_row.get('총충전기', 0))
                    }
                    
                    # 전월 대비 증감량 계산
                    all_months = sorted(gs_data['snapshot_month'].unique().tolist())
                    monthly_change = {
                        'prev_month': None,
                        'current_month': end_month,
                        'market_share_change': 0,
                        'stations': int(end_row.get('충전소증감', 0)),
                        'slow_chargers': int(end_row.get('완속증감', 0)),
                        'fast_chargers': int(end_row.get('급속증감', 0)),
                        'total_chargers': int(end_row.get('총증감', 0))
                    }
                    
                    if end_month in all_months:
                        current_idx = all_months.index(end_month)
                        if current_idx > 0:
                            prev_month = all_months[current_idx - 1]
                            prev_data = gs_data[gs_data['snapshot_month'] == prev_month]
                            if len(prev_data) > 0:
                                prev_row = prev_data.iloc[0]
                                monthly_change['prev_month'] = prev_month
                                
                                # 전월 시장점유율
                                try:
                                    prev_share_raw = prev_row.get('시장점유율', '0%')
                                    if isinstance(prev_share_raw, str):
                                        prev_share = float(prev_share_raw.replace('%', '').strip())
                                    else:
                                        prev_share = float(prev_share_raw) * 100 if prev_share_raw < 1 else float(prev_share_raw)
                                    
                                    share_change = round(current_share - prev_share, 1)
                                    monthly_change['market_share_change'] = share_change
                                except Exception as e:
                                    print(f'⚠️ 전월 시장점유율 파싱 오류: {e}', flush=True)
                                    monthly_change['market_share_change'] = 0
                    
                    gs_kpi = {
                        'current': current_kpi,
                        'monthly_change': monthly_change
                    }
        
        # 요약 테이블 데이터 (엑셀에서 직접 추출)
        summary_table = None
        if current_data is not None and len(current_data) > 0:
            data_source = None
            if end_month and 'snapshot_month' in current_data.columns:
                end_month_data = current_data[current_data['snapshot_month'] == end_month]
                if len(end_month_data) > 0 and 'data_source' in end_month_data.columns:
                    data_source = end_month_data['data_source'].iloc[0]
            
            if not data_source and 'data_source' in current_data.columns:
                sorted_data = current_data.sort_values('snapshot_month', ascending=False)
                data_source = sorted_data['data_source'].iloc[0]
            
            if data_source:
                loader = ChargingDataLoader()
                summary_table = loader.extract_summary_data(data_source)
        
        # 대시보드 데이터 구성
        dashboard_data = {
            'gs_kpi': gs_kpi,
            'summary_table': summary_table,
            'start_month': start_month,
            'end_month': end_month
        }
        
        print(f'📊 대시보드 데이터 생성 완료', flush=True)
        print(f'   - GS KPI: {gs_kpi is not None}', flush=True)
        print(f'   - 요약 테이블: {summary_table is not None}', flush=True)
        
        # 슬랙 전송
        from slack_sender import SlackDashboardSender, send_to_slack_webhook
        
        sender = SlackDashboardSender()
        
        # 슬랙 메시지 생성
        slack_message = sender.create_slack_message(dashboard_data, start_month or end_month, end_month)
        print(f'📝 슬랙 메시지 생성 완료 ({len(slack_message)} 자)', flush=True)
        
        # 슬랙으로 전송
        result = send_to_slack_webhook(slack_message)
        
        if result['success']:
            print(f'✅ 슬랙 전송 성공!', flush=True)
            return jsonify({
                'success': True,
                'message': '슬랙 전송이 완료되었습니다!',
                'slack_result': result
            })
        else:
            print(f'❌ 슬랙 전송 실패: {result["message"]}', flush=True)
            return jsonify({
                'success': False,
                'error': f'슬랙 전송 실패: {result["message"]}'
            }), 500
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export-dashboard', methods=['POST'])
def export_dashboard():
    """대시보드를 PDF/이미지로 내보내기"""
    try:
        data = request.json
        start_month = data.get('startMonth')
        end_month = data.get('endMonth')
        export_format = data.get('format', 'html')  # html, pdf, png
        
        if not end_month:
            return jsonify({
                'success': False,
                'error': '종료월을 선택해주세요'
            }), 400
        
        # 대시보드 데이터 생성 (위와 동일한 로직)
        # ... (슬랙 전송과 동일한 대시보드 데이터 생성 로직)
        
        from slack_sender import SlackDashboardSender
        sender = SlackDashboardSender()
        
        if export_format == 'html':
            # HTML 파일로 저장
            filepath = sender.save_dashboard_html({}, start_month or end_month, end_month)
            
            return jsonify({
                'success': True,
                'message': 'HTML 파일이 생성되었습니다',
                'filepath': filepath
            })
        else:
            return jsonify({
                'success': False,
                'error': f'지원하지 않는 형식: {export_format}'
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/slack-send', methods=['POST'])
def slack_send_simple():
    """간단한 슬랙 전송 API"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({
                'success': False,
                'error': '메시지가 없습니다'
            }), 400
        
        print(f"📤 슬랙 전송: {len(message)} 자")
        
        # 슬랙 Webhook URL (환경변수에서 로드)
        slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
        
        # 슬랙으로 전송
        import requests
        payload = {
            "text": message,
            "mrkdwn": True
        }
        
        response = requests.post(
            slack_webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200 and response.text == 'ok':
            print("✅ 슬랙 전송 성공!")
            return jsonify({
                'success': True,
                'message': '슬랙 전송 성공!'
            })
        else:
            print(f"❌ 슬랙 전송 실패: {response.status_code} - {response.text}")
            return jsonify({
                'success': False,
                'error': f'슬랙 전송 실패: {response.status_code}'
            }), 500
        
    except Exception as e:
        print(f"❌ 슬랙 전송 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # 앱 시작 시 데이터 자동 로드
    initialize_data()
    
    # use_reloader=False로 설정하여 파일 변경 시 자동 재시작 방지
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False, threaded=True)