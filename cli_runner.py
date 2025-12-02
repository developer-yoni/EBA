"""
CLI 기반 분석 실행 (웹 서버 없이 사용)
"""
import sys
from data_loader import ChargingDataLoader
from data_analyzer import ChargingDataAnalyzer
from ai_report_generator import AIReportGenerator
import json

def main():
    print('=' * 80)
    print('⚡ 한국 전기차 충전 인프라 자동 분석 리포트')
    print('=' * 80)
    print()
    
    # 1. 데이터 로드
    print('📥 STEP 1: 데이터 로드')
    print('-' * 80)
    loader = ChargingDataLoader()
    
    # 사용 가능한 파일 목록
    files = loader.list_available_files()
    if not files:
        print('❌ S3에 사용 가능한 파일이 없습니다.')
        return
    
    print(f'📂 사용 가능한 파일: {len(files)}개')
    for i, f in enumerate(files[:5], 1):
        print(f'  {i}. {f["filename"]} ({f["last_modified"]})')
    
    # 최신 파일 로드
    df = loader.load_latest()
    if df is None:
        print('❌ 데이터 로드 실패')
        return
    
    print()
    
    # 2. 데이터 분석
    print('📊 STEP 2: 데이터 분석')
    print('-' * 80)
    analyzer = ChargingDataAnalyzer(df)
    insights = analyzer.generate_insights()
    
    print(f'✅ 분석 완료')
    print(f'  - 총 레코드: {insights["summary"]["total_records"]:,}개')
    print(f'  - 컬럼 수: {len(insights["summary"]["columns"])}개')
    print()
    
    # 3. AI 리포트 생성
    print('🤖 STEP 3: AI 리포트 생성')
    print('-' * 80)
    generator = AIReportGenerator()
    report = generator.generate_full_report(insights)
    
    print()
    
    # 4. 리포트 출력
    print('=' * 80)
    print('📝 분석 리포트')
    print('=' * 80)
    print()
    
    if report['executive_summary']:
        print('■ 경영진 요약')
        print('-' * 80)
        print(report['executive_summary'])
        print()
    
    if report['cpo_analysis']:
        print('■ CPO 분석')
        print('-' * 80)
        print(report['cpo_analysis'])
        print()
    
    if report['regional_analysis']:
        print('■ 지역별 분석')
        print('-' * 80)
        print(report['regional_analysis'])
        print()
    
    if report['trend_forecast']:
        print('■ 트렌드 및 예측')
        print('-' * 80)
        print(report['trend_forecast'])
        print()
    
    # 5. 리포트 저장
    output_file = 'charging_infrastructure_report.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'insights': insights,
            'report': report
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print('=' * 80)
    print(f'✅ 리포트가 {output_file}에 저장되었습니다.')
    print('=' * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n⚠️ 사용자에 의해 중단되었습니다.')
        sys.exit(0)
    except Exception as e:
        print(f'\n❌ 오류 발생: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
