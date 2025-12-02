"""
전체 플로우 테스트 (웹 API 없이)
"""
from data_loader import ChargingDataLoader
from data_analyzer import ChargingDataAnalyzer
import json

print('=' * 80)
print('🧪 전체 플로우 테스트')
print('=' * 80)
print()

# 1. 데이터 로드
print('1️⃣ 데이터 로드...')
loader = ChargingDataLoader()
df = loader.load_latest()
print(f'✅ {len(df)} 행 로드 완료\n')

# 2. 데이터 분석
print('2️⃣ 데이터 분석...')
analyzer = ChargingDataAnalyzer(df)
insights = analyzer.generate_insights()
print('✅ 분석 완료\n')

# 3. JSON 직렬화 테스트
print('3️⃣ JSON 직렬화 테스트...')
try:
    json_str = json.dumps(insights, ensure_ascii=False, indent=2, default=str)
    print(f'✅ JSON 직렬화 성공 ({len(json_str)} 문자)\n')
    
    # 결과 미리보기
    print('📊 분석 결과 요약:')
    print(f"  - 총 레코드: {insights['summary']['total_records']}")
    print(f"  - 컬럼 수: {len(insights['summary']['columns'])}")
    
    if insights['cpo_analysis']:
        print(f"  - CPO 분석: {insights['cpo_analysis'].get('summary', 'N/A')}")
    
    if insights['region_analysis']:
        print(f"  - 지역 분석: {insights['region_analysis'].get('summary', 'N/A')}")
    
    if insights['trend']:
        print(f"  - 트렌드: {insights['trend'].get('summary', 'N/A')}")
    
    print()
    print('🎉 모든 테스트 통과!')
    
except Exception as e:
    print(f'❌ JSON 직렬화 실패: {e}')
    import traceback
    traceback.print_exc()
