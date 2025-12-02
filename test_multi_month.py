"""
여러 월 데이터 로드 및 트렌드 분석 테스트
"""
from data_loader import ChargingDataLoader
from data_analyzer import ChargingDataAnalyzer

print('=' * 80)
print('📊 다중 월 데이터 트렌드 분석')
print('=' * 80)
print()

loader = ChargingDataLoader()

# 사용 가능한 파일 목록
files = loader.list_available_files()
print(f'📂 사용 가능한 파일: {len(files)}개')
for f in files[:5]:
    print(f"  - {f['filename']}")
print()

# 여러 월 데이터 로드
print('📥 여러 월 데이터 로드 중...')
df = loader.load_multiple()

if df is not None:
    print(f'✅ 총 {len(df)} 행 로드 완료')
    print()
    
    # 월별 요약
    print('📅 월별 데이터 요약:')
    monthly_summary = df.groupby('snapshot_month').agg({
        'CPO명': 'count',
        '총충전기': 'sum',
        '충전소수': 'sum'
    }).reset_index()
    monthly_summary.columns = ['월', 'CPO수', '총충전기', '총충전소']
    
    for _, row in monthly_summary.iterrows():
        print(f"  {row['월']}: CPO {int(row['CPO수'])}개, "
              f"충전기 {int(row['총충전기']):,}기, "
              f"충전소 {int(row['총충전소']):,}개")
    
    print()
    
    # 트렌드 분석
    print('📈 트렌드 분석...')
    analyzer = ChargingDataAnalyzer(df)
    insights = analyzer.generate_insights()
    
    if insights['trend']:
        print(f"✅ 트렌드 데이터: {insights['trend']['summary']}")
    
    print()
    print('🎉 다중 월 분석 완료!')
