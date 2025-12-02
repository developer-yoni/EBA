"""
월 목록 조회 테스트
"""
from data_loader import ChargingDataLoader

loader = ChargingDataLoader()

# 파일 목록 조회
files = loader.list_available_files()
print(f'📂 파일 수: {len(files)}개\n')

if files:
    print('📄 파일 목록:')
    for i, f in enumerate(files[:5], 1):
        filename = f['filename']
        print(f'{i}. {filename}')
        
        # 날짜 파싱 테스트
        snapshot_date, snapshot_month = loader.parse_snapshot_date_from_filename(filename)
        print(f'   → 날짜: {snapshot_date}, 월: {snapshot_month}')
    
    print('\n📅 추출된 월 목록:')
    months = []
    for f in files:
        filename = f['filename']
        snapshot_date, snapshot_month = loader.parse_snapshot_date_from_filename(filename)
        if snapshot_month:
            months.append(snapshot_month)
    
    unique_months = sorted(list(set(months)), reverse=True)
    print(f'총 {len(unique_months)}개월: {unique_months}')
else:
    print('❌ 파일이 없습니다.')
