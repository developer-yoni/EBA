"""
요약 테이블 추출 테스트
"""
from data_loader import ChargingDataLoader

loader = ChargingDataLoader()

# 사용 가능한 파일 목록 조회
files = loader.list_available_files()
print(f'📂 사용 가능한 파일: {len(files)}개\n')

if files:
    # 최신 파일로 테스트
    latest_file = files[0]
    print(f'📄 테스트 파일: {latest_file["filename"]}')
    print(f'🔑 S3 Key: {latest_file["key"]}\n')
    
    # 요약 데이터 추출
    summary = loader.extract_summary_data(latest_file['key'])
    
    if summary:
        print('✅ 요약 데이터 추출 성공!\n')
        print('📊 전체CPO:')
        print(f'  - 충전사업자: {summary["total"]["cpos"]}')
        print(f'  - 충전소: {summary["total"]["stations"]}')
        print(f'  - 완속충전기: {summary["total"]["slow_chargers"]}')
        print(f'  - 급속충전기: {summary["total"]["fast_chargers"]}')
        print(f'  - 전체충전기: {summary["total"]["total_chargers"]}')
        
        print('\n📈 당월증감량:')
        print(f'  - 충전사업자: {summary["change"]["cpos"]}')
        print(f'  - 충전소: {summary["change"]["stations"]}')
        print(f'  - 완속충전기: {summary["change"]["slow_chargers"]}')
        print(f'  - 급속충전기: {summary["change"]["fast_chargers"]}')
        print(f'  - 전체충전기: {summary["change"]["total_chargers"]}')
    else:
        print('❌ 요약 데이터 추출 실패')
else:
    print('❌ 사용 가능한 파일이 없습니다.')
