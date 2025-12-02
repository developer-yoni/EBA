"""
실제 데이터 구조 확인
"""
from data_loader import ChargingDataLoader
import pandas as pd

print('=' * 80)
print('🔍 데이터 구조 상세 분석')
print('=' * 80)
print()

# 데이터 로드
loader = ChargingDataLoader()
df = loader.load_latest()

print(f'📊 기본 정보')
print(f'  - 총 행 수: {len(df)}')
print(f'  - 총 컬럼 수: {len(df.columns)}')
print()

print('📋 컬럼 목록:')
for i, col in enumerate(df.columns, 1):
    print(f'  {i:2d}. {col}')
print()

print('👀 데이터 샘플 (처음 5행):')
print(df.head().to_string())
print()

print('📈 데이터 타입:')
print(df.dtypes)
print()

print('🔢 각 컬럼의 고유값 개수:')
for col in df.columns:
    unique_count = df[col].nunique()
    print(f'  {col}: {unique_count}개')
print()

print('💡 해석:')
print(f'  - 143개 레코드 = 엑셀 파일의 데이터 행 143개')
print(f'  - 각 행은 하나의 데이터 포인트를 나타냅니다')
print(f'  - 예: CPO별 데이터, 지역별 데이터, 또는 충전소별 데이터')
