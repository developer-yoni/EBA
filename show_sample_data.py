"""
실제 데이터 샘플 확인
"""
from data_loader import ChargingDataLoader

loader = ChargingDataLoader()
df = loader.load_latest()

print('=' * 80)
print('📊 한국 전기차 충전 인프라 현황 데이터')
print('=' * 80)
print()

print(f'총 CPO 수: {len(df)}개')
print()

print('🏆 상위 10개 CPO (총충전기 기준):')
print('-' * 80)
top10 = df.nlargest(10, '총충전기')[['CPO명', '순위', '충전소수', '완속충전기', '급속충전기', '총충전기', '시장점유율']]

for idx, row in top10.iterrows():
    print(f"{int(row['순위']):2d}. {row['CPO명']:20s} | "
          f"충전소: {int(row['충전소수']):6,}개 | "
          f"완속: {int(row['완속충전기']):6,}기 | "
          f"급속: {int(row['급속충전기']):5,}기 | "
          f"총: {int(row['총충전기']):6,}기 | "
          f"점유율: {float(row['시장점유율'])*100:5.2f}%")

print()
print('📈 전체 통계:')
print('-' * 80)
print(f"총 충전소: {int(df['충전소수'].sum()):,}개")
print(f"총 완속충전기: {int(df['완속충전기'].sum()):,}기")
print(f"총 급속충전기: {int(df['급속충전기'].sum()):,}기")
print(f"총 충전기: {int(df['총충전기'].sum()):,}기")
print()
