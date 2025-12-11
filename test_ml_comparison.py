"""
ML 예측 방법 비교 테스트

두 가지 방법 비교:
1. ratio: GS충전기/시장전체 각각 예측 후 점유율 계산
2. direct: 점유율 직접 예측 (현재 방식)
"""

import sys
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 60)
    print("ML 예측 방법 비교 테스트")
    print("=" * 60)
    
    # RAG 데이터 (실제 추출값)
    data = [
        {'month': '2024-12', 'gs_chargers': 71836, 'share': 18.07},
        {'month': '2025-01', 'gs_chargers': 72519, 'share': 17.80},
        {'month': '2025-02', 'gs_chargers': 72943, 'share': 17.73},
        {'month': '2025-03', 'gs_chargers': 73435, 'share': 17.47},
        {'month': '2025-04', 'gs_chargers': 72474, 'share': 16.99},
        {'month': '2025-05', 'gs_chargers': 72693, 'share': 16.80},
        {'month': '2025-06', 'gs_chargers': 72868, 'share': 16.67},
        {'month': '2025-07', 'gs_chargers': 72927, 'share': 16.45},
        {'month': '2025-08', 'gs_chargers': 72924, 'share': 16.18},
        {'month': '2025-09', 'gs_chargers': 73238, 'share': 16.13},
        {'month': '2025-10', 'gs_chargers': 73290, 'share': 16.18},
        {'month': '2025-11', 'gs_chargers': 73912, 'share': 16.08},
    ]
    
    # 시장 전체 계산
    for d in data:
        d['market'] = int(d['gs_chargers'] / (d['share'] / 100))
    
    print(f"\n데이터 기간: {data[0]['month']} ~ {data[-1]['month']} ({len(data)}개월)")
    print(f"GS 점유율 변화: {data[0]['share']:.2f}% → {data[-1]['share']:.2f}%")
    print(f"GS 충전기 변화: {data[0]['gs_chargers']:,} → {data[-1]['gs_chargers']:,}")
    
    # 백테스트 실행
    print("\n" + "=" * 60)
    print("백테스트 결과")
    print("=" * 60)
    
    results = []
    
    # 다양한 기준월에서 테스트
    for base_idx in range(3, len(data) - 1):
        base_month = data[base_idx]['month']
        
        # 학습 데이터 (0 ~ base_idx)
        X_train = np.arange(base_idx + 1).reshape(-1, 1)
        gs_train = np.array([d['gs_chargers'] for d in data[:base_idx + 1]])
        market_train = np.array([d['market'] for d in data[:base_idx + 1]])
        share_train = np.array([d['share'] for d in data[:base_idx + 1]])
        
        # 모델 학습
        lr_gs = LinearRegression().fit(X_train, gs_train)
        lr_market = LinearRegression().fit(X_train, market_train)
        lr_share = LinearRegression().fit(X_train, share_train)
        
        # 1~6개월 예측
        for months_ahead in range(1, min(7, len(data) - base_idx)):
            target_idx = base_idx + months_ahead
            actual_share = data[target_idx]['share']
            
            # 방법 1: ratio (GS/시장 각각 예측)
            pred_gs = lr_gs.predict([[target_idx]])[0]
            pred_market = lr_market.predict([[target_idx]])[0]
            pred_ratio = (pred_gs / pred_market) * 100
            
            # 방법 2: direct (점유율 직접 예측)
            pred_direct = lr_share.predict([[target_idx]])[0]
            
            err_ratio = abs(pred_ratio - actual_share)
            err_direct = abs(pred_direct - actual_share)
            
            results.append({
                'base_month': base_month,
                'target_month': data[target_idx]['month'],
                'months_ahead': months_ahead,
                'actual': actual_share,
                'pred_ratio': pred_ratio,
                'pred_direct': pred_direct,
                'err_ratio': err_ratio,
                'err_direct': err_direct,
                'better': 'ratio' if err_ratio < err_direct else 'direct'
            })
    
    # 결과 출력
    print(f"\n{'기준월':<10} {'예측월':<10} {'기간':<6} {'실제':<8} {'ratio':<8} {'direct':<8} {'ratio오차':<10} {'direct오차':<10} {'승자':<8}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['base_month']:<10} {r['target_month']:<10} {r['months_ahead']:<6} "
              f"{r['actual']:<8.2f} {r['pred_ratio']:<8.2f} {r['pred_direct']:<8.2f} "
              f"{r['err_ratio']:<10.4f} {r['err_direct']:<10.4f} {r['better']:<8}")
    
    # 요약 통계
    print("\n" + "=" * 60)
    print("요약 통계")
    print("=" * 60)
    
    # 기간별 평균 오차
    for months in range(1, 7):
        period_results = [r for r in results if r['months_ahead'] == months]
        if period_results:
            avg_ratio = np.mean([r['err_ratio'] for r in period_results])
            avg_direct = np.mean([r['err_direct'] for r in period_results])
            ratio_wins = sum(1 for r in period_results if r['better'] == 'ratio')
            
            print(f"{months}개월 예측: ratio MAE={avg_ratio:.4f}%p, direct MAE={avg_direct:.4f}%p, "
                  f"ratio 승률={ratio_wins}/{len(period_results)}")
    
    # 전체 평균
    total_ratio = np.mean([r['err_ratio'] for r in results])
    total_direct = np.mean([r['err_direct'] for r in results])
    ratio_total_wins = sum(1 for r in results if r['better'] == 'ratio')
    
    print(f"\n전체 평균: ratio MAE={total_ratio:.4f}%p, direct MAE={total_direct:.4f}%p")
    print(f"ratio 승률: {ratio_total_wins}/{len(results)} ({ratio_total_wins/len(results)*100:.1f}%)")
    
    if total_ratio < total_direct:
        improvement = (total_direct - total_ratio) / total_direct * 100
        print(f"\n✅ 결론: ratio 방법이 {improvement:.1f}% 더 정확!")
        print("   → GS충전기와 시장전체를 각각 예측 후 점유율 계산하는 방식 권장")
    else:
        improvement = (total_ratio - total_direct) / total_ratio * 100
        print(f"\n✅ 결론: direct 방법이 {improvement:.1f}% 더 정확!")
        print("   → 점유율 직접 예측 방식 유지")

if __name__ == "__main__":
    main()
