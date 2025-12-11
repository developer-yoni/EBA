#!/usr/bin/env python3
"""
빠른 백테스트 - 개선된 ratio 방식 검증
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# RAG 데이터 (실제 추출값)
DATA = [
    {'month': '2024-12', 'gs': 71836, 'share': 18.07},
    {'month': '2025-01', 'gs': 72519, 'share': 17.80},
    {'month': '2025-02', 'gs': 72943, 'share': 17.73},
    {'month': '2025-03', 'gs': 73435, 'share': 17.47},
    {'month': '2025-04', 'gs': 72474, 'share': 16.99},
    {'month': '2025-05', 'gs': 72693, 'share': 16.80},
    {'month': '2025-06', 'gs': 72868, 'share': 16.67},
    {'month': '2025-07', 'gs': 72927, 'share': 16.45},
    {'month': '2025-08', 'gs': 72924, 'share': 16.18},
    {'month': '2025-09', 'gs': 73238, 'share': 16.13},
    {'month': '2025-10', 'gs': 73290, 'share': 16.18},
    {'month': '2025-11', 'gs': 73912, 'share': 16.08},
]

# 시장 전체 계산
for d in DATA:
    d['market'] = int(d['gs'] / (d['share'] / 100))

def predict_share(train_data, months_ahead, method='ratio'):
    """예측 수행"""
    n = len(train_data)
    X = np.arange(n).reshape(-1, 1)
    
    gs = np.array([d['gs'] for d in train_data])
    market = np.array([d['market'] for d in train_data])
    shares = np.array([d['share'] for d in train_data])
    
    lr_gs = LinearRegression().fit(X, gs)
    lr_market = LinearRegression().fit(X, market)
    lr_share = LinearRegression().fit(X, shares)
    
    future_idx = n + months_ahead - 1
    X_future = np.array([[future_idx]])
    
    pred_gs = lr_gs.predict(X_future)[0]
    pred_market = lr_market.predict(X_future)[0]
    pred_share_direct = lr_share.predict(X_future)[0]
    pred_share_ratio = (pred_gs / pred_market) * 100
    
    if method == 'ratio':
        return pred_share_ratio
    elif method == 'direct':
        return pred_share_direct
    else:  # weighted
        return pred_share_ratio * 0.7 + pred_share_direct * 0.3

def run_backtest():
    """백테스트 실행"""
    print("=" * 70)
    print("개선된 ML 예측 방식 백테스트")
    print("=" * 70)
    print(f"데이터: {DATA[0]['month']} ~ {DATA[-1]['month']} ({len(DATA)}개월)")
    print()
    
    results = {'ratio': [], 'direct': [], 'weighted': []}
    
    # 다양한 기준월에서 테스트
    for base_idx in range(3, len(DATA) - 1):
        train = DATA[:base_idx + 1]
        
        for months_ahead in range(1, min(7, len(DATA) - base_idx)):
            target_idx = base_idx + months_ahead
            actual = DATA[target_idx]['share']
            
            for method in ['ratio', 'direct', 'weighted']:
                pred = predict_share(train, months_ahead, method)
                error = abs(pred - actual)
                results[method].append({
                    'base': DATA[base_idx]['month'],
                    'target': DATA[target_idx]['month'],
                    'months': months_ahead,
                    'actual': actual,
                    'pred': pred,
                    'error': error
                })
    
    # 결과 출력
    print(f"{'기준월':<10} {'예측월':<10} {'기간':<5} {'실제':<7} {'ratio':<7} {'direct':<7} {'weighted':<7}")
    print("-" * 70)
    
    for i in range(len(results['ratio'])):
        r = results['ratio'][i]
        d = results['direct'][i]
        w = results['weighted'][i]
        print(f"{r['base']:<10} {r['target']:<10} {r['months']:<5} "
              f"{r['actual']:<7.2f} {r['pred']:<7.2f} {d['pred']:<7.2f} {w['pred']:<7.2f}")
    
    # 요약
    print("\n" + "=" * 70)
    print("요약 (MAE = 평균 절대 오차)")
    print("=" * 70)
    
    for method in ['ratio', 'direct', 'weighted']:
        mae = np.mean([r['error'] for r in results[method]])
        print(f"{method:>10}: MAE = {mae:.4f}%p")
    
    # 기간별 분석
    print("\n기간별 MAE:")
    for months in range(1, 7):
        print(f"  {months}개월: ", end="")
        for method in ['ratio', 'direct', 'weighted']:
            period_results = [r for r in results[method] if r['months'] == months]
            if period_results:
                mae = np.mean([r['error'] for r in period_results])
                print(f"{method}={mae:.3f}  ", end="")
        print()
    
    # 최종 결론
    maes = {m: np.mean([r['error'] for r in results[m]]) for m in results}
    best = min(maes, key=maes.get)
    print(f"\n✅ 최적 방법: {best} (MAE={maes[best]:.4f}%p)")
    
    if best != 'direct':
        improvement = (maes['direct'] - maes[best]) / maes['direct'] * 100
        print(f"   기존 direct 대비 {improvement:.1f}% 개선")

if __name__ == "__main__":
    run_backtest()
