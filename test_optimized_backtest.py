"""
최적화된 ML 로직 백테스트 검증

변경사항:
1. LinearRegression → Ridge(alpha=10.0)
2. Ratio 70% + Direct 30% → Ratio 100%
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge

from data_loader import ChargingDataLoader


def load_full_data():
    """전체 RAG 데이터 로드"""
    loader = ChargingDataLoader()
    full_data = loader.load_multiple()
    return full_data


def extract_histories(full_data: pd.DataFrame):
    """GS차지비 및 시장 히스토리 추출"""
    gs_data = full_data[full_data['CPO명'] == 'GS차지비'].copy()
    gs_data = gs_data.sort_values('snapshot_month')
    
    gs_history = []
    for _, row in gs_data.iterrows():
        market_share = row.get('시장점유율', 0)
        if pd.notna(market_share) and market_share < 1:
            market_share = market_share * 100
        
        gs_history.append({
            'month': row.get('snapshot_month'),
            'total_chargers': int(row.get('총충전기', 0)) if pd.notna(row.get('총충전기')) else 0,
            'market_share': round(float(market_share), 4) if pd.notna(market_share) else 0,
        })
    
    all_months = sorted(full_data['snapshot_month'].unique().tolist())
    market_history = []
    for month in all_months:
        month_data = full_data[full_data['snapshot_month'] == month]
        if len(month_data) > 0:
            total_chargers = month_data['총충전기'].sum()
            market_history.append({
                'month': month,
                'total_chargers': int(total_chargers),
            })
    
    return gs_history, market_history


def run_optimized_backtest(gs_history, market_history, test_months):
    """최적화된 방식 (Ridge + Ratio 100%) 백테스트"""
    n = len(gs_history)
    if n < test_months + 3:
        return None
    
    # 학습/테스트 분리
    train_gs = gs_history[:-test_months]
    train_market = market_history[:-test_months]
    test_gs = gs_history[-test_months:]
    
    n_train = len(train_gs)
    X_train = np.arange(n_train).reshape(-1, 1)
    
    gs_chargers = np.array([h['total_chargers'] for h in train_gs])
    market_chargers = np.array([m['total_chargers'] for m in train_market[:n_train]])
    
    actual_shares = [h['market_share'] for h in test_gs]
    
    # Ridge(alpha=10.0) 모델 학습
    gs_model = Ridge(alpha=10.0)
    gs_model.fit(X_train, gs_chargers)
    
    market_model = Ridge(alpha=10.0)
    market_model.fit(X_train, market_chargers)
    
    # 예측 (Ratio 100%)
    errors = []
    predictions = []
    
    for i in range(1, test_months + 1):
        future_idx = n_train + i - 1
        X_future = np.array([[future_idx]])
        
        pred_gs = gs_model.predict(X_future)[0]
        pred_market = market_model.predict(X_future)[0]
        pred_share = (pred_gs / pred_market) * 100 if pred_market > 0 else 0
        
        actual = actual_shares[i-1]
        abs_error = abs(pred_share - actual)
        pct_error = abs_error / actual * 100 if actual > 0 else 0
        
        errors.append({
            'month': i,
            'predicted': pred_share,
            'actual': actual,
            'abs_error': abs_error,
            'pct_error': pct_error
        })
        predictions.append(pred_share)
    
    mae = np.mean([e['abs_error'] for e in errors])
    mape = np.mean([e['pct_error'] for e in errors])
    
    return {
        'mae': mae,
        'mape': mape,
        'errors': errors,
        'predictions': predictions,
        'actuals': actual_shares
    }


def run_original_backtest(gs_history, market_history, test_months):
    """기존 방식 (LinearRegression + Ratio 70%) 백테스트"""
    from sklearn.linear_model import LinearRegression
    
    n = len(gs_history)
    if n < test_months + 3:
        return None
    
    # 학습/테스트 분리
    train_gs = gs_history[:-test_months]
    train_market = market_history[:-test_months]
    test_gs = gs_history[-test_months:]
    
    n_train = len(train_gs)
    X_train = np.arange(n_train).reshape(-1, 1)
    
    gs_chargers = np.array([h['total_chargers'] for h in train_gs])
    gs_shares = np.array([h['market_share'] for h in train_gs])
    market_chargers = np.array([m['total_chargers'] for m in train_market[:n_train]])
    
    actual_shares = [h['market_share'] for h in test_gs]
    
    # LinearRegression 모델 학습
    gs_model = LinearRegression()
    gs_model.fit(X_train, gs_chargers)
    
    market_model = LinearRegression()
    market_model.fit(X_train, market_chargers)
    
    share_model = LinearRegression()
    share_model.fit(X_train, gs_shares)
    
    # 예측 (Ratio 70% + Direct 30%)
    errors = []
    
    for i in range(1, test_months + 1):
        future_idx = n_train + i - 1
        X_future = np.array([[future_idx]])
        
        pred_gs = gs_model.predict(X_future)[0]
        pred_market = market_model.predict(X_future)[0]
        pred_ratio = (pred_gs / pred_market) * 100 if pred_market > 0 else 0
        pred_direct = share_model.predict(X_future)[0]
        
        # 기존 가중치: Ratio 70%, Direct 30%
        pred_share = pred_ratio * 0.7 + pred_direct * 0.3
        
        actual = actual_shares[i-1]
        abs_error = abs(pred_share - actual)
        pct_error = abs_error / actual * 100 if actual > 0 else 0
        
        errors.append({
            'abs_error': abs_error,
            'pct_error': pct_error
        })
    
    mae = np.mean([e['abs_error'] for e in errors])
    mape = np.mean([e['pct_error'] for e in errors])
    
    return {
        'mae': mae,
        'mape': mape
    }


def main():
    print("=" * 60)
    print("🔬 최적화된 ML 로직 백테스트 검증")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n변경사항:")
    print("  - LinearRegression → Ridge(alpha=10.0)")
    print("  - Ratio 70% + Direct 30% → Ratio 100%")
    
    # 데이터 로드
    print("\n📥 데이터 로드 중...")
    full_data = load_full_data()
    if full_data is None:
        print("❌ 데이터 로드 실패")
        return
    
    gs_history, market_history = extract_histories(full_data)
    print(f"✅ 데이터 로드 완료: {len(gs_history)}개월")
    
    print("\n" + "=" * 60)
    print("📊 기존 vs 최적화 비교")
    print("=" * 60)
    
    results = []
    
    for test_months in [1, 2, 3, 4, 5, 6]:
        if len(gs_history) < test_months + 4:
            continue
        
        original = run_original_backtest(gs_history, market_history, test_months)
        optimized = run_optimized_backtest(gs_history, market_history, test_months)
        
        if original and optimized:
            improvement = (original['mae'] - optimized['mae']) / original['mae'] * 100
            
            print(f"\n{test_months}개월 예측:")
            print(f"   기존 (LR + 70/30): MAE={original['mae']:.4f}, MAPE={original['mape']:.2f}%")
            print(f"   최적화 (Ridge + 100/0): MAE={optimized['mae']:.4f}, MAPE={optimized['mape']:.2f}%")
            print(f"   → 개선율: {improvement:+.1f}%")
            
            # 상세 예측 결과
            print(f"\n   예측 vs 실제:")
            for e in optimized['errors']:
                print(f"      {e['month']}개월: 실제={e['actual']:.2f}%, 예측={e['predicted']:.2f}% (오차:{e['abs_error']:.3f})")
            
            results.append({
                'test_months': test_months,
                'original_mae': original['mae'],
                'optimized_mae': optimized['mae'],
                'improvement': improvement
            })
    
    # 요약
    if results:
        print("\n" + "=" * 60)
        print("📊 종합 결과")
        print("=" * 60)
        
        avg_original = np.mean([r['original_mae'] for r in results])
        avg_optimized = np.mean([r['optimized_mae'] for r in results])
        avg_improvement = np.mean([r['improvement'] for r in results])
        
        print(f"\n평균 MAE:")
        print(f"   기존: {avg_original:.4f}")
        print(f"   최적화: {avg_optimized:.4f}")
        print(f"   평균 개선율: {avg_improvement:+.1f}%")
        
        if avg_improvement > 0:
            print(f"\n✅ 최적화 성공! 평균 {avg_improvement:.1f}% 오차 감소")
        else:
            print(f"\n⚠️ 최적화 효과 미미 또는 역효과")


if __name__ == "__main__":
    main()
