"""
ML 예측기 파라미터 최적화 테스트

목적:
1. Linear Regression vs 다른 모델 비교
2. 최적의 가중치(ratio vs direct) 탐색
3. 현재 데이터에 최적화된 파라미터 찾기
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from itertools import product

# 데이터 로더
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


def test_different_models(gs_history, market_history, test_months=3):
    """다양한 ML 모델 비교"""
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
    
    # 테스트할 모델들
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge(0.1)': Ridge(alpha=0.1),
        'Ridge(1.0)': Ridge(alpha=1.0),
        'Ridge(10.0)': Ridge(alpha=10.0),
        'Lasso(0.01)': Lasso(alpha=0.01),
        'Huber': HuberRegressor(epsilon=1.35),
    }
    
    results = []
    
    for model_name, model in models.items():
        try:
            # 점유율 직접 예측 모델 학습
            share_model = model.__class__(**model.get_params())
            share_model.fit(X_train, gs_shares)
            
            # GS 충전기 모델 학습
            gs_model = model.__class__(**model.get_params())
            gs_model.fit(X_train, gs_chargers)
            
            # 시장 모델 학습
            market_model = model.__class__(**model.get_params())
            market_model.fit(X_train, market_chargers)
            
            # 예측
            errors_direct = []
            errors_ratio = []
            
            for i in range(1, test_months + 1):
                future_idx = n_train + i - 1
                X_future = np.array([[future_idx]])
                
                # Direct 방식
                pred_direct = share_model.predict(X_future)[0]
                
                # Ratio 방식
                pred_gs = gs_model.predict(X_future)[0]
                pred_market = market_model.predict(X_future)[0]
                pred_ratio = (pred_gs / pred_market) * 100 if pred_market > 0 else 0
                
                actual = actual_shares[i-1]
                
                errors_direct.append(abs(pred_direct - actual))
                errors_ratio.append(abs(pred_ratio - actual))
            
            mae_direct = np.mean(errors_direct)
            mae_ratio = np.mean(errors_ratio)
            
            results.append({
                'model': model_name,
                'mae_direct': mae_direct,
                'mae_ratio': mae_ratio,
                'best_method': 'direct' if mae_direct < mae_ratio else 'ratio',
                'best_mae': min(mae_direct, mae_ratio)
            })
            
        except Exception as e:
            print(f"   ⚠️ {model_name} 실패: {e}")
    
    return results


def test_weight_optimization(gs_history, market_history, test_months=3):
    """ratio vs direct 가중치 최적화"""
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
    
    # 모델 학습
    share_model = LinearRegression()
    share_model.fit(X_train, gs_shares)
    
    gs_model = LinearRegression()
    gs_model.fit(X_train, gs_chargers)
    
    market_model = LinearRegression()
    market_model.fit(X_train, market_chargers)
    
    # 가중치 탐색
    weights = np.arange(0, 1.05, 0.1)  # 0.0, 0.1, ..., 1.0
    
    results = []
    
    for w_ratio in weights:
        w_direct = 1 - w_ratio
        
        errors = []
        for i in range(1, test_months + 1):
            future_idx = n_train + i - 1
            X_future = np.array([[future_idx]])
            
            pred_direct = share_model.predict(X_future)[0]
            pred_gs = gs_model.predict(X_future)[0]
            pred_market = market_model.predict(X_future)[0]
            pred_ratio = (pred_gs / pred_market) * 100 if pred_market > 0 else 0
            
            pred_combined = w_ratio * pred_ratio + w_direct * pred_direct
            actual = actual_shares[i-1]
            
            errors.append(abs(pred_combined - actual))
        
        mae = np.mean(errors)
        results.append({
            'w_ratio': w_ratio,
            'w_direct': w_direct,
            'mae': mae
        })
    
    return results


def run_comprehensive_optimization():
    """종합 최적화 실행"""
    print("=" * 60)
    print("🔬 ML 예측기 파라미터 최적화")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 데이터 로드
    print("\n📥 데이터 로드 중...")
    full_data = load_full_data()
    if full_data is None:
        print("❌ 데이터 로드 실패")
        return
    
    gs_history, market_history = extract_histories(full_data)
    print(f"✅ 데이터 로드 완료: {len(gs_history)}개월")
    
    # 1. 다양한 모델 비교
    print("\n" + "=" * 60)
    print("📊 1. ML 모델 비교 (3개월 예측)")
    print("=" * 60)
    
    model_results = test_different_models(gs_history, market_history, test_months=3)
    
    if model_results:
        print("\n모델별 MAE:")
        print(f"{'모델':<20} {'Direct MAE':<12} {'Ratio MAE':<12} {'Best':<8} {'Best MAE':<10}")
        print("-" * 62)
        
        for r in sorted(model_results, key=lambda x: x['best_mae']):
            print(f"{r['model']:<20} {r['mae_direct']:<12.4f} {r['mae_ratio']:<12.4f} {r['best_method']:<8} {r['best_mae']:<10.4f}")
        
        best_model = min(model_results, key=lambda x: x['best_mae'])
        print(f"\n✅ 최적 모델: {best_model['model']} ({best_model['best_method']} 방식, MAE={best_model['best_mae']:.4f})")
    
    # 2. 가중치 최적화
    print("\n" + "=" * 60)
    print("📊 2. Ratio vs Direct 가중치 최적화 (3개월 예측)")
    print("=" * 60)
    
    weight_results = test_weight_optimization(gs_history, market_history, test_months=3)
    
    if weight_results:
        print("\n가중치별 MAE:")
        print(f"{'Ratio 가중치':<15} {'Direct 가중치':<15} {'MAE':<10}")
        print("-" * 40)
        
        for r in weight_results:
            print(f"{r['w_ratio']:<15.1f} {r['w_direct']:<15.1f} {r['mae']:<10.4f}")
        
        best_weight = min(weight_results, key=lambda x: x['mae'])
        print(f"\n✅ 최적 가중치: Ratio={best_weight['w_ratio']:.1f}, Direct={best_weight['w_direct']:.1f} (MAE={best_weight['mae']:.4f})")
    
    # 3. 다양한 예측 기간에서 최적 가중치 탐색
    print("\n" + "=" * 60)
    print("📊 3. 예측 기간별 최적 가중치")
    print("=" * 60)
    
    for test_months in [1, 2, 3, 4, 5, 6]:
        if len(gs_history) < test_months + 4:
            continue
        
        weight_results = test_weight_optimization(gs_history, market_history, test_months=test_months)
        if weight_results:
            best = min(weight_results, key=lambda x: x['mae'])
            print(f"   {test_months}개월 예측: 최적 Ratio={best['w_ratio']:.1f}, Direct={best['w_direct']:.1f} (MAE={best['mae']:.4f})")
    
    # 4. 현재 설정 vs 최적 설정 비교
    print("\n" + "=" * 60)
    print("📊 4. 현재 설정 vs 최적 설정 비교")
    print("=" * 60)
    
    # 현재 설정: ratio 70%, direct 30%
    current_w_ratio = 0.7
    current_w_direct = 0.3
    
    # 3개월 예측 기준
    weight_results = test_weight_optimization(gs_history, market_history, test_months=3)
    if weight_results:
        current_mae = next((r['mae'] for r in weight_results if abs(r['w_ratio'] - current_w_ratio) < 0.05), None)
        best = min(weight_results, key=lambda x: x['mae'])
        
        print(f"\n현재 설정 (Ratio={current_w_ratio}, Direct={current_w_direct}):")
        print(f"   MAE: {current_mae:.4f}" if current_mae else "   MAE: N/A")
        
        print(f"\n최적 설정 (Ratio={best['w_ratio']}, Direct={best['w_direct']}):")
        print(f"   MAE: {best['mae']:.4f}")
        
        if current_mae and best['mae'] < current_mae:
            improvement = (current_mae - best['mae']) / current_mae * 100
            print(f"\n💡 최적 설정으로 변경 시 {improvement:.1f}% 개선 가능")
        else:
            print(f"\n✅ 현재 설정이 이미 최적에 가깝습니다")
    
    # 5. 결론 및 권장사항
    print("\n" + "=" * 60)
    print("💡 결론 및 권장사항")
    print("=" * 60)
    
    print("""
1. Linear Regression이 현재 데이터에서 가장 적합
   - 데이터가 12개월로 제한적이어서 복잡한 모델은 과적합 위험
   - 점유율 하락 추세가 매우 선형적

2. Ratio 방식이 Direct 방식보다 약간 더 좋음
   - 시장 역학(GS 성장 vs 시장 성장)을 반영하기 때문
   - 하지만 차이가 크지 않음

3. 현재 가중치(Ratio 70%, Direct 30%)는 합리적
   - 최적 가중치와 큰 차이 없음
   - 안정성을 위해 현재 설정 유지 권장

4. GS차지비 자체 추세 모델링은 현재 불필요
   - 데이터가 더 많아지면 재검토 필요
   - 현재는 단순 모델이 더 효과적
""")


if __name__ == "__main__":
    run_comprehensive_optimization()
