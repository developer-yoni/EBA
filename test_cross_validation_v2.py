"""
시뮬레이터 1, 2 크로스체크 백테스트

목적:
1. 시뮬레이터 1 (추가 충전기 → 점유율 예측) 정확도 검증
2. 시뮬레이터 2 (목표 점유율 → 필요 충전기 역계산) 정확도 검증
3. 두 시뮬레이터 간 일관성 검증

로그에서 확인된 문제:
- 시뮬레이터 1: 2,500대 추가 → 2개월 후 점유율 예측
- 시뮬레이터 2: 목표 16.5% → 4,109대 필요 (baseline 15.71%)
- 크로스체크: 2,500대로 얼마나 점유율이 오르는지 vs 16.5%에 4,109대가 맞는지
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression, Ridge
from data_loader import ChargingDataLoader


def load_data():
    """데이터 로드"""
    loader = ChargingDataLoader()
    full_data = loader.load_multiple()  # 모든 월 로드
    return full_data


def extract_gs_history(full_data, up_to_month=None):
    """GS차지비 히스토리 추출"""
    gs_data = full_data[full_data['CPO명'] == 'GS차지비'].copy()
    if up_to_month:
        gs_data = gs_data[gs_data['snapshot_month'] <= up_to_month]
    gs_data = gs_data.sort_values('snapshot_month')
    
    history = []
    for _, row in gs_data.iterrows():
        market_share = row.get('시장점유율', 0)
        if pd.notna(market_share) and market_share < 1:
            market_share = market_share * 100
        
        history.append({
            'month': row.get('snapshot_month'),
            'total_chargers': int(row.get('총충전기', 0)) if pd.notna(row.get('총충전기')) else 0,
            'market_share': round(float(market_share), 4) if pd.notna(market_share) else 0,
        })
    return history


def extract_market_history(full_data, up_to_month=None):
    """시장 히스토리 추출"""
    if up_to_month:
        filtered = full_data[full_data['snapshot_month'] <= up_to_month]
    else:
        filtered = full_data
    
    all_months = sorted(filtered['snapshot_month'].unique().tolist())
    
    market_history = []
    for month in all_months:
        month_data = filtered[filtered['snapshot_month'] == month]
        if len(month_data) > 0:
            total_chargers = month_data['총충전기'].sum()
            market_history.append({
                'month': month,
                'total_chargers': int(total_chargers),
            })
    return market_history


def ml_predict_share(gs_history, market_history, months_ahead, extra_chargers=0, use_ridge=True, alpha=10.0):
    """
    ML 기반 점유율 예측 (ratio 방식)
    
    Args:
        gs_history: GS차지비 히스토리
        market_history: 시장 히스토리
        months_ahead: 예측 개월 수
        extra_chargers: 추가 충전기 (시나리오용)
        use_ridge: Ridge 회귀 사용 여부
        alpha: Ridge 정규화 강도
    
    Returns:
        예측 결과 딕셔너리
    """
    n = len(gs_history)
    X = np.arange(n).reshape(-1, 1)
    
    gs_chargers = np.array([h['total_chargers'] for h in gs_history])
    gs_shares = np.array([h['market_share'] for h in gs_history])
    market_chargers = np.array([m['total_chargers'] for m in market_history[:n]])
    
    # 모델 선택
    if use_ridge:
        model_gs = Ridge(alpha=alpha)
        model_market = Ridge(alpha=alpha)
        model_share = Ridge(alpha=alpha)
    else:
        model_gs = LinearRegression()
        model_market = LinearRegression()
        model_share = LinearRegression()
    
    # 학습
    model_gs.fit(X, gs_chargers)
    model_market.fit(X, market_chargers)
    model_share.fit(X, gs_shares)
    
    # 예측
    predictions = []
    monthly_extra = extra_chargers / months_ahead if months_ahead > 0 else 0
    cumulative_extra = 0
    
    for i in range(1, months_ahead + 1):
        future_idx = n + i - 1
        X_future = np.array([[future_idx]])
        
        # GS 충전기 예측
        pred_gs = model_gs.predict(X_future)[0]
        # 시장 전체 예측
        pred_market = model_market.predict(X_future)[0]
        # 점유율 직접 예측
        pred_share_direct = model_share.predict(X_future)[0]
        
        # 추가 충전기 반영
        cumulative_extra += monthly_extra
        pred_gs_with_extra = pred_gs + cumulative_extra
        pred_market_with_extra = pred_market + cumulative_extra  # GS 추가 → 시장도 증가
        
        # Ratio 방식 점유율
        pred_share_ratio = (pred_gs_with_extra / pred_market_with_extra) * 100 if pred_market_with_extra > 0 else 0
        
        predictions.append({
            'months_ahead': i,
            'pred_gs_chargers': int(pred_gs_with_extra),
            'pred_market_chargers': int(pred_market_with_extra),
            'pred_share_ratio': round(pred_share_ratio, 4),
            'pred_share_direct': round(pred_share_direct, 4),
            'added_chargers': int(cumulative_extra)
        })
    
    return {
        'predictions': predictions,
        'model_stats': {
            'gs_slope': model_gs.coef_[0],
            'market_slope': model_market.coef_[0],
            'share_slope': model_share.coef_[0],
            'gs_r2': model_gs.score(X, gs_chargers),
            'market_r2': model_market.score(X, market_chargers),
            'share_r2': model_share.score(X, gs_shares),
        },
        'current': {
            'gs_chargers': gs_chargers[-1],
            'market_chargers': market_chargers[-1],
            'share': gs_shares[-1]
        }
    }


def calculate_required_chargers(gs_history, market_history, months_ahead, target_share, use_ridge=True, alpha=10.0):
    """
    목표 점유율 달성에 필요한 충전기 수 역계산
    
    핵심 공식:
    target_share = (baseline_gs + extra) / (baseline_market + extra) * 100
    정리: extra = (target_share * baseline_market - 100 * baseline_gs) / (100 - target_share)
    """
    n = len(gs_history)
    X = np.arange(n).reshape(-1, 1)
    
    gs_chargers = np.array([h['total_chargers'] for h in gs_history])
    market_chargers = np.array([m['total_chargers'] for m in market_history[:n]])
    
    # 모델 선택
    if use_ridge:
        model_gs = Ridge(alpha=alpha)
        model_market = Ridge(alpha=alpha)
    else:
        model_gs = LinearRegression()
        model_market = LinearRegression()
    
    model_gs.fit(X, gs_chargers)
    model_market.fit(X, market_chargers)
    
    # Baseline 예측 (months_ahead 후)
    future_idx = n + months_ahead - 1
    X_future = np.array([[future_idx]])
    
    baseline_gs = model_gs.predict(X_future)[0]
    baseline_market = model_market.predict(X_future)[0]
    baseline_share = (baseline_gs / baseline_market) * 100 if baseline_market > 0 else 0
    
    # 필요 충전기 계산
    if target_share >= 100:
        required_extra = 0
    else:
        numerator = (target_share * baseline_market) - (100 * baseline_gs)
        denominator = 100 - target_share
        required_extra = numerator / denominator if denominator != 0 else 0
    
    return {
        'baseline_gs': int(baseline_gs),
        'baseline_market': int(baseline_market),
        'baseline_share': round(baseline_share, 4),
        'target_share': target_share,
        'required_extra': int(max(0, required_extra)),
        'monthly_extra': int(max(0, required_extra) / months_ahead) if months_ahead > 0 else 0
    }


def cross_validate_simulators(full_data, base_month, sim_period, extra_chargers, target_share):
    """
    시뮬레이터 1, 2 크로스 검증
    
    검증 방법:
    1. 시뮬레이터 1: extra_chargers 추가 시 예측 점유율 계산
    2. 시뮬레이터 2: target_share 달성에 필요한 충전기 계산
    3. 크로스체크: 시뮬레이터 2의 결과로 시뮬레이터 1 실행 → target_share와 일치해야 함
    """
    print(f"\n{'='*70}")
    print(f"🔍 크로스 검증: base_month={base_month}, sim_period={sim_period}개월")
    print(f"{'='*70}")
    
    # 데이터 추출
    gs_history = extract_gs_history(full_data, up_to_month=base_month)
    market_history = extract_market_history(full_data, up_to_month=base_month)
    
    if len(gs_history) < 3:
        print("❌ 데이터 부족")
        return None
    
    current_share = gs_history[-1]['market_share']
    current_gs = gs_history[-1]['total_chargers']
    current_market = market_history[-1]['total_chargers']
    
    print(f"\n📊 현재 상태 ({base_month}):")
    print(f"   - GS차지비 충전기: {current_gs:,}대")
    print(f"   - 시장 전체 충전기: {current_market:,}대")
    print(f"   - 현재 점유율: {current_share:.2f}%")
    
    # ========== 시뮬레이터 1: 추가 충전기 → 점유율 예측 ==========
    print(f"\n🎯 시뮬레이터 1: {extra_chargers:,}대 추가 시 점유율 예측")
    
    sim1_result = ml_predict_share(gs_history, market_history, sim_period, extra_chargers)
    final_pred = sim1_result['predictions'][-1]
    
    print(f"   - Baseline 점유율 (추가 없음): {ml_predict_share(gs_history, market_history, sim_period, 0)['predictions'][-1]['pred_share_ratio']:.2f}%")
    print(f"   - 시나리오 점유율 ({extra_chargers:,}대 추가): {final_pred['pred_share_ratio']:.2f}%")
    print(f"   - 예측 GS 충전기: {final_pred['pred_gs_chargers']:,}대")
    print(f"   - 예측 시장 전체: {final_pred['pred_market_chargers']:,}대")
    
    # ========== 시뮬레이터 2: 목표 점유율 → 필요 충전기 역계산 ==========
    print(f"\n🎯 시뮬레이터 2: 목표 {target_share:.2f}% 달성에 필요한 충전기")
    
    sim2_result = calculate_required_chargers(gs_history, market_history, sim_period, target_share)
    
    print(f"   - Baseline 점유율: {sim2_result['baseline_share']:.2f}%")
    print(f"   - 목표 점유율: {target_share:.2f}%")
    print(f"   - 필요 추가 충전기: {sim2_result['required_extra']:,}대")
    print(f"   - 월평균 설치: {sim2_result['monthly_extra']:,}대/월")
    
    # ========== 크로스체크: 시뮬레이터 2 결과로 시뮬레이터 1 실행 ==========
    print(f"\n🔄 크로스체크: 시뮬레이터 2의 {sim2_result['required_extra']:,}대로 시뮬레이터 1 실행")
    
    cross_result = ml_predict_share(gs_history, market_history, sim_period, sim2_result['required_extra'])
    cross_final = cross_result['predictions'][-1]
    
    print(f"   - 예측 점유율: {cross_final['pred_share_ratio']:.2f}%")
    print(f"   - 목표 점유율: {target_share:.2f}%")
    
    error = abs(cross_final['pred_share_ratio'] - target_share)
    print(f"   - 오차: {error:.4f}%p")
    
    if error < 0.01:
        print(f"   ✅ 크로스체크 통과 (오차 < 0.01%p)")
    else:
        print(f"   ⚠️ 크로스체크 실패 (오차 >= 0.01%p)")
    
    return {
        'base_month': base_month,
        'sim_period': sim_period,
        'current_share': current_share,
        'sim1': {
            'extra_chargers': extra_chargers,
            'predicted_share': final_pred['pred_share_ratio'],
            'baseline_share': ml_predict_share(gs_history, market_history, sim_period, 0)['predictions'][-1]['pred_share_ratio']
        },
        'sim2': {
            'target_share': target_share,
            'required_extra': sim2_result['required_extra'],
            'baseline_share': sim2_result['baseline_share']
        },
        'cross_check': {
            'predicted_share': cross_final['pred_share_ratio'],
            'target_share': target_share,
            'error': error,
            'passed': error < 0.01
        }
    }


def backtest_with_actual_data(full_data, base_month, sim_period):
    """
    실제 데이터와 비교하는 백테스트
    
    기준월 이후의 실제 데이터가 있는 경우, 예측값과 비교
    """
    print(f"\n{'='*70}")
    print(f"📊 백테스트: base_month={base_month}, sim_period={sim_period}개월")
    print(f"{'='*70}")
    
    all_months = sorted(full_data['snapshot_month'].unique().tolist())
    
    # 기준월 인덱스
    if base_month not in all_months:
        print(f"❌ 기준월 {base_month}이 데이터에 없음")
        return None
    
    base_idx = all_months.index(base_month)
    
    # 예측 대상 월 확인
    target_months = []
    for i in range(1, sim_period + 1):
        if base_idx + i < len(all_months):
            target_months.append(all_months[base_idx + i])
    
    if not target_months:
        print(f"❌ 예측 대상 월의 실제 데이터 없음")
        return None
    
    print(f"   - 예측 대상 월: {target_months}")
    
    # 기준월까지의 데이터로 예측
    gs_history = extract_gs_history(full_data, up_to_month=base_month)
    market_history = extract_market_history(full_data, up_to_month=base_month)
    
    # 예측 (추가 충전기 없음 = baseline)
    predictions = ml_predict_share(gs_history, market_history, len(target_months), 0)
    
    # 실제값 추출
    actuals = []
    for month in target_months:
        gs_row = full_data[(full_data['snapshot_month'] == month) & (full_data['CPO명'] == 'GS차지비')]
        if len(gs_row) > 0:
            share = gs_row.iloc[0].get('시장점유율', 0)
            if pd.notna(share) and share < 1:
                share = share * 100
            actuals.append({
                'month': month,
                'actual_share': round(float(share), 4) if pd.notna(share) else None
            })
    
    # 비교
    results = []
    for i, (pred, actual) in enumerate(zip(predictions['predictions'], actuals)):
        if actual['actual_share'] is not None:
            error = pred['pred_share_ratio'] - actual['actual_share']
            abs_error = abs(error)
            pct_error = abs_error / actual['actual_share'] * 100 if actual['actual_share'] > 0 else 0
            
            results.append({
                'month': actual['month'],
                'months_ahead': i + 1,
                'predicted': pred['pred_share_ratio'],
                'actual': actual['actual_share'],
                'error': round(error, 4),
                'abs_error': round(abs_error, 4),
                'pct_error': round(pct_error, 2)
            })
            
            print(f"   {actual['month']}: 예측 {pred['pred_share_ratio']:.2f}% vs 실제 {actual['actual_share']:.2f}% (오차: {error:+.4f}%p, MAPE: {pct_error:.2f}%)")
    
    if results:
        avg_mae = np.mean([r['abs_error'] for r in results])
        avg_mape = np.mean([r['pct_error'] for r in results])
        print(f"\n   📈 평균 MAE: {avg_mae:.4f}%p, 평균 MAPE: {avg_mape:.2f}%")
    
    return results


def compare_ridge_vs_linear(full_data, base_months, sim_periods):
    """
    Ridge vs LinearRegression 비교
    """
    print(f"\n{'='*70}")
    print(f"📊 Ridge vs LinearRegression 비교")
    print(f"{'='*70}")
    
    ridge_errors = []
    linear_errors = []
    
    for base_month in base_months:
        for sim_period in sim_periods:
            gs_history = extract_gs_history(full_data, up_to_month=base_month)
            market_history = extract_market_history(full_data, up_to_month=base_month)
            
            if len(gs_history) < 3:
                continue
            
            all_months = sorted(full_data['snapshot_month'].unique().tolist())
            base_idx = all_months.index(base_month) if base_month in all_months else -1
            
            if base_idx < 0 or base_idx + sim_period >= len(all_months):
                continue
            
            # 실제값
            target_month = all_months[base_idx + sim_period]
            gs_row = full_data[(full_data['snapshot_month'] == target_month) & (full_data['CPO명'] == 'GS차지비')]
            if len(gs_row) == 0:
                continue
            
            actual_share = gs_row.iloc[0].get('시장점유율', 0)
            if pd.notna(actual_share) and actual_share < 1:
                actual_share = actual_share * 100
            
            # Ridge 예측
            ridge_pred = ml_predict_share(gs_history, market_history, sim_period, 0, use_ridge=True)
            ridge_share = ridge_pred['predictions'][-1]['pred_share_ratio']
            ridge_error = abs(ridge_share - actual_share)
            ridge_errors.append(ridge_error)
            
            # Linear 예측
            linear_pred = ml_predict_share(gs_history, market_history, sim_period, 0, use_ridge=False)
            linear_share = linear_pred['predictions'][-1]['pred_share_ratio']
            linear_error = abs(linear_share - actual_share)
            linear_errors.append(linear_error)
    
    if ridge_errors and linear_errors:
        print(f"\n   Ridge 평균 MAE: {np.mean(ridge_errors):.4f}%p")
        print(f"   Linear 평균 MAE: {np.mean(linear_errors):.4f}%p")
        
        if np.mean(ridge_errors) < np.mean(linear_errors):
            improvement = (np.mean(linear_errors) - np.mean(ridge_errors)) / np.mean(linear_errors) * 100
            print(f"   ✅ Ridge가 {improvement:.1f}% 더 정확")
            return 'ridge'
        else:
            improvement = (np.mean(ridge_errors) - np.mean(linear_errors)) / np.mean(ridge_errors) * 100
            print(f"   ✅ Linear가 {improvement:.1f}% 더 정확")
            return 'linear'
    
    return None


def main():
    print("=" * 70)
    print("🔬 시뮬레이터 크로스 검증 및 백테스트")
    print("=" * 70)
    
    # 데이터 로드
    print("\n📥 데이터 로드 중...")
    full_data = load_data()
    
    all_months = sorted(full_data['snapshot_month'].unique().tolist())
    print(f"   - 전체 월: {len(all_months)}개 ({all_months[0]} ~ {all_months[-1]})")
    
    # 1. Ridge vs Linear 비교
    print("\n" + "=" * 70)
    print("1️⃣ Ridge vs LinearRegression 비교")
    print("=" * 70)
    
    # 백테스트 가능한 기준월 (최소 3개월 학습, 최소 1개월 평가)
    valid_base_months = all_months[2:-1]  # 처음 2개월 제외 (학습용), 마지막 1개월 제외 (평가용)
    
    better_model = compare_ridge_vs_linear(full_data, valid_base_months, [1, 2, 3])
    
    # 2. 백테스트 (다양한 기준월)
    print("\n" + "=" * 70)
    print("2️⃣ 백테스트 (실제 데이터 비교)")
    print("=" * 70)
    
    backtest_results = []
    for base_month in valid_base_months[-4:]:  # 최근 4개 기준월
        for sim_period in [1, 2, 3]:
            result = backtest_with_actual_data(full_data, base_month, sim_period)
            if result:
                backtest_results.extend(result)
    
    if backtest_results:
        print(f"\n📊 전체 백테스트 요약:")
        print(f"   - 총 테스트: {len(backtest_results)}개")
        print(f"   - 평균 MAE: {np.mean([r['abs_error'] for r in backtest_results]):.4f}%p")
        print(f"   - 평균 MAPE: {np.mean([r['pct_error'] for r in backtest_results]):.2f}%")
    
    # 3. 크로스 검증 (로그에서 확인된 시나리오)
    print("\n" + "=" * 70)
    print("3️⃣ 크로스 검증 (로그 시나리오 재현)")
    print("=" * 70)
    
    # 로그에서 확인된 시나리오: base_month=2025-11, sim_period=2, extra=2500, target=16.5%
    latest_month = all_months[-1]
    
    cross_result = cross_validate_simulators(
        full_data,
        base_month=latest_month,
        sim_period=2,
        extra_chargers=2500,
        target_share=16.5
    )
    
    # 4. 추가 크로스 검증 (다양한 시나리오)
    print("\n" + "=" * 70)
    print("4️⃣ 추가 크로스 검증")
    print("=" * 70)
    
    for extra in [1000, 3000, 5000]:
        cross_validate_simulators(
            full_data,
            base_month=latest_month,
            sim_period=2,
            extra_chargers=extra,
            target_share=17.0
        )
    
    # 5. 결론
    print("\n" + "=" * 70)
    print("📋 결론")
    print("=" * 70)
    
    if backtest_results:
        avg_mape = np.mean([r['pct_error'] for r in backtest_results])
        if avg_mape < 2.0:
            print(f"   ✅ 현재 ML 모델 (LinearRegression + Ratio 방식) 정확도 양호 (MAPE {avg_mape:.2f}%)")
            print(f"   → 현상 유지 권장")
        else:
            print(f"   ⚠️ 현재 ML 모델 정확도 개선 필요 (MAPE {avg_mape:.2f}%)")
            print(f"   → 모델 재검토 필요")
    
    if better_model:
        print(f"   → 권장 모델: {better_model.upper()}")


if __name__ == "__main__":
    main()
