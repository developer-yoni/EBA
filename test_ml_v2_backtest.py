"""
ML 예측기 V1 vs V2 백테스트 비교

목적:
1. 기존 방식(V1: 전체 시장 추세만)과 개선된 방식(V2: GS 자체 추세 + 상대 성장률) 비교
2. 어떤 방식이 더 정확한지 검증
3. 최적의 파라미터 탐색
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 데이터 로더
from data_loader import ChargingDataLoader

def load_full_data():
    """전체 RAG 데이터 로드"""
    print("=" * 60)
    print("📥 전체 RAG 데이터 로드 중...")
    print("=" * 60)
    
    loader = ChargingDataLoader()
    full_data = loader.load_multiple()
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        return None
    
    all_months = sorted(full_data['snapshot_month'].unique().tolist())
    print(f"✅ 데이터 로드 완료: {len(full_data)} 행, {len(all_months)} 개월")
    print(f"   기간: {all_months[0]} ~ {all_months[-1]}")
    
    return full_data


def extract_histories(full_data: pd.DataFrame):
    """GS차지비 및 시장 히스토리 추출"""
    # GS차지비 데이터 추출
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
            'total_change': int(row.get('총증감', 0)) if pd.notna(row.get('총증감')) else 0
        })
    
    # 시장 히스토리 추출
    all_months = sorted(full_data['snapshot_month'].unique().tolist())
    market_history = []
    for month in all_months:
        month_data = full_data[full_data['snapshot_month'] == month]
        if len(month_data) > 0:
            total_chargers = month_data['총충전기'].sum()
            total_cpos = len(month_data[month_data['총충전기'] > 0])
            market_history.append({
                'month': month,
                'total_chargers': int(total_chargers),
                'total_cpos': int(total_cpos)
            })
    
    return gs_history, market_history


def run_v1_backtest(gs_history, market_history, test_months):
    """V1 (기존) 방식 백테스트"""
    from ml_predictor import ImprovedMLPredictor
    
    n = len(gs_history)
    if n < test_months + 3:
        return None
    
    # 학습/테스트 분리
    train_gs = gs_history[:-test_months]
    train_market = market_history[:-test_months]
    test_gs = gs_history[-test_months:]
    
    # 학습
    predictor = ImprovedMLPredictor()
    fit_result = predictor.fit(train_gs, train_market)
    
    if 'error' in fit_result:
        return None
    
    # 예측 (ratio 방식)
    predictions = predictor.predict(test_months, method='ratio')
    
    # 실제값
    actual_shares = [h['market_share'] for h in test_gs]
    
    # 오차 계산
    errors = []
    for pred, actual in zip(predictions, actual_shares):
        abs_error = abs(pred['predicted_share'] - actual)
        pct_error = abs_error / actual * 100 if actual > 0 else 0
        errors.append({
            'abs_error': abs_error,
            'pct_error': pct_error
        })
    
    mae = np.mean([e['abs_error'] for e in errors])
    mape = np.mean([e['pct_error'] for e in errors])
    
    return {
        'mae': mae,
        'mape': mape,
        'predictions': [p['predicted_share'] for p in predictions],
        'actuals': actual_shares
    }


def run_v2_backtest(gs_history, market_history, test_months):
    """V2 (개선) 방식 백테스트"""
    from ml_predictor_v2 import ImprovedMLPredictorV2
    
    predictor = ImprovedMLPredictorV2()
    result = predictor.backtest(gs_history, market_history, test_months)
    
    if 'error' in result:
        return None
    
    return {
        'mae': result['mae'],
        'mape': result['mape'],
        'predictions': [e['predicted'] for e in result['errors']],
        'actuals': [e['actual'] for e in result['errors']],
        'fit_result': result['fit_result']
    }


def run_comprehensive_backtest():
    """종합 백테스트 실행"""
    # 데이터 로드
    full_data = load_full_data()
    if full_data is None:
        return
    
    gs_history, market_history = extract_histories(full_data)
    
    print(f"\n📊 GS차지비 히스토리: {len(gs_history)}개월")
    print(f"📊 시장 히스토리: {len(market_history)}개월")
    
    # 히스토리 출력
    print("\n📈 GS차지비 점유율 추이:")
    for h in gs_history:
        print(f"   {h['month']}: {h['market_share']:.2f}% ({h['total_chargers']:,}대)")
    
    print("\n" + "=" * 60)
    print("🔬 V1 vs V2 백테스트 비교")
    print("=" * 60)
    
    results = []
    
    # 다양한 테스트 기간으로 비교
    for test_months in [1, 2, 3, 4, 5, 6]:
        if len(gs_history) < test_months + 4:
            print(f"\n⚠️ {test_months}개월 테스트: 데이터 부족")
            continue
        
        print(f"\n📊 {test_months}개월 예측 테스트:")
        
        # V1 테스트
        v1_result = run_v1_backtest(gs_history, market_history, test_months)
        
        # V2 테스트
        v2_result = run_v2_backtest(gs_history, market_history, test_months)
        
        if v1_result and v2_result:
            improvement = (v1_result['mae'] - v2_result['mae']) / v1_result['mae'] * 100 if v1_result['mae'] > 0 else 0
            better = 'V2' if v2_result['mae'] < v1_result['mae'] else 'V1'
            
            print(f"   V1 (기존): MAE={v1_result['mae']:.4f}, MAPE={v1_result['mape']:.2f}%")
            print(f"   V2 (개선): MAE={v2_result['mae']:.4f}, MAPE={v2_result['mape']:.2f}%")
            print(f"   → {better} 승리 (개선율: {improvement:+.1f}%)")
            
            # 상세 비교
            print(f"\n   예측 vs 실제:")
            for i, (v1_pred, v2_pred, actual) in enumerate(zip(
                v1_result['predictions'], 
                v2_result['predictions'], 
                v1_result['actuals']
            )):
                v1_err = abs(v1_pred - actual)
                v2_err = abs(v2_pred - actual)
                winner = "V2✓" if v2_err < v1_err else "V1✓" if v1_err < v2_err else "동점"
                print(f"      {i+1}개월: 실제={actual:.2f}%, V1={v1_pred:.2f}%(오차:{v1_err:.3f}), V2={v2_pred:.2f}%(오차:{v2_err:.3f}) [{winner}]")
            
            results.append({
                'test_months': test_months,
                'v1_mae': v1_result['mae'],
                'v1_mape': v1_result['mape'],
                'v2_mae': v2_result['mae'],
                'v2_mape': v2_result['mape'],
                'improvement': improvement,
                'better': better
            })
    
    # 요약
    if results:
        print("\n" + "=" * 60)
        print("📊 종합 결과")
        print("=" * 60)
        
        v2_wins = sum(1 for r in results if r['better'] == 'V2')
        v1_wins = len(results) - v2_wins
        
        avg_v1_mae = np.mean([r['v1_mae'] for r in results])
        avg_v2_mae = np.mean([r['v2_mae'] for r in results])
        avg_v1_mape = np.mean([r['v1_mape'] for r in results])
        avg_v2_mape = np.mean([r['v2_mape'] for r in results])
        avg_improvement = np.mean([r['improvement'] for r in results])
        
        print(f"\n승패: V1 {v1_wins}승 vs V2 {v2_wins}승")
        print(f"\n평균 MAE:")
        print(f"   V1: {avg_v1_mae:.4f}")
        print(f"   V2: {avg_v2_mae:.4f}")
        print(f"   개선율: {avg_improvement:+.1f}%")
        
        print(f"\n평균 MAPE:")
        print(f"   V1: {avg_v1_mape:.2f}%")
        print(f"   V2: {avg_v2_mape:.2f}%")
        
        # 권장사항
        print("\n" + "=" * 60)
        print("💡 권장사항")
        print("=" * 60)
        
        if v2_wins > v1_wins:
            print(f"\n✅ V2 (개선된 방식) 사용 권장")
            print(f"   - GS차지비 자체 추세 + 상대 성장률 모델링이 효과적")
            print(f"   - 평균 {avg_improvement:.1f}% 오차 감소")
        elif v1_wins > v2_wins:
            print(f"\n✅ V1 (기존 방식) 유지 권장")
            print(f"   - 현재 데이터에서는 기존 방식이 더 안정적")
        else:
            print(f"\n⚠️ 두 방식 성능 유사")
            print(f"   - 상황에 따라 선택 가능")
        
        return results
    
    return None


def analyze_relative_growth():
    """상대 성장률 분석"""
    full_data = load_full_data()
    if full_data is None:
        return
    
    gs_history, market_history = extract_histories(full_data)
    
    print("\n" + "=" * 60)
    print("📊 GS차지비 vs 시장 상대 성장률 분석")
    print("=" * 60)
    
    print("\n월별 상대 성장률:")
    print("(양수 = GS가 시장보다 빠르게 성장, 음수 = 시장이 더 빠름)")
    
    for i in range(1, len(gs_history)):
        gs_prev = gs_history[i-1]['total_chargers']
        gs_curr = gs_history[i]['total_chargers']
        market_prev = market_history[i-1]['total_chargers']
        market_curr = market_history[i]['total_chargers']
        
        gs_growth = (gs_curr / gs_prev - 1) * 100 if gs_prev > 0 else 0
        market_growth = (market_curr / market_prev - 1) * 100 if market_prev > 0 else 0
        relative_growth = gs_growth - market_growth
        
        share_prev = gs_history[i-1]['market_share']
        share_curr = gs_history[i]['market_share']
        share_change = share_curr - share_prev
        
        print(f"   {gs_history[i]['month']}: GS {gs_growth:+.2f}% vs 시장 {market_growth:+.2f}% → 상대 {relative_growth:+.2f}% (점유율 {share_change:+.2f}%p)")


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 ML 예측기 V1 vs V2 백테스트")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 상대 성장률 분석
    analyze_relative_growth()
    
    # 종합 백테스트
    results = run_comprehensive_backtest()
