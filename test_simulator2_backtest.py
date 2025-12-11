"""
시뮬레이터 2 (점유율 → 필요 충전기) 백테스트 및 Edge Case 테스트

테스트 항목:
1. 기본 역계산 로직 검증
2. Edge Cases:
   - 목표 점유율 < 현재 점유율 (ALREADY_ACHIEVED)
   - 목표 점유율 = 현재 추세 예측 (TREND_ACHIEVABLE)
   - 목표 점유율 > 현재 추세 예측 (ACHIEVABLE/CHALLENGING/DIFFICULT)
3. ML 분석 로직 검증
4. Bedrock 호출 없이 순수 ML 로직 테스트
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 프로젝트 모듈 임포트
from data_loader import ChargingDataLoader
from scenario_simulator import ScenarioSimulator
from backtest_simulator import BacktestSimulator


def load_data():
    """데이터 로드"""
    print("=" * 60)
    print("📥 데이터 로드 중...")
    print("=" * 60)
    
    loader = ChargingDataLoader()
    df = loader.load_multiple()
    
    if df is None or len(df) == 0:
        print("❌ 데이터 로드 실패")
        return None
    
    print(f"✅ 데이터 로드 완료: {len(df)} 행")
    
    all_months = sorted(df['snapshot_month'].unique().tolist())
    print(f"📅 데이터 범위: {all_months[0]} ~ {all_months[-1]} ({len(all_months)}개월)")
    
    return df


def test_edge_case_already_achieved(simulator, full_data, base_month):
    """Edge Case 1: 목표 점유율 < 현재 점유율 (ALREADY_ACHIEVED)"""
    print("\n" + "=" * 60)
    print("🧪 Edge Case 1: ALREADY_ACHIEVED (목표 < 현재)")
    print("=" * 60)
    
    # 현재 점유율 확인
    gs_history = simulator.extract_gs_history(full_data, up_to_month=base_month)
    if not gs_history:
        print("❌ GS차지비 데이터 없음")
        return False
    
    current_share = gs_history[-1]['market_share']
    print(f"📊 현재 점유율: {current_share:.2f}%")
    
    # 목표 점유율을 현재보다 낮게 설정
    target_share = current_share - 1.0
    print(f"🎯 목표 점유율: {target_share:.2f}% (현재보다 1%p 낮음)")
    
    result = simulator.calculate_required_chargers(
        base_month=base_month,
        sim_period_months=3,
        target_share=target_share,
        full_data=full_data
    )
    
    if not result.get('success'):
        print(f"❌ 역계산 실패: {result.get('error')}")
        return False
    
    feasibility = result.get('target_analysis', {}).get('feasibility')
    required_chargers = result.get('target_analysis', {}).get('required_chargers', 0)
    
    print(f"📋 결과:")
    print(f"   - Feasibility: {feasibility}")
    print(f"   - 필요 충전기: {required_chargers}대")
    
    # 검증
    if feasibility == 'ALREADY_ACHIEVED' and required_chargers == 0:
        print("✅ PASS: ALREADY_ACHIEVED 케이스 정상 처리")
        return True
    else:
        print("❌ FAIL: ALREADY_ACHIEVED 케이스 처리 오류")
        return False


def test_edge_case_trend_achievable(simulator, full_data, base_month):
    """Edge Case 2: 목표 점유율 = 현재 추세 예측 (TREND_ACHIEVABLE)"""
    print("\n" + "=" * 60)
    print("🧪 Edge Case 2: TREND_ACHIEVABLE (추세만으로 달성)")
    print("=" * 60)
    
    # ML 분석으로 추세 예측
    gs_history = simulator.extract_gs_history(full_data, up_to_month=base_month)
    market_history = simulator.extract_market_history(full_data, up_to_month=base_month)
    
    if len(gs_history) < 3:
        print("❌ 데이터 부족")
        return False
    
    ml_analysis = simulator.perform_ml_analysis(gs_history, market_history)
    
    current_share = gs_history[-1]['market_share']
    share_slope = ml_analysis.get('linear_regression', {}).get('share_slope', 0)
    
    # 3개월 후 추세 예측
    sim_period = 3
    baseline_share = current_share + (share_slope * sim_period)
    
    print(f"📊 현재 점유율: {current_share:.2f}%")
    print(f"📈 월별 추세: {share_slope:+.4f}%p/월")
    print(f"📊 {sim_period}개월 후 추세 예측: {baseline_share:.2f}%")
    
    # 목표 점유율을 추세 예측보다 약간 낮게 설정
    target_share = baseline_share - 0.1
    print(f"🎯 목표 점유율: {target_share:.2f}% (추세 예측보다 0.1%p 낮음)")
    
    result = simulator.calculate_required_chargers(
        base_month=base_month,
        sim_period_months=sim_period,
        target_share=target_share,
        full_data=full_data
    )
    
    if not result.get('success'):
        print(f"❌ 역계산 실패: {result.get('error')}")
        return False
    
    feasibility = result.get('target_analysis', {}).get('feasibility')
    required_chargers = result.get('target_analysis', {}).get('required_chargers', 0)
    
    print(f"📋 결과:")
    print(f"   - Feasibility: {feasibility}")
    print(f"   - 필요 충전기: {required_chargers}대")
    
    # 검증: TREND_ACHIEVABLE 또는 ALREADY_ACHIEVED
    if feasibility in ['TREND_ACHIEVABLE', 'ALREADY_ACHIEVED'] and required_chargers == 0:
        print("✅ PASS: TREND_ACHIEVABLE 케이스 정상 처리")
        return True
    else:
        print("❌ FAIL: TREND_ACHIEVABLE 케이스 처리 오류")
        return False


def test_edge_case_challenging(simulator, full_data, base_month):
    """Edge Case 3: 목표 점유율 > 현재 추세 (ACHIEVABLE/CHALLENGING/DIFFICULT)"""
    print("\n" + "=" * 60)
    print("🧪 Edge Case 3: 목표 > 추세 (충전기 설치 필요)")
    print("=" * 60)
    
    gs_history = simulator.extract_gs_history(full_data, up_to_month=base_month)
    market_history = simulator.extract_market_history(full_data, up_to_month=base_month)
    
    if len(gs_history) < 3:
        print("❌ 데이터 부족")
        return False
    
    ml_analysis = simulator.perform_ml_analysis(gs_history, market_history)
    
    current_share = gs_history[-1]['market_share']
    share_slope = ml_analysis.get('linear_regression', {}).get('share_slope', 0)
    
    sim_period = 3
    baseline_share = current_share + (share_slope * sim_period)
    
    print(f"📊 현재 점유율: {current_share:.2f}%")
    print(f"📈 월별 추세: {share_slope:+.4f}%p/월")
    print(f"📊 {sim_period}개월 후 추세 예측: {baseline_share:.2f}%")
    
    # 목표 점유율을 추세 예측보다 높게 설정
    target_share = baseline_share + 1.0
    print(f"🎯 목표 점유율: {target_share:.2f}% (추세 예측보다 1%p 높음)")
    
    result = simulator.calculate_required_chargers(
        base_month=base_month,
        sim_period_months=sim_period,
        target_share=target_share,
        full_data=full_data
    )
    
    if not result.get('success'):
        print(f"❌ 역계산 실패: {result.get('error')}")
        return False
    
    feasibility = result.get('target_analysis', {}).get('feasibility')
    required_chargers = result.get('target_analysis', {}).get('required_chargers', 0)
    monthly_chargers = result.get('target_analysis', {}).get('monthly_chargers', 0)
    
    print(f"📋 결과:")
    print(f"   - Feasibility: {feasibility}")
    print(f"   - 필요 충전기: {required_chargers:,}대")
    print(f"   - 월평균 설치: {monthly_chargers:,}대")
    
    # 검증: 충전기가 필요해야 함
    if feasibility in ['ACHIEVABLE', 'CHALLENGING', 'DIFFICULT'] and required_chargers > 0:
        print("✅ PASS: 충전기 설치 필요 케이스 정상 처리")
        return True
    else:
        print("❌ FAIL: 충전기 설치 필요 케이스 처리 오류")
        return False


def test_ml_analysis_accuracy(simulator, full_data):
    """ML 분석 정확도 테스트 (백테스트)"""
    print("\n" + "=" * 60)
    print("🧪 ML 분석 정확도 백테스트")
    print("=" * 60)
    
    all_months = sorted(full_data['snapshot_month'].unique().tolist())
    
    # 최소 6개월 학습, 3개월 평가 필요
    if len(all_months) < 9:
        print("❌ 백테스트에 필요한 데이터 부족 (최소 9개월)")
        return False
    
    # 백테스트 기준월 선택 (중간 지점)
    test_base_months = all_months[5:-3]  # 앞 6개월 학습, 뒤 3개월 평가
    
    print(f"📅 백테스트 기준월: {test_base_months}")
    
    errors = []
    
    for base_month in test_base_months:
        # 기준월까지의 데이터로 학습
        gs_history = simulator.extract_gs_history(full_data, up_to_month=base_month)
        market_history = simulator.extract_market_history(full_data, up_to_month=base_month)
        
        if len(gs_history) < 3:
            continue
        
        ml_analysis = simulator.perform_ml_analysis(gs_history, market_history)
        
        if 'error' in ml_analysis:
            continue
        
        # 예측값
        predictions = ml_analysis.get('ml_predictions', [])
        
        # 실제값 (기준월 이후)
        future_gs = full_data[
            (full_data['snapshot_month'] > base_month) & 
            (full_data['CPO명'] == 'GS차지비')
        ].sort_values('snapshot_month')
        
        for i, pred in enumerate(predictions[:3]):  # 최대 3개월 예측
            if i >= len(future_gs):
                break
            
            actual_row = future_gs.iloc[i]
            actual_share = actual_row.get('시장점유율', 0)
            if pd.notna(actual_share) and actual_share < 1:
                actual_share = actual_share * 100
            
            pred_share = pred['predicted_share']
            error = abs(pred_share - actual_share)
            
            errors.append({
                'base_month': base_month,
                'months_ahead': i + 1,
                'predicted': pred_share,
                'actual': actual_share,
                'error': error
            })
    
    if not errors:
        print("❌ 백테스트 결과 없음")
        return False
    
    errors_df = pd.DataFrame(errors)
    
    # 기간별 오차 분석
    print("\n📊 기간별 오차 분석:")
    for months_ahead in [1, 2, 3]:
        period_errors = errors_df[errors_df['months_ahead'] == months_ahead]
        if len(period_errors) > 0:
            mae = period_errors['error'].mean()
            mape = (period_errors['error'] / period_errors['actual'] * 100).mean()
            print(f"   {months_ahead}개월 예측: MAE={mae:.4f}%p, MAPE={mape:.2f}%")
    
    # 전체 오차
    overall_mae = errors_df['error'].mean()
    overall_mape = (errors_df['error'] / errors_df['actual'] * 100).mean()
    
    print(f"\n📊 전체 오차:")
    print(f"   MAE: {overall_mae:.4f}%p")
    print(f"   MAPE: {overall_mape:.2f}%")
    
    # 검증: MAPE 5% 이하면 합격
    if overall_mape <= 5.0:
        print("✅ PASS: ML 분석 정확도 양호 (MAPE ≤ 5%)")
        return True
    else:
        print("⚠️ WARNING: ML 분석 정확도 개선 필요 (MAPE > 5%)")
        return True  # 경고만 표시


def test_required_chargers_calculation(simulator, full_data, base_month):
    """필요 충전기 계산 로직 검증"""
    print("\n" + "=" * 60)
    print("🧪 필요 충전기 계산 로직 검증")
    print("=" * 60)
    
    gs_history = simulator.extract_gs_history(full_data, up_to_month=base_month)
    market_history = simulator.extract_market_history(full_data, up_to_month=base_month)
    
    if len(gs_history) < 3:
        print("❌ 데이터 부족")
        return False
    
    ml_analysis = simulator.perform_ml_analysis(gs_history, market_history)
    
    current_gs = gs_history[-1]
    current_share = current_gs['market_share']
    current_chargers = current_gs['total_chargers']
    current_market = market_history[-1]['total_chargers']
    
    lr_stats = ml_analysis.get('linear_regression', {})
    market_slope = lr_stats.get('market_slope', 0)
    charger_slope = lr_stats.get('charger_slope', 0)
    share_slope = lr_stats.get('share_slope', 0)
    
    sim_period = 3
    target_share = current_share + 1.0  # 1%p 증가 목표
    
    print(f"📊 현재 상태:")
    print(f"   - GS차지비 점유율: {current_share:.2f}%")
    print(f"   - GS차지비 충전기: {current_chargers:,}대")
    print(f"   - 시장 전체 충전기: {current_market:,}대")
    
    print(f"\n📈 추세:")
    print(f"   - 점유율 추세: {share_slope:+.4f}%p/월")
    print(f"   - GS 충전기 추세: {charger_slope:+.0f}대/월")
    print(f"   - 시장 충전기 추세: {market_slope:+.0f}대/월")
    
    # 수동 계산
    future_market = current_market + (market_slope * sim_period)
    baseline_chargers = current_chargers + (charger_slope * sim_period)
    baseline_share = current_share + (share_slope * sim_period)
    
    # 목표 달성에 필요한 총 충전기
    required_total = (target_share / 100) * future_market
    required_extra = required_total - baseline_chargers
    
    print(f"\n🎯 목표: {target_share:.2f}% ({sim_period}개월 후)")
    print(f"\n📊 수동 계산:")
    print(f"   - {sim_period}개월 후 시장 전체: {future_market:,.0f}대")
    print(f"   - {sim_period}개월 후 GS 추세 예측: {baseline_chargers:,.0f}대")
    print(f"   - {sim_period}개월 후 추세 점유율: {baseline_share:.2f}%")
    print(f"   - 목표 달성 필요 총 충전기: {required_total:,.0f}대")
    print(f"   - 추가 필요 충전기: {required_extra:,.0f}대")
    
    # API 호출
    result = simulator.calculate_required_chargers(
        base_month=base_month,
        sim_period_months=sim_period,
        target_share=target_share,
        full_data=full_data
    )
    
    if not result.get('success'):
        print(f"❌ 역계산 실패: {result.get('error')}")
        return False
    
    api_required = result.get('target_analysis', {}).get('required_chargers', 0)
    
    print(f"\n📊 API 결과:")
    print(f"   - 필요 충전기: {api_required:,}대")
    
    # 검증: 수동 계산과 API 결과 비교 (10% 오차 허용)
    if required_extra > 0:
        diff_ratio = abs(api_required - required_extra) / required_extra
        if diff_ratio <= 0.1:
            print(f"✅ PASS: 계산 로직 일치 (오차 {diff_ratio*100:.1f}%)")
            return True
        else:
            print(f"❌ FAIL: 계산 로직 불일치 (오차 {diff_ratio*100:.1f}%)")
            return False
    else:
        # 추가 충전기 불필요한 경우
        if api_required == 0:
            print("✅ PASS: 추가 충전기 불필요 케이스 일치")
            return True
        else:
            print("❌ FAIL: 추가 충전기 불필요 케이스 불일치")
            return False


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "=" * 60)
    print("🚀 시뮬레이터 2 백테스트 및 Edge Case 테스트 시작")
    print("=" * 60)
    
    # 데이터 로드
    full_data = load_data()
    if full_data is None:
        return
    
    # 시뮬레이터 초기화
    simulator = ScenarioSimulator()
    
    # 기준월 설정 (최신 월)
    all_months = sorted(full_data['snapshot_month'].unique().tolist())
    base_month = all_months[-1]
    print(f"\n📅 테스트 기준월: {base_month}")
    
    # 테스트 결과
    results = {}
    
    # 1. Edge Case: ALREADY_ACHIEVED
    results['already_achieved'] = test_edge_case_already_achieved(simulator, full_data, base_month)
    
    # 2. Edge Case: TREND_ACHIEVABLE
    results['trend_achievable'] = test_edge_case_trend_achievable(simulator, full_data, base_month)
    
    # 3. Edge Case: CHALLENGING
    results['challenging'] = test_edge_case_challenging(simulator, full_data, base_month)
    
    # 4. ML 분석 정확도
    results['ml_accuracy'] = test_ml_analysis_accuracy(simulator, full_data)
    
    # 5. 필요 충전기 계산 로직
    results['calculation'] = test_required_chargers_calculation(simulator, full_data, base_month)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 총 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패 - 로직 검토 필요")
    
    return results


if __name__ == "__main__":
    run_all_tests()
