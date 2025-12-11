"""
시뮬레이터 크로스 검증 테스트

문제 상황:
- 시뮬레이터 1: 2개월간 2,500대 추가 → 예상 점유율 15.98%
- 시뮬레이터 2: 2개월 후 16% 유지 목표 → 필요 충전기 0대

이 불일치의 원인을 분석하고 수정합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ChargingDataLoader as DataLoader
from simulator_cross_validator import SimulatorCrossValidator, run_full_validation


def diagnose_inconsistency():
    """불일치 원인 진단"""
    print("\n" + "="*70)
    print("🔍 시뮬레이터 불일치 진단")
    print("="*70)
    
    # 데이터 로드
    loader = DataLoader()
    full_data = loader.load_multiple()  # 모든 월 데이터 로드
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    print(f"✅ 데이터 로드 완료: {len(full_data)} 행")
    
    # 검증기 초기화
    validator = SimulatorCrossValidator(full_data)
    
    # 문제 상황 재현
    base_month = '2025-11'
    sim_period = 2
    extra_chargers = 2500
    target_share = 16.0
    
    print(f"\n📋 문제 상황 재현")
    print(f"   기준월: {base_month}")
    print(f"   예측 기간: {sim_period}개월")
    print(f"   추가 충전기: {extra_chargers:,}대")
    print(f"   목표 점유율: {target_share}%")
    
    # 현재 상태 확인
    current = validator.get_actual_data(base_month)
    if current:
        print(f"\n📊 현재 상태 ({base_month})")
        print(f"   GS차지비 충전기: {current['gs_chargers']:,}대")
        print(f"   시장 전체 충전기: {current['market_total']:,}대")
        print(f"   현재 점유율: {current['market_share']:.4f}%")
    
    # ML 예측 확인
    ml_result = validator.calculate_ml_predictions(base_month, sim_period)
    if 'error' not in ml_result:
        print(f"\n📈 ML 추세 분석")
        print(f"   GS 월평균 증가: {ml_result['trends']['gs_monthly_increase']:,.0f}대/월")
        print(f"   시장 월평균 증가: {ml_result['trends']['market_monthly_increase']:,.0f}대/월")
        print(f"   점유율 월평균 변화: {ml_result['trends']['share_monthly_change']:+.4f}%p/월")
        
        print(f"\n📅 Baseline 예측 ({sim_period}개월 후)")
        for pred in ml_result['predictions']:
            print(f"   {pred['months_ahead']}개월 후: GS {pred['pred_gs_chargers']:,}대, "
                  f"시장 {pred['pred_market_total']:,}대, 점유율 {pred['pred_share']:.4f}%")
    
    # 시뮬레이터 1 테스트
    print(f"\n{'='*60}")
    print(f"🔬 시뮬레이터 1: {extra_chargers:,}대 추가 시 점유율 예측")
    print(f"{'='*60}")
    
    sim1_result = validator.simulate_with_extra_chargers(base_month, sim_period, extra_chargers)
    
    print(f"   Baseline 최종 점유율: {sim1_result.get('baseline_final_share', 0):.4f}%")
    print(f"   시나리오 최종 점유율: {sim1_result.get('scenario_final_share', 0):.4f}%")
    print(f"   점유율 증가: {sim1_result.get('share_increase', 0):+.4f}%p")
    
    print(f"\n   월별 상세:")
    for pred in sim1_result.get('predictions', []):
        print(f"   {pred['months_ahead']}개월 후: "
              f"GS {pred['baseline_gs']:,} → {pred['scenario_gs']:,}대 (+{pred['added_chargers']:,}), "
              f"시장 {pred['baseline_market']:,} → {pred['scenario_market']:,}대, "
              f"점유율 {pred['baseline_share']:.4f}% → {pred['scenario_share']:.4f}%")
    
    # 시뮬레이터 2 테스트
    print(f"\n{'='*60}")
    print(f"🔬 시뮬레이터 2: {target_share}% 달성에 필요한 충전기")
    print(f"{'='*60}")
    
    sim2_result = validator.calculate_required_chargers(base_month, sim_period, target_share)
    
    print(f"   Baseline 최종 점유율: {sim2_result.get('baseline_final_share', 0):.4f}%")
    print(f"   필요 추가 충전기: {sim2_result.get('required_extra_chargers', 0):,}대")
    print(f"   월평균 필요: {sim2_result.get('monthly_required', 0):,}대/월")
    print(f"   달성 가능성: {sim2_result.get('feasibility', 'N/A')}")
    print(f"   사유: {sim2_result.get('feasibility_reason', 'N/A')}")
    
    cross_val = sim2_result.get('cross_validation', {})
    print(f"\n   크로스 검증:")
    print(f"   - 검증 점유율: {cross_val.get('verified_share', 0):.4f}%")
    print(f"   - 목표 점유율: {cross_val.get('target_share', 0):.4f}%")
    print(f"   - 오차: {cross_val.get('error', 0):.4f}%p")
    print(f"   - 일관성: {'✅' if cross_val.get('is_consistent') else '❌'}")
    
    # 역방향 검증: 시뮬레이터 1 결과로 시뮬레이터 2 테스트
    print(f"\n{'='*60}")
    print(f"🔄 역방향 검증: 시뮬레이터 1 결과 → 시뮬레이터 2")
    print(f"{'='*60}")
    
    sim1_final_share = sim1_result.get('scenario_final_share', 0)
    print(f"   시뮬레이터 1 예측 점유율: {sim1_final_share:.4f}%")
    
    # 이 점유율을 달성하려면 얼마나 필요한지 역계산
    reverse_result = validator.calculate_required_chargers(base_month, sim_period, sim1_final_share)
    
    print(f"   역계산 필요 충전기: {reverse_result.get('required_extra_chargers', 0):,}대")
    print(f"   원래 입력 충전기: {extra_chargers:,}대")
    print(f"   차이: {reverse_result.get('required_extra_chargers', 0) - extra_chargers:,}대")
    
    # 불일치 원인 분석
    print(f"\n{'='*60}")
    print(f"📋 불일치 원인 분석")
    print(f"{'='*60}")
    
    # 핵심 문제: 시장 성장률 반영 방식
    if ml_result and 'error' not in ml_result:
        gs_growth = ml_result['trends']['gs_monthly_increase']
        market_growth = ml_result['trends']['market_monthly_increase']
        
        print(f"\n   1. 시장 성장률 분석:")
        print(f"      - GS 월평균 증가: {gs_growth:,.0f}대")
        print(f"      - 시장 월평균 증가: {market_growth:,.0f}대")
        print(f"      - GS 비중: {gs_growth/market_growth*100:.1f}%" if market_growth > 0 else "")
        
        # 점유율 변화 예측
        current_share = current['market_share'] if current else 0
        baseline_final = sim1_result.get('baseline_final_share', 0)
        
        print(f"\n   2. 점유율 변화 분석:")
        print(f"      - 현재 점유율: {current_share:.4f}%")
        print(f"      - {sim_period}개월 후 baseline: {baseline_final:.4f}%")
        print(f"      - 변화: {baseline_final - current_share:+.4f}%p")
        
        if baseline_final < current_share:
            print(f"\n   ⚠️ 핵심 발견: 시장 성장률이 GS 성장률보다 높아 점유율이 자연 하락합니다!")
            print(f"      → 현재 점유율 유지를 위해서도 추가 설치가 필요합니다.")
        
        # 16% 유지에 필요한 충전기 계산
        print(f"\n   3. 목표 점유율 {target_share}% 달성 분석:")
        
        if baseline_final >= target_share:
            print(f"      - Baseline({baseline_final:.4f}%)이 목표({target_share}%)보다 높음")
            print(f"      - 추가 설치 불필요 (자연 달성)")
        else:
            gap = target_share - baseline_final
            print(f"      - Baseline({baseline_final:.4f}%)과 목표({target_share}%)의 차이: {gap:.4f}%p")
            print(f"      - 이 차이를 메우기 위한 추가 충전기 필요")
    
    return {
        'current': current,
        'ml_result': ml_result,
        'sim1_result': sim1_result,
        'sim2_result': sim2_result,
        'reverse_result': reverse_result
    }


def run_comprehensive_test():
    """종합 테스트 실행"""
    print("\n" + "="*70)
    print("🧪 종합 크로스 검증 테스트")
    print("="*70)
    
    # 데이터 로드
    loader = DataLoader()
    full_data = loader.load_multiple()  # 모든 월 데이터 로드
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    # 전체 검증 실행
    results = run_full_validation(full_data, base_month='2025-11', sim_period=2)
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("시뮬레이터 크로스 검증 테스트")
    print("="*70)
    
    # 1. 불일치 진단
    diagnosis = diagnose_inconsistency()
    
    # 2. 종합 테스트
    # results = run_comprehensive_test()
    
    print("\n" + "="*70)
    print("✅ 테스트 완료")
    print("="*70)
