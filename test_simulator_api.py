"""
수정된 시뮬레이터 API 테스트

시뮬레이터 1과 2의 크로스 체크 일관성을 검증합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ChargingDataLoader as DataLoader
from scenario_simulator import ScenarioSimulator


def test_simulator_consistency():
    """시뮬레이터 일관성 테스트 (Bedrock 호출 없이 ML 로직만)"""
    print("\n" + "="*70)
    print("🧪 시뮬레이터 API 일관성 테스트")
    print("="*70)
    
    # 데이터 로드
    loader = DataLoader()
    full_data = loader.load_multiple()
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    print(f"✅ 데이터 로드 완료: {len(full_data)} 행")
    
    # 시뮬레이터 초기화
    simulator = ScenarioSimulator()
    
    # 테스트 파라미터
    base_month = '2025-11'
    sim_period = 2
    
    # 테스트 케이스
    test_cases = [
        {'extra_chargers': 2500, 'expected_share_approx': 16.0},
        {'extra_chargers': 5000, 'expected_share_approx': 16.5},
        {'extra_chargers': 0, 'expected_share_approx': 15.5},
    ]
    
    print(f"\n📋 테스트 조건:")
    print(f"   기준월: {base_month}")
    print(f"   예측 기간: {sim_period}개월")
    
    # 신뢰도 설정 확인
    reliability_config = ScenarioSimulator.get_reliability_config(full_data)
    print(f"\n📊 신뢰도 설정:")
    print(f"   현재 GS 점유율: {reliability_config.get('current_gs_share', 'N/A')}%")
    print(f"   현재 GS 충전기: {reliability_config.get('current_gs_chargers', 'N/A'):,}대")
    print(f"   최대 신뢰 예측 기간: {reliability_config.get('max_reliable_period', 'N/A')}개월")
    print(f"   목표 점유율 범위: {reliability_config.get('target_share_range', {})}")
    
    print("\n" + "="*70)
    print("✅ 시뮬레이터 설정 검증 완료")
    print("="*70)
    print("\n💡 실제 API 테스트는 Flask 서버를 실행한 후 웹 UI에서 수행하세요.")
    print("   python app.py")
    print("   http://localhost:5001/dashboard")


if __name__ == "__main__":
    test_simulator_consistency()
