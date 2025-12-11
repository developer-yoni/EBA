"""
백테스트 검증 실행 스크립트

수정된 시뮬레이터의 정확도를 과거 데이터로 검증합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ChargingDataLoader as DataLoader
from simulator_cross_validator import SimulatorCrossValidator


def run_backtest():
    """백테스트 실행"""
    print("\n" + "="*70)
    print("📊 백테스트 검증 실행")
    print("="*70)
    
    # 데이터 로드
    loader = DataLoader()
    full_data = loader.load_multiple()
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    print(f"✅ 데이터 로드 완료: {len(full_data)} 행")
    
    # 검증기 초기화
    validator = SimulatorCrossValidator(full_data)
    
    # 백테스트 실행
    results = validator.run_backtest_validation(
        sim_periods=[1, 2, 3]
    )
    
    # 결과 요약
    if results and results.get('backtest_results'):
        print("\n" + "="*70)
        print("📈 백테스트 결과 요약")
        print("="*70)
        
        import pandas as pd
        df = pd.DataFrame(results['backtest_results'])
        
        # 기간별 통계
        print("\n기간별 예측 오차:")
        for period in sorted(df['sim_period'].unique()):
            period_df = df[df['sim_period'] == period]
            mae = period_df['share_error'].abs().mean()
            mape = period_df['share_error_pct'].mean()
            print(f"  {period}개월: MAE={mae:.4f}%p, MAPE={mape:.2f}%")
        
        # 전체 통계
        print(f"\n전체 통계:")
        print(f"  총 테스트: {len(df)}개")
        print(f"  평균 MAE: {df['share_error'].abs().mean():.4f}%p")
        print(f"  평균 MAPE: {df['share_error_pct'].mean():.2f}%")
    
    return results


def run_cross_validation_test():
    """크로스 검증 테스트"""
    print("\n" + "="*70)
    print("🔄 크로스 검증 테스트")
    print("="*70)
    
    # 데이터 로드
    loader = DataLoader()
    full_data = loader.load_multiple()
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    # 검증기 초기화
    validator = SimulatorCrossValidator(full_data)
    
    # 크로스 검증 실행
    results = validator.run_cross_validation(
        base_month='2025-11',
        sim_period=2,
        test_chargers=[0, 1000, 2000, 2500, 3000, 5000],
        test_shares=[15.0, 15.5, 16.0, 16.5, 17.0]
    )
    
    return results


if __name__ == "__main__":
    # 1. 백테스트 실행
    backtest_results = run_backtest()
    
    # 2. 크로스 검증 테스트
    cross_results = run_cross_validation_test()
    
    print("\n" + "="*70)
    print("✅ 모든 검증 완료")
    print("="*70)
