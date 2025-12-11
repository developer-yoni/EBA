#!/usr/bin/env python3
"""
백테스트 실행 스크립트

사용법:
    python run_backtest.py
"""

import sys
import pandas as pd
from data_loader import ChargingDataLoader
from backtest_simulator import BacktestSimulator, run_full_backtest


def main():
    print("\n" + "="*70)
    print("🔬 ScenarioSimulator 백테스트 실행")
    print("="*70)
    
    # 1. 데이터 로드
    print("\n📥 데이터 로드 중...")
    loader = ChargingDataLoader()
    full_data = loader.load_multiple()  # 모든 월 데이터 로드
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        sys.exit(1)
    
    all_months = sorted(full_data['snapshot_month'].unique().tolist())
    print(f"✅ 데이터 로드 완료: {len(full_data):,}행, {len(all_months)}개월")
    print(f"   기간: {all_months[0]} ~ {all_months[-1]}")
    
    # 2. 백테스트 실행
    print("\n" + "-"*70)
    results, analysis, backtester = run_full_backtest(full_data)
    
    # 3. 결과 저장
    output_file = "backtest_results.csv"
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {output_file}")
    
    # 4. 상세 결과 출력
    print("\n" + "="*70)
    print("📋 백테스트 상세 결과")
    print("="*70)
    
    valid_results = results[results['mae'].notna()]
    print(valid_results.to_string(index=False))
    
    # 5. 권장사항 출력
    print("\n" + "="*70)
    print("💡 권장사항")
    print("="*70)
    
    if 'recommended_max_period' in analysis:
        print(f"  1. 권장 최대 예측 기간: {analysis['recommended_max_period']}개월")
    
    if 'suggested_thresholds' in analysis:
        thresholds = analysis['suggested_thresholds']
        print(f"  2. Confidence Level 경계 조정 제안:")
        print(f"     - HIGH >= {thresholds['high']}")
        print(f"     - MEDIUM >= {thresholds['medium']}")
    
    if 'correlation' in analysis:
        corr = analysis['correlation']
        if corr['score_vs_mape'] < -0.3:
            print(f"  3. ✅ Confidence Score가 오차와 음의 상관관계 ({corr['score_vs_mape']:.3f})")
            print(f"     → 신뢰도 점수가 예측 품질을 잘 반영함")
        else:
            print(f"  3. ⚠️ Confidence Score와 오차 간 상관관계 약함 ({corr['score_vs_mape']:.3f})")
            print(f"     → 신뢰도 계산 로직 개선 필요")
    
    print("\n" + "="*70)
    print("✅ 백테스트 완료")
    print("="*70 + "\n")
    
    return results, analysis, backtester


if __name__ == "__main__":
    main()
