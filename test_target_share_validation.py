"""
목표 점유율 범위 검증

목적:
1. 시뮬레이터2에서 사용하는 목표 점유율 범위가 적합한지 검증
2. 신뢰할 수 있는 최소~최대 목표 점유율 범위 계산
3. 백테스트 기반으로 현실적인 범위 도출
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from data_loader import ChargingDataLoader


class TargetShareValidator:
    """목표 점유율 범위 검증기"""
    
    def __init__(self, full_data: pd.DataFrame):
        self.full_data = full_data
        self.all_months = sorted(full_data['snapshot_month'].unique().tolist())
        
        # 데이터 추출
        self.gs_history = self._extract_gs_history()
        self.market_history = self._extract_market_history()
        
    def _extract_gs_history(self) -> List[Dict]:
        """GS차지비 히스토리 추출"""
        gs_data = self.full_data[self.full_data['CPO명'] == 'GS차지비'].copy()
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
                'total_change': int(row.get('총증감', 0)) if pd.notna(row.get('총증감')) else 0
            })
        
        return history
    
    def _extract_market_history(self) -> List[Dict]:
        """시장 전체 히스토리 추출"""
        market_history = []
        for month in self.all_months:
            month_data = self.full_data[self.full_data['snapshot_month'] == month]
            if len(month_data) > 0:
                total_chargers = month_data['총충전기'].sum()
                market_history.append({
                    'month': month,
                    'total_chargers': int(total_chargers)
                })
        return market_history
    
    def analyze_share_range(self) -> Dict:
        """점유율 범위 분석"""
        shares = [h['market_share'] for h in self.gs_history]
        
        current_share = shares[-1]
        min_share = min(shares)
        max_share = max(shares)
        avg_share = np.mean(shares)
        std_share = np.std(shares)
        
        # 월별 변화량 분석
        share_changes = np.diff(shares)
        avg_monthly_change = np.mean(share_changes)
        max_monthly_increase = max(share_changes) if len(share_changes) > 0 else 0
        max_monthly_decrease = min(share_changes) if len(share_changes) > 0 else 0
        
        return {
            'current_share': current_share,
            'min_share': min_share,
            'max_share': max_share,
            'avg_share': avg_share,
            'std_share': std_share,
            'avg_monthly_change': avg_monthly_change,
            'max_monthly_increase': max_monthly_increase,
            'max_monthly_decrease': max_monthly_decrease,
            'data_period': f"{self.all_months[0]} ~ {self.all_months[-1]}",
            'n_months': len(shares)
        }
    
    def calculate_reliable_target_range(self, max_period: int = 8) -> Dict:
        """
        신뢰할 수 있는 목표 점유율 범위 계산
        
        기준:
        1. 과거 데이터의 변동 범위 기반
        2. 예측 기간 동안 현실적으로 달성 가능한 범위
        3. 백테스트 오차를 고려한 안전 마진
        """
        share_analysis = self.analyze_share_range()
        
        current_share = share_analysis['current_share']
        avg_monthly_change = share_analysis['avg_monthly_change']
        max_monthly_increase = share_analysis['max_monthly_increase']
        max_monthly_decrease = share_analysis['max_monthly_decrease']
        std_share = share_analysis['std_share']
        
        print("\n" + "=" * 70)
        print("📊 목표 점유율 범위 검증")
        print("=" * 70)
        
        print(f"\n   현재 점유율: {current_share:.2f}%")
        print(f"   과거 점유율 범위: {share_analysis['min_share']:.2f}% ~ {share_analysis['max_share']:.2f}%")
        print(f"   월평균 변화: {avg_monthly_change:.4f}%p")
        print(f"   최대 월 증가: {max_monthly_increase:.4f}%p")
        print(f"   최대 월 감소: {max_monthly_decrease:.4f}%p")
        print(f"   표준편차: {std_share:.4f}%p")
        
        # 예측 기간별 신뢰 가능한 목표 범위 계산
        print(f"\n   예측 기간별 신뢰 가능한 목표 점유율 범위:")
        print(f"   {'기간':^8} | {'최소 목표':^12} | {'최대 목표':^12} | {'현실적 범위':^20}")
        print("   " + "-" * 60)
        
        period_ranges = {}
        for period in range(1, max_period + 1):
            # 보수적 접근: 과거 최대 변화량 기반
            # 최소: 현재 - (최대 월 감소 * 기간) - 안전마진
            # 최대: 현재 + (최대 월 증가 * 기간) + 안전마진
            
            # 안전 마진: 백테스트 오차 (약 0.3%p) + 표준편차
            safety_margin = 0.3 + std_share * 0.5
            
            # 현실적인 최소/최대 계산
            # 하락 시나리오: 추세 기반 (시장이 더 빨리 성장)
            min_target = current_share + (avg_monthly_change * period) - safety_margin
            min_target = max(min_target, current_share - 3.0)  # 최대 3%p 하락까지만
            
            # 상승 시나리오: 추가 충전기 설치 시
            # 과거 최대 월 증가량의 2배까지 가능하다고 가정
            max_increase_per_month = max(0.1, max_monthly_increase * 2)
            max_target = current_share + (max_increase_per_month * period) + safety_margin
            max_target = min(max_target, current_share + 5.0)  # 최대 5%p 상승까지만
            
            # 최소값은 10% 이상으로 제한
            min_target = max(10.0, min_target)
            
            period_ranges[period] = {
                'min_target': round(min_target, 1),
                'max_target': round(max_target, 1),
                'current': current_share
            }
            
            print(f"   {period}개월{' '*3} | {min_target:^12.1f}% | {max_target:^12.1f}% | {min_target:.1f}% ~ {max_target:.1f}%")
        
        # 8개월 기준 최종 범위
        final_range = period_ranges[max_period]
        
        print("   " + "-" * 60)
        print(f"\n✅ 권장 목표 점유율 범위 (8개월 기준):")
        print(f"   최소: {final_range['min_target']:.1f}%")
        print(f"   최대: {final_range['max_target']:.1f}%")
        print(f"   현재: {current_share:.2f}%")
        
        return {
            'current_share': current_share,
            'recommended_min': final_range['min_target'],
            'recommended_max': final_range['max_target'],
            'period_ranges': period_ranges,
            'share_analysis': share_analysis
        }
    
    def validate_charger_calculation(self, target_share: float, period: int = 8) -> Dict:
        """
        목표 점유율 달성에 필요한 충전기 수 계산 검증
        
        공식: target_share = (gs_chargers + extra) / (market_total + extra) * 100
        역산: extra = (target_share * market_total - gs_chargers * 100) / (100 - target_share)
        """
        n = len(self.gs_history)
        X = np.arange(n).reshape(-1, 1)
        gs_chargers = np.array([h['total_chargers'] for h in self.gs_history])
        market_chargers = np.array([m['total_chargers'] for m in self.market_history])
        
        # 모델 학습
        lr_gs = LinearRegression().fit(X, gs_chargers)
        lr_market = LinearRegression().fit(X, market_chargers)
        
        # 예측 기간 후 baseline 예측
        X_future = np.array([[n + period - 1]])
        pred_gs = lr_gs.predict(X_future)[0]
        pred_market = lr_market.predict(X_future)[0]
        baseline_share = (pred_gs / pred_market) * 100
        
        # 필요 충전기 역산
        # target_share = (pred_gs + extra) / (pred_market + extra) * 100
        # target_share * (pred_market + extra) = (pred_gs + extra) * 100
        # target_share * pred_market + target_share * extra = pred_gs * 100 + extra * 100
        # target_share * pred_market - pred_gs * 100 = extra * 100 - target_share * extra
        # target_share * pred_market - pred_gs * 100 = extra * (100 - target_share)
        # extra = (target_share * pred_market - pred_gs * 100) / (100 - target_share)
        
        if target_share >= 100:
            required_extra = float('inf')
        else:
            required_extra = (target_share * pred_market - pred_gs * 100) / (100 - target_share)
        
        return {
            'target_share': target_share,
            'period': period,
            'baseline_gs': int(pred_gs),
            'baseline_market': int(pred_market),
            'baseline_share': round(baseline_share, 2),
            'required_extra_chargers': int(max(0, required_extra)),
            'monthly_extra': int(max(0, required_extra) / period) if period > 0 else 0
        }


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🚀 목표 점유율 범위 검증 시작")
    print("=" * 70)
    
    # 데이터 로드
    print("\n📥 RAG 데이터 로드 중...")
    loader = ChargingDataLoader()
    full_data = loader.load_multiple()
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    print(f"✅ 데이터 로드 완료: {len(full_data)} 행")
    
    # 검증기 생성
    validator = TargetShareValidator(full_data)
    
    # 목표 점유율 범위 계산
    result = validator.calculate_reliable_target_range(max_period=8)
    
    # 다양한 목표 점유율에 대한 필요 충전기 계산
    print("\n" + "=" * 70)
    print("📊 목표 점유율별 필요 충전기 수 (8개월 기준)")
    print("=" * 70)
    
    current = result['current_share']
    test_targets = [
        round(current - 2, 1),
        round(current - 1, 1),
        round(current, 1),
        round(current + 0.5, 1),
        round(current + 1, 1),
        round(current + 2, 1),
        round(current + 3, 1),
    ]
    
    print(f"\n   {'목표 점유율':^12} | {'필요 충전기':^12} | {'월평균 설치':^12} | {'현실성':^10}")
    print("   " + "-" * 55)
    
    for target in test_targets:
        calc = validator.validate_charger_calculation(target, period=8)
        
        # 현실성 평가 (월 1000대 이하면 현실적)
        monthly = calc['monthly_extra']
        if monthly <= 0:
            status = "✅ 자연 달성"
        elif monthly <= 500:
            status = "✅ 매우 현실적"
        elif monthly <= 1000:
            status = "✅ 현실적"
        elif monthly <= 1500:
            status = "⚠️ 도전적"
        else:
            status = "❌ 비현실적"
        
        extra = calc['required_extra_chargers']
        print(f"   {target:^12.1f}% | {extra:>10,}대 | {monthly:>10,}대/월 | {status}")
    
    print("   " + "-" * 55)
    
    # 최종 권장 범위
    print("\n" + "=" * 70)
    print("📌 최종 권장 설정")
    print("=" * 70)
    print(f"\n   최대 예측 기간: 8개월")
    print(f"   목표 점유율 범위: {result['recommended_min']:.1f}% ~ {result['recommended_max']:.1f}%")
    print(f"   현재 점유율: {result['current_share']:.2f}%")
    
    return result


if __name__ == "__main__":
    main()
