"""
최대 예측 기간 및 최대 충전기 수 제한값 검증

목적:
1. 현재 6개월 최대 예측 기간이 적합한지 검증
2. 현재 5000대 최대 충전기 수가 적합한지 검증
3. 더 높거나 낮은 값이 더 적합한지 ML 관점에서 분석

검증 기준:
- MAPE 2% 이하 유지
- 신뢰도 95% 이상 유지
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


class MaxLimitsValidator:
    """최대 제한값 검증기"""
    
    # 신뢰도 기준
    RELIABILITY_THRESHOLD = 95.0  # 95% 이상
    MAPE_THRESHOLD = 2.0  # 2% 이하
    
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
    
    def validate_max_period(self, test_periods: List[int] = None) -> Dict:
        """
        최대 예측 기간 검증
        
        다양한 예측 기간에 대해 백테스트를 수행하고
        신뢰도 95% 이상, MAPE 2% 이하를 유지하는 최대 기간 찾기
        """
        if test_periods is None:
            test_periods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        print("\n" + "=" * 70)
        print("📊 최대 예측 기간 검증")
        print("=" * 70)
        print(f"   기준: 신뢰도 >= {self.RELIABILITY_THRESHOLD}%, MAPE <= {self.MAPE_THRESHOLD}%")
        
        results = {}
        
        for period in test_periods:
            period_results = []
            
            # 유효한 기준월 선택 (최소 3개월 학습 + period개월 검증)
            for i in range(3, len(self.all_months) - period):
                base_month = self.all_months[i]
                
                # 학습 데이터
                train_gs = self.gs_history[:i+1]
                train_market = self.market_history[:i+1]
                
                # 검증 데이터
                test_gs = self.gs_history[i+1:i+1+period]
                
                if len(test_gs) < period:
                    continue
                
                # 모델 학습
                n_train = len(train_gs)
                X_train = np.arange(n_train).reshape(-1, 1)
                gs_train = np.array([h['total_chargers'] for h in train_gs])
                market_train = np.array([m['total_chargers'] for m in train_market])
                
                lr_gs = LinearRegression().fit(X_train, gs_train)
                lr_market = LinearRegression().fit(X_train, market_train)
                
                # 예측 및 오차 계산
                errors = []
                for j in range(period):
                    X_pred = np.array([[n_train + j]])
                    pred_gs = lr_gs.predict(X_pred)[0]
                    pred_market = lr_market.predict(X_pred)[0]
                    pred_share = (pred_gs / pred_market) * 100
                    
                    actual_share = test_gs[j]['market_share']
                    error = abs(pred_share - actual_share)
                    pct_error = (error / actual_share) * 100 if actual_share > 0 else 0
                    errors.append(pct_error)
                
                mape = np.mean(errors)
                reliability = 100 - mape
                
                period_results.append({
                    'base_month': base_month,
                    'mape': mape,
                    'reliability': reliability
                })
            
            if period_results:
                avg_mape = np.mean([r['mape'] for r in period_results])
                avg_reliability = np.mean([r['reliability'] for r in period_results])
                min_reliability = min([r['reliability'] for r in period_results])
                max_mape = max([r['mape'] for r in period_results])
                
                results[period] = {
                    'n_tests': len(period_results),
                    'avg_mape': round(avg_mape, 2),
                    'max_mape': round(max_mape, 2),
                    'avg_reliability': round(avg_reliability, 2),
                    'min_reliability': round(min_reliability, 2),
                    'meets_criteria': avg_reliability >= self.RELIABILITY_THRESHOLD and avg_mape <= self.MAPE_THRESHOLD
                }
        
        # 결과 출력
        print(f"\n{'기간':^8} | {'테스트수':^8} | {'평균MAPE':^10} | {'최대MAPE':^10} | {'평균신뢰도':^12} | {'최소신뢰도':^12} | {'기준충족':^8}")
        print("-" * 85)
        
        max_reliable_period = 0
        for period, stats in results.items():
            status = "✅" if stats['meets_criteria'] else "❌"
            print(f"{period}개월{' '*3} | {stats['n_tests']:^8} | {stats['avg_mape']:^10.2f}% | {stats['max_mape']:^10.2f}% | {stats['avg_reliability']:^12.2f}% | {stats['min_reliability']:^12.2f}% | {status:^8}")
            
            if stats['meets_criteria']:
                max_reliable_period = period
        
        print("-" * 85)
        print(f"\n✅ 신뢰도 기준 충족하는 최대 예측 기간: {max_reliable_period}개월")
        
        return {
            'results': results,
            'max_reliable_period': max_reliable_period,
            'current_setting': 6,
            'recommendation': 'keep' if max_reliable_period == 6 else ('increase' if max_reliable_period > 6 else 'decrease'),
            'recommended_value': max_reliable_period
        }
    
    def validate_max_chargers(self, test_chargers: List[int] = None) -> Dict:
        """
        최대 충전기 수 검증
        
        시나리오 예측에서 추가 충전기 수에 따른 예측 정확도 분석
        
        핵심: 추가 충전기가 많아질수록 예측 불확실성이 증가하는지 확인
        - 시뮬레이터 공식: scenario_share = (baseline_gs + extra) / (baseline_market + extra) * 100
        - 이 공식은 수학적으로 extra가 커질수록 점유율 변화가 작아지는 특성이 있음
        """
        if test_chargers is None:
            test_chargers = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        
        print("\n" + "=" * 70)
        print("📊 최대 충전기 수 검증")
        print("=" * 70)
        
        # 현재 데이터 기준 분석
        n = len(self.gs_history)
        X = np.arange(n).reshape(-1, 1)
        gs_chargers = np.array([h['total_chargers'] for h in self.gs_history])
        market_chargers = np.array([m['total_chargers'] for m in self.market_history])
        gs_shares = np.array([h['market_share'] for h in self.gs_history])
        
        # 모델 학습
        lr_gs = LinearRegression().fit(X, gs_chargers)
        lr_market = LinearRegression().fit(X, market_chargers)
        
        # 현재 상태
        current_gs = gs_chargers[-1]
        current_market = market_chargers[-1]
        current_share = gs_shares[-1]
        
        print(f"\n   현재 GS 충전기: {current_gs:,}대")
        print(f"   현재 시장 전체: {current_market:,}대")
        print(f"   현재 점유율: {current_share:.2f}%")
        
        # 과거 월별 증가량 분석
        monthly_changes = [h['total_change'] for h in self.gs_history if h['total_change'] != 0]
        if monthly_changes:
            avg_monthly_change = np.mean(monthly_changes)
            max_monthly_change = max(monthly_changes)
            print(f"\n   과거 월평균 증가량: {avg_monthly_change:.0f}대")
            print(f"   과거 최대 월 증가량: {max_monthly_change}대")
        else:
            avg_monthly_change = 0
            max_monthly_change = 0
        
        # 6개월 예측 기준 분석
        prediction_months = 6
        X_future = np.array([[n + prediction_months - 1]])
        pred_gs_baseline = lr_gs.predict(X_future)[0]
        pred_market_baseline = lr_market.predict(X_future)[0]
        baseline_share = (pred_gs_baseline / pred_market_baseline) * 100
        
        print(f"\n   6개월 후 예측 (baseline):")
        print(f"   - GS 충전기: {pred_gs_baseline:,.0f}대")
        print(f"   - 시장 전체: {pred_market_baseline:,.0f}대")
        print(f"   - 점유율: {baseline_share:.2f}%")
        
        # 추가 충전기별 시나리오 분석
        print(f"\n{'추가충전기':^12} | {'예측점유율':^12} | {'점유율증가':^12} | {'증가효율':^15} | {'현실성':^10}")
        print("-" * 70)
        
        results = {}
        for extra in test_chargers:
            # 시나리오 점유율 계산 (시뮬레이터 공식)
            scenario_gs = pred_gs_baseline + extra
            scenario_market = pred_market_baseline + extra
            scenario_share = (scenario_gs / scenario_market) * 100
            
            share_increase = scenario_share - baseline_share
            efficiency = (share_increase / extra * 1000) if extra > 0 else 0  # 1000대당 점유율 증가
            
            # 현실성 평가 (6개월간 달성 가능한 증가량 기준)
            # 과거 최대 월 증가량 * 6 * 1.5 (50% 여유)
            realistic_max = max_monthly_change * 6 * 1.5 if max_monthly_change > 0 else 3000
            is_realistic = extra <= realistic_max
            
            results[extra] = {
                'scenario_share': round(scenario_share, 2),
                'share_increase': round(share_increase, 4),
                'efficiency': round(efficiency, 4),
                'is_realistic': is_realistic,
                'realistic_max': realistic_max
            }
            
            status = "✅" if is_realistic else "⚠️"
            print(f"{extra:>10,}대 | {scenario_share:^12.2f}% | {share_increase:^12.4f}%p | {efficiency:^15.4f}%p/1000대 | {status:^10}")
        
        print("-" * 70)
        
        # 현실적인 최대 충전기 수 결정
        realistic_max_chargers = int(realistic_max)
        
        # 효율성 분석 (수확체감 법칙)
        print("\n📈 효율성 분석 (수확체감 법칙):")
        print("   추가 충전기가 많아질수록 점유율 증가 효율이 감소합니다.")
        print("   이는 시장 전체도 함께 증가하기 때문입니다.")
        
        # 효율성이 급격히 떨어지는 지점 찾기
        efficiencies = [(k, v['efficiency']) for k, v in results.items() if k > 0]
        if len(efficiencies) >= 2:
            first_eff = efficiencies[0][1]
            for chargers, eff in efficiencies:
                if eff < first_eff * 0.5:  # 효율이 50% 이하로 떨어지는 지점
                    print(f"   → {chargers:,}대 이상에서 효율이 50% 이하로 감소")
                    break
        
        # 권장 최대값 결정
        # 기준: 현실성 + 효율성
        recommended_max = min(realistic_max_chargers, 10000)
        recommended_max = max(recommended_max, 3000)  # 최소 3000대
        recommended_max = (recommended_max // 1000) * 1000  # 1000단위로 반올림
        
        print(f"\n✅ 권장 최대 충전기 수: {recommended_max:,}대")
        print(f"   (과거 최대 월 증가량 {max_monthly_change}대 × 6개월 × 1.5 = {realistic_max_chargers:,}대 기준)")
        
        current_setting = 5000
        if recommended_max == current_setting:
            recommendation = 'keep'
        elif recommended_max > current_setting:
            recommendation = 'increase'
        else:
            recommendation = 'decrease'
        
        return {
            'results': results,
            'realistic_max': realistic_max_chargers,
            'recommended_max': recommended_max,
            'current_setting': current_setting,
            'recommendation': recommendation,
            'max_monthly_change': max_monthly_change,
            'avg_monthly_change': avg_monthly_change
        }
    
    def generate_validation_report(self) -> str:
        """종합 검증 리포트 생성"""
        period_validation = self.validate_max_period()
        charger_validation = self.validate_max_chargers()
        
        report = []
        report.append("\n" + "=" * 70)
        report.append("📋 최대 제한값 검증 종합 리포트")
        report.append("=" * 70)
        
        # 예측 기간 검증 결과
        report.append("\n[1] 최대 예측 기간 검증 결과")
        report.append("-" * 50)
        report.append(f"   현재 설정: {period_validation['current_setting']}개월")
        report.append(f"   권장 값: {period_validation['recommended_value']}개월")
        report.append(f"   권장 조치: {period_validation['recommendation']}")
        
        if period_validation['recommendation'] == 'keep':
            report.append(f"   → ✅ 현재 설정 유지 (변경 불필요)")
        elif period_validation['recommendation'] == 'increase':
            report.append(f"   → ⬆️ {period_validation['recommended_value']}개월로 증가 가능")
        else:
            report.append(f"   → ⬇️ {period_validation['recommended_value']}개월로 감소 권장")
        
        # 충전기 수 검증 결과
        report.append("\n[2] 최대 충전기 수 검증 결과")
        report.append("-" * 50)
        report.append(f"   현재 설정: {charger_validation['current_setting']:,}대")
        report.append(f"   권장 값: {charger_validation['recommended_max']:,}대")
        report.append(f"   권장 조치: {charger_validation['recommendation']}")
        
        if charger_validation['recommendation'] == 'keep':
            report.append(f"   → ✅ 현재 설정 유지 (변경 불필요)")
        elif charger_validation['recommendation'] == 'increase':
            report.append(f"   → ⬆️ {charger_validation['recommended_max']:,}대로 증가 가능")
        else:
            report.append(f"   → ⬇️ {charger_validation['recommended_max']:,}대로 감소 권장")
        
        # 최종 결론
        report.append("\n" + "=" * 70)
        report.append("📌 최종 결론")
        report.append("=" * 70)
        
        changes_needed = []
        if period_validation['recommendation'] != 'keep':
            changes_needed.append(f"예측 기간: {period_validation['current_setting']} → {period_validation['recommended_value']}개월")
        if charger_validation['recommendation'] != 'keep':
            changes_needed.append(f"최대 충전기: {charger_validation['current_setting']:,} → {charger_validation['recommended_max']:,}대")
        
        if not changes_needed:
            report.append("\n✅ 현재 설정이 최적입니다. 코드 수정이 필요하지 않습니다.")
        else:
            report.append("\n⚠️ 다음 설정 변경이 권장됩니다:")
            for change in changes_needed:
                report.append(f"   - {change}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report), period_validation, charger_validation


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🚀 최대 제한값 검증 시작")
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
    validator = MaxLimitsValidator(full_data)
    
    # 종합 리포트 생성
    report, period_result, charger_result = validator.generate_validation_report()
    print(report)
    
    return {
        'period_validation': period_result,
        'charger_validation': charger_result
    }


if __name__ == "__main__":
    main()
