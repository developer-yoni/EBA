"""
시뮬레이터 크로스 검증 모듈

목적:
1. 시뮬레이터 1 (충전기 추가 → 점유율 예측)과 
   시뮬레이터 2 (목표 점유율 → 필요 충전기 역계산)의 일관성 검증
2. 백테스트를 통한 예측 정확도 검증
3. 불일치 원인 분석 및 수정

핵심 원칙:
- 시뮬레이터 1: extra_chargers → predicted_share
- 시뮬레이터 2: target_share → required_chargers
- 크로스 체크: sim1(sim2_result) ≈ target_share
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression


class SimulatorCrossValidator:
    """시뮬레이터 크로스 검증 클래스"""
    
    def __init__(self, full_data: pd.DataFrame):
        self.full_data = full_data
        self.all_months = sorted(full_data['snapshot_month'].unique().tolist())
        self.validation_results = []
        
    def extract_gs_data(self, up_to_month: str = None) -> pd.DataFrame:
        """GS차지비 데이터 추출"""
        gs_data = self.full_data[self.full_data['CPO명'] == 'GS차지비'].copy()
        if up_to_month:
            gs_data = gs_data[gs_data['snapshot_month'] <= up_to_month]
        return gs_data.sort_values('snapshot_month')
    
    def extract_market_totals(self, up_to_month: str = None) -> pd.DataFrame:
        """월별 시장 전체 충전기 수 추출"""
        data = self.full_data.copy()
        if up_to_month:
            data = data[data['snapshot_month'] <= up_to_month]
        
        market_totals = data.groupby('snapshot_month').agg({
            '총충전기': 'sum'
        }).reset_index()
        market_totals.columns = ['month', 'market_total']
        return market_totals.sort_values('month')
    
    def get_actual_data(self, month: str) -> Dict:
        """특정 월의 실제 데이터 조회"""
        gs_row = self.full_data[
            (self.full_data['snapshot_month'] == month) & 
            (self.full_data['CPO명'] == 'GS차지비')
        ]
        
        if len(gs_row) == 0:
            return None
        
        row = gs_row.iloc[0]
        market_share = row.get('시장점유율', 0)
        if pd.notna(market_share) and market_share < 1:
            market_share = market_share * 100
        
        # 시장 전체 충전기
        month_data = self.full_data[self.full_data['snapshot_month'] == month]
        market_total = month_data['총충전기'].sum()
        
        return {
            'month': month,
            'gs_chargers': int(row.get('총충전기', 0)),
            'market_share': round(float(market_share), 4),
            'market_total': int(market_total),
            'gs_change': int(row.get('총증감', 0)) if pd.notna(row.get('총증감')) else 0
        }
    
    def calculate_ml_predictions(self, base_month: str, sim_period: int) -> Dict:
        """
        ML 기반 예측 수행 (시뮬레이터 공통 로직)
        
        핵심: 점유율 = GS충전기 / 시장전체충전기 * 100
        """
        # 기준월까지의 데이터만 사용
        gs_data = self.extract_gs_data(up_to_month=base_month)
        market_data = self.extract_market_totals(up_to_month=base_month)
        
        if len(gs_data) < 2:
            return {'error': '데이터 부족'}
        
        # 데이터 준비
        n = len(gs_data)
        months_idx = np.arange(n).reshape(-1, 1)
        
        gs_chargers = gs_data['총충전기'].values
        gs_shares = []
        for _, row in gs_data.iterrows():
            ms = row.get('시장점유율', 0)
            if pd.notna(ms) and ms < 1:
                ms = ms * 100
            gs_shares.append(float(ms) if pd.notna(ms) else 0)
        gs_shares = np.array(gs_shares)
        
        market_totals = market_data['market_total'].values
        
        # 선형 회귀 - GS 충전기
        lr_gs = LinearRegression()
        lr_gs.fit(months_idx, gs_chargers)
        gs_slope = lr_gs.coef_[0]
        gs_intercept = lr_gs.intercept_
        
        # 선형 회귀 - 시장 전체
        lr_market = LinearRegression()
        lr_market.fit(months_idx, market_totals)
        market_slope = lr_market.coef_[0]
        market_intercept = lr_market.intercept_
        
        # 선형 회귀 - 점유율 (참고용)
        lr_share = LinearRegression()
        lr_share.fit(months_idx, gs_shares)
        share_slope = lr_share.coef_[0]
        
        # 현재 상태 (기준월)
        current_gs = int(gs_chargers[-1])
        current_market = int(market_totals[-1])
        current_share = gs_shares[-1]
        
        # 미래 예측
        predictions = []
        for i in range(1, sim_period + 1):
            future_idx = n + i - 1
            
            # 충전기 수 예측
            pred_gs = gs_intercept + gs_slope * future_idx
            pred_market = market_intercept + market_slope * future_idx
            
            # 점유율 계산 (핵심: 충전기 비율로 계산)
            pred_share = (pred_gs / pred_market) * 100 if pred_market > 0 else 0
            
            predictions.append({
                'months_ahead': i,
                'pred_gs_chargers': int(pred_gs),
                'pred_market_total': int(pred_market),
                'pred_share': round(pred_share, 4)
            })
        
        return {
            'base_month': base_month,
            'n_data_points': n,
            'current': {
                'gs_chargers': current_gs,
                'market_total': current_market,
                'market_share': round(current_share, 4)
            },
            'trends': {
                'gs_monthly_increase': round(gs_slope, 2),
                'market_monthly_increase': round(market_slope, 2),
                'share_monthly_change': round(share_slope, 4)
            },
            'predictions': predictions
        }
    
    def simulate_with_extra_chargers(
        self, 
        base_month: str, 
        sim_period: int, 
        extra_chargers: int
    ) -> Dict:
        """
        시뮬레이터 1: 추가 충전기 → 예상 점유율
        
        핵심 로직:
        1. 기준월까지의 추세로 baseline 예측
        2. 추가 충전기를 GS에만 더함
        3. 시장 전체는 baseline 추세 유지 (GS 추가분 포함)
        4. 점유율 = (GS baseline + 추가) / (시장 baseline + 추가) * 100
        """
        ml_result = self.calculate_ml_predictions(base_month, sim_period)
        
        if 'error' in ml_result:
            return ml_result
        
        current = ml_result['current']
        trends = ml_result['trends']
        baseline_preds = ml_result['predictions']
        
        # 월별 추가 충전기 분배 (균등)
        monthly_extra = extra_chargers / sim_period if sim_period > 0 else 0
        
        scenario_predictions = []
        cumulative_extra = 0
        
        for pred in baseline_preds:
            i = pred['months_ahead']
            cumulative_extra += monthly_extra
            
            # 시나리오: GS에 추가 충전기 반영
            scenario_gs = pred['pred_gs_chargers'] + cumulative_extra
            
            # 시장 전체도 GS 추가분만큼 증가
            scenario_market = pred['pred_market_total'] + cumulative_extra
            
            # 점유율 재계산
            scenario_share = (scenario_gs / scenario_market) * 100 if scenario_market > 0 else 0
            
            scenario_predictions.append({
                'months_ahead': i,
                'baseline_gs': pred['pred_gs_chargers'],
                'scenario_gs': int(scenario_gs),
                'added_chargers': int(cumulative_extra),
                'baseline_market': pred['pred_market_total'],
                'scenario_market': int(scenario_market),
                'baseline_share': pred['pred_share'],
                'scenario_share': round(scenario_share, 4)
            })
        
        final_pred = scenario_predictions[-1] if scenario_predictions else {}
        
        return {
            'type': 'simulator1',
            'input': {
                'base_month': base_month,
                'sim_period': sim_period,
                'extra_chargers': extra_chargers
            },
            'current': current,
            'trends': trends,
            'baseline_final_share': final_pred.get('baseline_share', 0),
            'scenario_final_share': final_pred.get('scenario_share', 0),
            'share_increase': round(
                final_pred.get('scenario_share', 0) - final_pred.get('baseline_share', 0), 4
            ),
            'predictions': scenario_predictions
        }
    
    def calculate_required_chargers(
        self, 
        base_month: str, 
        sim_period: int, 
        target_share: float
    ) -> Dict:
        """
        시뮬레이터 2: 목표 점유율 → 필요 충전기
        
        핵심 로직:
        1. 기준월까지의 추세로 baseline 예측
        2. 목표 점유율 달성에 필요한 GS 충전기 계산
        3. 필요 충전기 = 목표GS - baseline GS
        
        수정된 공식:
        - 목표점유율 = (GS baseline + 추가) / (시장 baseline + 추가) * 100
        - 추가 = (목표점유율 * 시장baseline - 100 * GS baseline) / (100 - 목표점유율)
        """
        ml_result = self.calculate_ml_predictions(base_month, sim_period)
        
        if 'error' in ml_result:
            return ml_result
        
        current = ml_result['current']
        trends = ml_result['trends']
        baseline_preds = ml_result['predictions']
        
        # 최종 예측 (sim_period 후)
        final_baseline = baseline_preds[-1] if baseline_preds else {}
        baseline_gs = final_baseline.get('pred_gs_chargers', current['gs_chargers'])
        baseline_market = final_baseline.get('pred_market_total', current['market_total'])
        baseline_share = final_baseline.get('pred_share', current['market_share'])
        
        # 필요 충전기 계산 (수정된 공식)
        # target_share = (baseline_gs + extra) / (baseline_market + extra) * 100
        # 정리하면:
        # extra = (target_share * baseline_market - 100 * baseline_gs) / (100 - target_share)
        
        if target_share >= 100:
            return {
                'error': '목표 점유율이 100% 이상입니다',
                'target_share': target_share
            }
        
        numerator = (target_share * baseline_market) - (100 * baseline_gs)
        denominator = 100 - target_share
        
        if denominator == 0:
            required_extra = 0
        else:
            required_extra = numerator / denominator
        
        # 음수면 이미 달성 (추가 설치 불필요)
        if required_extra < 0:
            required_extra = 0
            feasibility = 'ALREADY_ACHIEVABLE'
            feasibility_reason = f'현재 추세로 {sim_period}개월 후 {baseline_share:.2f}%로 목표({target_share:.2f}%)를 초과 달성합니다.'
        else:
            required_extra = int(required_extra)
            monthly_required = required_extra / sim_period if sim_period > 0 else 0
            
            # 달성 가능성 평가
            avg_monthly = trends['gs_monthly_increase']
            if avg_monthly > 0:
                ratio = monthly_required / avg_monthly
                if ratio <= 1.5:
                    feasibility = 'ACHIEVABLE'
                elif ratio <= 3:
                    feasibility = 'CHALLENGING'
                else:
                    feasibility = 'DIFFICULT'
            else:
                feasibility = 'CHALLENGING'
            
            feasibility_reason = f'월평균 {monthly_required:.0f}대 설치 필요 (과거 평균: {avg_monthly:.0f}대/월)'
        
        # 검증: 계산된 충전기로 시뮬레이션 1 실행
        if required_extra > 0:
            verification = self.simulate_with_extra_chargers(
                base_month, sim_period, required_extra
            )
            verified_share = verification.get('scenario_final_share', 0)
        else:
            verified_share = baseline_share
        
        return {
            'type': 'simulator2',
            'input': {
                'base_month': base_month,
                'sim_period': sim_period,
                'target_share': target_share
            },
            'current': current,
            'trends': trends,
            'baseline_final_share': round(baseline_share, 4),
            'required_extra_chargers': int(required_extra),
            'monthly_required': int(required_extra / sim_period) if sim_period > 0 else 0,
            'feasibility': feasibility,
            'feasibility_reason': feasibility_reason,
            # 크로스 검증 결과
            'cross_validation': {
                'verified_share': round(verified_share, 4),
                'target_share': target_share,
                'error': round(abs(verified_share - target_share), 4),
                'is_consistent': abs(verified_share - target_share) < 0.1
            }
        }
    
    def run_cross_validation(
        self, 
        base_month: str, 
        sim_period: int,
        test_chargers: List[int] = None,
        test_shares: List[float] = None
    ) -> Dict:
        """
        크로스 검증 실행
        
        테스트 케이스:
        1. 시뮬레이터 1 → 시뮬레이터 2 → 시뮬레이터 1 (일관성 검증)
        2. 실제 데이터와 비교 (정확도 검증)
        """
        if test_chargers is None:
            test_chargers = [0, 500, 1000, 2000, 2500, 5000]
        
        if test_shares is None:
            # 현재 점유율 기준으로 테스트 범위 설정
            current = self.get_actual_data(base_month)
            if current:
                current_share = current['market_share']
                test_shares = [
                    round(current_share - 1, 1),
                    round(current_share, 1),
                    round(current_share + 0.5, 1),
                    round(current_share + 1, 1),
                    round(current_share + 2, 1)
                ]
            else:
                test_shares = [14, 15, 16, 17, 18]
        
        results = {
            'base_month': base_month,
            'sim_period': sim_period,
            'current_data': self.get_actual_data(base_month),
            'simulator1_tests': [],
            'simulator2_tests': [],
            'cross_validation_summary': {}
        }
        
        # 시뮬레이터 1 테스트
        print(f"\n{'='*60}")
        print(f"🔬 시뮬레이터 1 테스트 (충전기 → 점유율)")
        print(f"{'='*60}")
        
        for extra in test_chargers:
            sim1_result = self.simulate_with_extra_chargers(base_month, sim_period, extra)
            results['simulator1_tests'].append(sim1_result)
            
            print(f"  +{extra:,}대 → {sim1_result.get('scenario_final_share', 0):.2f}% "
                  f"(baseline: {sim1_result.get('baseline_final_share', 0):.2f}%)")
        
        # 시뮬레이터 2 테스트
        print(f"\n{'='*60}")
        print(f"🔬 시뮬레이터 2 테스트 (목표 점유율 → 필요 충전기)")
        print(f"{'='*60}")
        
        for target in test_shares:
            sim2_result = self.calculate_required_chargers(base_month, sim_period, target)
            results['simulator2_tests'].append(sim2_result)
            
            cross_val = sim2_result.get('cross_validation', {})
            print(f"  목표 {target:.1f}% → 필요 {sim2_result.get('required_extra_chargers', 0):,}대 "
                  f"(검증: {cross_val.get('verified_share', 0):.2f}%, "
                  f"오차: {cross_val.get('error', 0):.4f}%p)")
        
        # 크로스 검증 요약
        all_errors = [
            t.get('cross_validation', {}).get('error', 0) 
            for t in results['simulator2_tests']
            if t.get('cross_validation')
        ]
        
        results['cross_validation_summary'] = {
            'total_tests': len(results['simulator2_tests']),
            'avg_error': round(np.mean(all_errors), 4) if all_errors else None,
            'max_error': round(np.max(all_errors), 4) if all_errors else None,
            'all_consistent': all(e < 0.1 for e in all_errors) if all_errors else False
        }
        
        print(f"\n{'='*60}")
        print(f"📊 크로스 검증 요약")
        print(f"{'='*60}")
        print(f"  평균 오차: {results['cross_validation_summary']['avg_error']:.4f}%p")
        print(f"  최대 오차: {results['cross_validation_summary']['max_error']:.4f}%p")
        print(f"  일관성: {'✅ 통과' if results['cross_validation_summary']['all_consistent'] else '❌ 불일치'}")
        
        return results
    
    def run_backtest_validation(
        self, 
        base_months: List[str] = None,
        sim_periods: List[int] = None
    ) -> Dict:
        """
        백테스트 검증: 과거 데이터로 예측 정확도 검증
        """
        if base_months is None:
            # 검증 가능한 기준월 선택 (최소 3개월 학습, 최소 1개월 검증)
            base_months = self.all_months[2:-1]  # 처음 2개월 제외, 마지막 1개월 검증용
        
        if sim_periods is None:
            sim_periods = [1, 2, 3]
        
        results = {
            'backtest_results': [],
            'summary': {}
        }
        
        print(f"\n{'='*60}")
        print(f"📊 백테스트 검증 시작")
        print(f"   기준월: {len(base_months)}개 ({base_months[0]} ~ {base_months[-1]})")
        print(f"   예측 기간: {sim_periods}")
        print(f"{'='*60}")
        
        for base_month in base_months:
            for sim_period in sim_periods:
                # 검증 대상 월 계산
                base_date = datetime.strptime(base_month, '%Y-%m')
                target_date = base_date + relativedelta(months=sim_period)
                target_month = target_date.strftime('%Y-%m')
                
                # 실제 데이터가 있는지 확인
                if target_month not in self.all_months:
                    continue
                
                # 예측 수행
                ml_result = self.calculate_ml_predictions(base_month, sim_period)
                if 'error' in ml_result:
                    continue
                
                # 실제값 조회
                actual = self.get_actual_data(target_month)
                if not actual:
                    continue
                
                # 예측값
                pred = ml_result['predictions'][-1] if ml_result['predictions'] else {}
                pred_share = pred.get('pred_share', 0)
                pred_gs = pred.get('pred_gs_chargers', 0)
                
                # 오차 계산
                share_error = pred_share - actual['market_share']
                charger_error = pred_gs - actual['gs_chargers']
                
                result = {
                    'base_month': base_month,
                    'target_month': target_month,
                    'sim_period': sim_period,
                    'predicted_share': round(pred_share, 4),
                    'actual_share': actual['market_share'],
                    'share_error': round(share_error, 4),
                    'share_error_pct': round(abs(share_error) / actual['market_share'] * 100, 2) if actual['market_share'] > 0 else 0,
                    'predicted_chargers': pred_gs,
                    'actual_chargers': actual['gs_chargers'],
                    'charger_error': charger_error
                }
                
                results['backtest_results'].append(result)
                
                print(f"  {base_month} → {target_month} ({sim_period}개월): "
                      f"예측 {pred_share:.2f}% vs 실제 {actual['market_share']:.2f}% "
                      f"(오차: {share_error:+.2f}%p)")
        
        # 요약 통계
        if results['backtest_results']:
            df = pd.DataFrame(results['backtest_results'])
            
            # 기간별 통계
            period_stats = df.groupby('sim_period').agg({
                'share_error': ['mean', 'std', lambda x: np.mean(np.abs(x))],
                'share_error_pct': 'mean'
            }).round(4)
            
            results['summary'] = {
                'total_tests': len(df),
                'overall_mae': round(df['share_error'].abs().mean(), 4),
                'overall_mape': round(df['share_error_pct'].mean(), 2),
                'period_stats': period_stats.to_dict()
            }
            
            print(f"\n📊 백테스트 요약")
            print(f"   총 테스트: {results['summary']['total_tests']}개")
            print(f"   평균 절대 오차 (MAE): {results['summary']['overall_mae']:.4f}%p")
            print(f"   평균 백분율 오차 (MAPE): {results['summary']['overall_mape']:.2f}%")
        
        return results


def run_full_validation(full_data: pd.DataFrame, base_month: str = None, sim_period: int = 2):
    """전체 검증 실행"""
    validator = SimulatorCrossValidator(full_data)
    
    if base_month is None:
        base_month = validator.all_months[-1]
    
    print(f"\n{'='*70}")
    print(f"🔍 시뮬레이터 크로스 검증 시작")
    print(f"   기준월: {base_month}")
    print(f"   예측 기간: {sim_period}개월")
    print(f"{'='*70}")
    
    # 1. 크로스 검증
    cross_results = validator.run_cross_validation(base_month, sim_period)
    
    # 2. 백테스트 검증
    backtest_results = validator.run_backtest_validation()
    
    return {
        'cross_validation': cross_results,
        'backtest': backtest_results,
        'validator': validator
    }


if __name__ == "__main__":
    print("시뮬레이터 크로스 검증 모듈")
    print("사용법: from simulator_cross_validator import run_full_validation")
