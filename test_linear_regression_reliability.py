"""
Linear Regression (Ratio 기반) 예측 모델 신뢰도 검증

목적:
- RAG 데이터(2024-12 ~ 2025-11)를 활용한 백테스트
- 다양한 테스트 셋으로 예측 정확도 검증
- 실제 신뢰도(%) 계산

테스트 방식:
1. 기준월을 변경하며 다양한 테스트 셋 생성
2. 기준월까지의 데이터로 학습 → 이후 데이터로 검증
3. 예측값 vs 실제값 비교하여 오차 계산
4. 평균 신뢰도 산출
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Tuple
import sys

# 프로젝트 모듈 임포트
from data_loader import ChargingDataLoader
from config import Config


class LinearRegressionReliabilityTester:
    """Linear Regression (Ratio 기반) 예측 신뢰도 테스터"""
    
    def __init__(self, full_data: pd.DataFrame):
        self.full_data = full_data
        self.all_months = sorted(full_data['snapshot_month'].unique().tolist())
        self.test_results = []
        
    def get_data_range(self) -> Dict:
        """데이터 범위 확인"""
        return {
            'earliest': self.all_months[0],
            'latest': self.all_months[-1],
            'total_months': len(self.all_months),
            'all_months': self.all_months
        }
    
    def extract_gs_history(self, data: pd.DataFrame) -> List[Dict]:
        """GS차지비 히스토리 추출"""
        gs_data = data[data['CPO명'] == 'GS차지비'].copy()
        gs_data = gs_data.sort_values('snapshot_month')
        
        history = []
        for _, row in gs_data.iterrows():
            market_share = row.get('시장점유율', 0)
            if pd.notna(market_share) and market_share < 1:
                market_share = market_share * 100
            
            history.append({
                'month': row.get('snapshot_month'),
                'total_chargers': int(row.get('총충전기', 0)) if pd.notna(row.get('총충전기')) else 0,
                'market_share': round(float(market_share), 4) if pd.notna(market_share) else 0
            })
        
        return history
    
    def extract_market_history(self, data: pd.DataFrame) -> List[Dict]:
        """시장 전체 히스토리 추출"""
        all_months = sorted(data['snapshot_month'].unique().tolist())
        
        market_history = []
        for month in all_months:
            month_data = data[data['snapshot_month'] == month]
            if len(month_data) > 0:
                total_chargers = month_data['총충전기'].sum()
                market_history.append({
                    'month': month,
                    'total_chargers': int(total_chargers)
                })
        
        return market_history
    
    def predict_with_linear_regression_ratio(
        self, 
        gs_history: List[Dict], 
        market_history: List[Dict],
        months_ahead: int
    ) -> List[Dict]:
        """
        Linear Regression (Ratio 기반) 예측
        
        핵심 로직:
        1. GS 충전기 수를 Linear Regression으로 예측
        2. 시장 전체 충전기 수를 Linear Regression으로 예측
        3. 점유율 = (예측 GS 충전기 / 예측 시장 전체) * 100
        """
        n = len(gs_history)
        if n < 3:
            return []
        
        # 데이터 준비
        months_idx = np.arange(n).reshape(-1, 1)
        gs_chargers = np.array([h['total_chargers'] for h in gs_history])
        market_chargers = np.array([m['total_chargers'] for m in market_history])
        
        # Linear Regression 모델 학습
        lr_gs = LinearRegression()
        lr_gs.fit(months_idx, gs_chargers)
        
        lr_market = LinearRegression()
        lr_market.fit(months_idx, market_chargers)
        
        # 미래 예측
        predictions = []
        for i in range(1, months_ahead + 1):
            future_idx = np.array([[n + i - 1]])
            
            # GS 충전기와 시장 전체 각각 예측
            pred_gs = lr_gs.predict(future_idx)[0]
            pred_market = lr_market.predict(future_idx)[0]
            
            # Ratio 방식: 점유율 = GS충전기 / 시장전체 * 100
            pred_share = (pred_gs / pred_market) * 100 if pred_market > 0 else 0
            
            predictions.append({
                'months_ahead': i,
                'predicted_gs_chargers': int(pred_gs),
                'predicted_market_chargers': int(pred_market),
                'predicted_share': round(pred_share, 4)
            })
        
        return predictions
    
    def get_actual_values(self, base_month: str, months_ahead: int) -> List[Dict]:
        """실제값 추출 (기준월 이후)"""
        base_date = datetime.strptime(base_month, '%Y-%m')
        
        actual_values = []
        for i in range(1, months_ahead + 1):
            target_date = base_date + relativedelta(months=i)
            target_month = target_date.strftime('%Y-%m')
            
            gs_row = self.full_data[
                (self.full_data['snapshot_month'] == target_month) & 
                (self.full_data['CPO명'] == 'GS차지비')
            ]
            
            if len(gs_row) > 0:
                market_share = gs_row.iloc[0].get('시장점유율', 0)
                if pd.notna(market_share) and market_share < 1:
                    market_share = market_share * 100
                
                # 시장 전체 충전기
                month_data = self.full_data[self.full_data['snapshot_month'] == target_month]
                market_total = month_data['총충전기'].sum()
                
                actual_values.append({
                    'month': target_month,
                    'months_ahead': i,
                    'actual_share': round(float(market_share), 4) if pd.notna(market_share) else None,
                    'actual_gs_chargers': int(gs_row.iloc[0].get('총충전기', 0)),
                    'actual_market_chargers': int(market_total)
                })
            else:
                actual_values.append({
                    'month': target_month,
                    'months_ahead': i,
                    'actual_share': None,
                    'actual_gs_chargers': None,
                    'actual_market_chargers': None
                })
        
        return actual_values
    
    def calculate_errors(self, predictions: List[Dict], actuals: List[Dict]) -> Dict:
        """예측 오차 계산"""
        errors = []
        details = []
        
        for pred, actual in zip(predictions, actuals):
            if actual['actual_share'] is not None:
                error = pred['predicted_share'] - actual['actual_share']
                abs_error = abs(error)
                pct_error = (abs_error / actual['actual_share']) * 100 if actual['actual_share'] > 0 else 0
                
                errors.append({
                    'month': actual['month'],
                    'months_ahead': pred['months_ahead'],
                    'predicted': pred['predicted_share'],
                    'actual': actual['actual_share'],
                    'error': round(error, 4),
                    'abs_error': round(abs_error, 4),
                    'pct_error': round(pct_error, 2)
                })
                
                details.append({
                    'month': actual['month'],
                    'predicted_share': pred['predicted_share'],
                    'actual_share': actual['actual_share'],
                    'error': round(error, 4),
                    'predicted_gs': pred['predicted_gs_chargers'],
                    'actual_gs': actual['actual_gs_chargers'],
                    'predicted_market': pred['predicted_market_chargers'],
                    'actual_market': actual['actual_market_chargers']
                })
        
        if not errors:
            return {'valid': False}
        
        abs_errors = [e['abs_error'] for e in errors]
        pct_errors = [e['pct_error'] for e in errors]
        
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean([e**2 for e in abs_errors]))
        mape = np.mean(pct_errors)
        max_error = max(abs_errors)
        
        # 신뢰도 계산 (100 - MAPE)
        reliability = max(0, 100 - mape)
        
        return {
            'valid': True,
            'n_predictions': len(errors),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2),
            'max_error': round(max_error, 4),
            'reliability': round(reliability, 2),
            'errors': errors,
            'details': details
        }
    
    def run_single_test(self, base_month: str, prediction_months: int) -> Dict:
        """단일 테스트 실행"""
        # 기준월까지의 데이터만 사용 (미래 정보 누출 방지)
        train_data = self.full_data[self.full_data['snapshot_month'] <= base_month].copy()
        
        # 히스토리 추출
        gs_history = self.extract_gs_history(train_data)
        market_history = self.extract_market_history(train_data)
        
        if len(gs_history) < 3:
            return {
                'base_month': base_month,
                'prediction_months': prediction_months,
                'error': '학습 데이터 부족 (최소 3개월 필요)'
            }
        
        # 예측 수행
        predictions = self.predict_with_linear_regression_ratio(
            gs_history, market_history, prediction_months
        )
        
        # 실제값 추출
        actuals = self.get_actual_values(base_month, prediction_months)
        
        # 오차 계산
        error_stats = self.calculate_errors(predictions, actuals)
        
        return {
            'base_month': base_month,
            'prediction_months': prediction_months,
            'train_months': len(gs_history),
            'last_train_share': gs_history[-1]['market_share'],
            **error_stats
        }
    
    def run_comprehensive_test(
        self, 
        prediction_periods: List[int] = [1, 2, 3, 4, 5, 6],
        min_train_months: int = 3
    ) -> Dict:
        """
        종합 테스트 실행
        
        다양한 기준월과 예측 기간으로 테스트 수행
        """
        print("\n" + "="*70)
        print("📊 Linear Regression (Ratio 기반) 예측 신뢰도 검증")
        print("="*70)
        
        data_range = self.get_data_range()
        print(f"\n📅 RAG 데이터 범위: {data_range['earliest']} ~ {data_range['latest']}")
        print(f"   총 {data_range['total_months']}개월 데이터")
        
        all_results = []
        
        # 각 예측 기간별로 테스트
        for pred_months in prediction_periods:
            print(f"\n{'─'*70}")
            print(f"🔍 {pred_months}개월 예측 테스트")
            print(f"{'─'*70}")
            
            # 유효한 기준월 선택
            # 조건: 최소 학습 데이터 + 예측 기간만큼의 검증 데이터 필요
            valid_base_months = []
            for i, month in enumerate(self.all_months):
                train_count = i + 1
                eval_count = len(self.all_months) - i - 1
                
                if train_count >= min_train_months and eval_count >= pred_months:
                    valid_base_months.append(month)
            
            if not valid_base_months:
                print(f"   ⚠️ 유효한 기준월 없음")
                continue
            
            print(f"   테스트 기준월: {valid_base_months[0]} ~ {valid_base_months[-1]} ({len(valid_base_months)}개)")
            
            period_results = []
            for base_month in valid_base_months:
                result = self.run_single_test(base_month, pred_months)
                if result.get('valid', False):
                    period_results.append(result)
                    all_results.append(result)
            
            # 기간별 통계
            if period_results:
                maes = [r['mae'] for r in period_results]
                mapes = [r['mape'] for r in period_results]
                reliabilities = [r['reliability'] for r in period_results]
                
                print(f"\n   📈 {pred_months}개월 예측 결과 ({len(period_results)}개 테스트):")
                print(f"      MAE  (평균): {np.mean(maes):.4f}%p (범위: {min(maes):.4f} ~ {max(maes):.4f})")
                print(f"      MAPE (평균): {np.mean(mapes):.2f}% (범위: {min(mapes):.2f} ~ {max(mapes):.2f})")
                print(f"      신뢰도 (평균): {np.mean(reliabilities):.2f}% (범위: {min(reliabilities):.2f} ~ {max(reliabilities):.2f})")
        
        # 전체 요약
        print("\n" + "="*70)
        print("📊 전체 테스트 요약")
        print("="*70)
        
        if all_results:
            # 예측 기간별 요약
            summary_by_period = {}
            for pred_months in prediction_periods:
                period_results = [r for r in all_results if r['prediction_months'] == pred_months]
                if period_results:
                    summary_by_period[pred_months] = {
                        'n_tests': len(period_results),
                        'avg_mae': round(np.mean([r['mae'] for r in period_results]), 4),
                        'avg_mape': round(np.mean([r['mape'] for r in period_results]), 2),
                        'avg_reliability': round(np.mean([r['reliability'] for r in period_results]), 2),
                        'min_reliability': round(min([r['reliability'] for r in period_results]), 2),
                        'max_reliability': round(max([r['reliability'] for r in period_results]), 2)
                    }
            
            print("\n예측 기간별 신뢰도:")
            print("-" * 70)
            print(f"{'기간':^8} | {'테스트수':^8} | {'평균MAE':^10} | {'평균MAPE':^10} | {'평균신뢰도':^12} | {'신뢰도범위':^15}")
            print("-" * 70)
            
            for period, stats in summary_by_period.items():
                print(f"{period}개월{' '*3} | {stats['n_tests']:^8} | {stats['avg_mae']:^10.4f} | {stats['avg_mape']:^10.2f}% | {stats['avg_reliability']:^12.2f}% | {stats['min_reliability']:.1f}~{stats['max_reliability']:.1f}%")
            
            # 전체 평균
            all_maes = [r['mae'] for r in all_results]
            all_mapes = [r['mape'] for r in all_results]
            all_reliabilities = [r['reliability'] for r in all_results]
            
            print("-" * 70)
            print(f"{'전체':^8} | {len(all_results):^8} | {np.mean(all_maes):^10.4f} | {np.mean(all_mapes):^10.2f}% | {np.mean(all_reliabilities):^12.2f}% | {min(all_reliabilities):.1f}~{max(all_reliabilities):.1f}%")
            print("-" * 70)
            
            # 결론
            print("\n" + "="*70)
            print("📋 결론")
            print("="*70)
            
            avg_reliability = np.mean(all_reliabilities)
            print(f"\n✅ Linear Regression (Ratio 기반) 예측 모델의 평균 신뢰도: {avg_reliability:.2f}%")
            print(f"   - 평균 MAPE: {np.mean(all_mapes):.2f}% (예측값과 실제값의 평균 오차율)")
            print(f"   - 평균 MAE: {np.mean(all_maes):.4f}%p (점유율 예측 평균 절대 오차)")
            
            if avg_reliability >= 98:
                print(f"\n🎯 신뢰도 평가: 매우 높음 (98% 이상)")
                print(f"   → 예측 결과를 높은 신뢰도로 활용 가능")
            elif avg_reliability >= 95:
                print(f"\n🎯 신뢰도 평가: 높음 (95~98%)")
                print(f"   → 예측 결과를 신뢰할 수 있음")
            elif avg_reliability >= 90:
                print(f"\n🎯 신뢰도 평가: 양호 (90~95%)")
                print(f"   → 참고용으로 활용 가능, 중요 의사결정 시 추가 검토 권장")
            else:
                print(f"\n🎯 신뢰도 평가: 보통 (90% 미만)")
                print(f"   → 참고용으로만 활용, 예측 기간 단축 권장")
            
            # 기간별 권장사항
            print("\n📌 예측 기간별 권장사항:")
            for period, stats in summary_by_period.items():
                if stats['avg_reliability'] >= 98:
                    status = "✅ 매우 신뢰"
                elif stats['avg_reliability'] >= 95:
                    status = "✅ 신뢰"
                elif stats['avg_reliability'] >= 90:
                    status = "⚠️ 양호"
                else:
                    status = "❌ 주의"
                print(f"   {period}개월 예측: {status} (신뢰도 {stats['avg_reliability']:.1f}%)")
            
            return {
                'total_tests': len(all_results),
                'summary_by_period': summary_by_period,
                'overall': {
                    'avg_mae': round(np.mean(all_maes), 4),
                    'avg_mape': round(np.mean(all_mapes), 2),
                    'avg_reliability': round(avg_reliability, 2),
                    'min_reliability': round(min(all_reliabilities), 2),
                    'max_reliability': round(max(all_reliabilities), 2)
                },
                'all_results': all_results
            }
        
        return {'error': '유효한 테스트 결과 없음'}
    
    def print_detailed_results(self, results: Dict):
        """상세 결과 출력"""
        if 'all_results' not in results:
            return
        
        print("\n" + "="*70)
        print("📋 상세 테스트 결과")
        print("="*70)
        
        for result in results['all_results']:
            if not result.get('valid'):
                continue
            
            print(f"\n기준월: {result['base_month']} → {result['prediction_months']}개월 예측")
            print(f"  학습 데이터: {result['train_months']}개월")
            print(f"  MAE: {result['mae']:.4f}%p, MAPE: {result['mape']:.2f}%, 신뢰도: {result['reliability']:.2f}%")
            
            if 'details' in result:
                print("  상세:")
                for d in result['details']:
                    error_sign = "+" if d['error'] > 0 else ""
                    print(f"    {d['month']}: 예측 {d['predicted_share']:.2f}% vs 실제 {d['actual_share']:.2f}% (오차: {error_sign}{d['error']:.4f}%p)")


def main():
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("🚀 Linear Regression (Ratio 기반) 예측 모델 신뢰도 검증 시작")
    print("="*70)
    
    # 데이터 로드
    print("\n📥 RAG 데이터 로드 중...")
    loader = ChargingDataLoader()
    full_data = loader.load_multiple()
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    print(f"✅ 데이터 로드 완료: {len(full_data)} 행")
    
    # 테스터 생성 및 실행
    tester = LinearRegressionReliabilityTester(full_data)
    
    # 종합 테스트 실행
    results = tester.run_comprehensive_test(
        prediction_periods=[1, 2, 3, 4, 5, 6],
        min_train_months=3
    )
    
    # 상세 결과 출력 (선택적)
    if '--detail' in sys.argv:
        tester.print_detailed_results(results)
    
    print("\n" + "="*70)
    print("✅ 테스트 완료")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
