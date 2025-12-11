"""
백테스트 모듈 - ScenarioSimulator 예측 신뢰도 검증

목적:
1. 과거 데이터로 예측 정확도 검증
2. confidence_score와 실제 오차 간 상관관계 분석
3. 예측 기간별 오차 패턴 분석
4. 로직 강화를 위한 인사이트 도출
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class BacktestSimulator:
    """ScenarioSimulator 백테스트 클래스"""
    
    def __init__(self, full_data: pd.DataFrame):
        """
        Args:
            full_data: 전체 RAG 데이터 (모든 월 포함)
        """
        self.full_data = full_data
        self.all_months = sorted(full_data['snapshot_month'].unique().tolist())
        self.backtest_results = None
        self.backtest_stats = None
        
    def get_valid_base_months(self, min_train_months: int = 3, min_eval_months: int = 1) -> List[str]:
        """
        백테스트 가능한 기준월 목록 반환
        
        Args:
            min_train_months: 최소 학습 데이터 개월 수
            min_eval_months: 최소 평가 데이터 개월 수
        """
        valid_months = []
        for i, month in enumerate(self.all_months):
            train_count = i + 1  # 해당 월까지의 데이터 수
            eval_count = len(self.all_months) - i - 1  # 이후 데이터 수
            
            if train_count >= min_train_months and eval_count >= min_eval_months:
                valid_months.append(month)
        
        return valid_months
    
    def split_data(self, base_month: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        기준월 기준으로 학습/평가 데이터 분리 (미래 정보 누출 방지)
        
        Args:
            base_month: 기준월 (YYYY-MM)
            
        Returns:
            (train_data, eval_data) 튜플
        """
        train_data = self.full_data[self.full_data['snapshot_month'] <= base_month].copy()
        eval_data = self.full_data[self.full_data['snapshot_month'] > base_month].copy()
        
        return train_data, eval_data
    
    def extract_gs_history(self, data: pd.DataFrame) -> List[Dict]:
        """GS차지비 히스토리 추출"""
        gs_data = data[data['CPO명'] == 'GS차지비'].copy()
        gs_history = gs_data.sort_values('snapshot_month')
        
        history = []
        for _, row in gs_history.iterrows():
            market_share = row.get('시장점유율', 0)
            if pd.notna(market_share) and market_share < 1:
                market_share = market_share * 100
            
            history.append({
                'month': row.get('snapshot_month'),
                'rank': int(row.get('순위', 0)) if pd.notna(row.get('순위')) else None,
                'total_chargers': int(row.get('총충전기', 0)) if pd.notna(row.get('총충전기')) else 0,
                'market_share': round(float(market_share), 4) if pd.notna(market_share) else 0,
                'total_change': int(row.get('총증감', 0)) if pd.notna(row.get('총증감')) else 0
            })
        
        return history
    
    def extract_market_history(self, data: pd.DataFrame) -> List[Dict]:
        """시장 히스토리 추출"""
        all_months = sorted(data['snapshot_month'].unique().tolist())
        
        market_history = []
        for month in all_months:
            month_data = data[data['snapshot_month'] == month]
            if len(month_data) > 0:
                total_chargers = month_data['총충전기'].sum()
                total_cpos = len(month_data[month_data['총충전기'] > 0])
                market_history.append({
                    'month': month,
                    'total_chargers': int(total_chargers),
                    'total_cpos': int(total_cpos)
                })
        
        return market_history
    
    def perform_ml_analysis(self, gs_history: List[Dict], market_history: List[Dict]) -> Dict:
        """ML 기반 분석 수행 (ScenarioSimulator와 동일 로직)"""
        n = len(gs_history)
        if n < 3:
            return {'error': '데이터 부족', 'confidence': {'score': 0, 'level': 'LOW'}}
        
        months_idx = np.arange(n).reshape(-1, 1)
        gs_shares = np.array([h['market_share'] for h in gs_history])
        gs_chargers = np.array([h['total_chargers'] for h in gs_history])
        market_chargers = np.array([m['total_chargers'] for m in market_history])
        
        # 선형 회귀 - 시장점유율
        lr_share = LinearRegression()
        lr_share.fit(months_idx, gs_shares)
        share_slope = lr_share.coef_[0]
        share_intercept = lr_share.intercept_
        share_r2 = lr_share.score(months_idx, gs_shares)
        
        # 선형 회귀 - 충전기
        lr_chargers = LinearRegression()
        lr_chargers.fit(months_idx, gs_chargers)
        charger_slope = lr_chargers.coef_[0]
        charger_r2 = lr_chargers.score(months_idx, gs_chargers)
        
        # 시장 전체
        lr_market = LinearRegression()
        lr_market.fit(months_idx, market_chargers)
        market_slope = lr_market.coef_[0]
        
        # 통계
        share_mean = np.mean(gs_shares)
        share_std = np.std(gs_shares)
        
        # 신뢰도 점수
        data_consistency = max(0, 1 - share_std / share_mean) if share_mean > 0 else 0
        confidence_score = (share_r2 * 0.4 + charger_r2 * 0.3 + data_consistency * 0.3) * 100
        confidence_score = max(0, min(100, confidence_score))
        
        # 미래 예측 (선형 회귀 기반)
        predictions = []
        for i in range(1, 13):
            future_idx = n + i - 1
            pred_share = share_intercept + share_slope * future_idx
            pred_chargers = lr_chargers.intercept_ + charger_slope * future_idx
            pred_market = lr_market.intercept_ + market_slope * future_idx
            
            predictions.append({
                'months_ahead': i,
                'predicted_share': round(pred_share, 4),
                'predicted_chargers': int(pred_chargers),
                'predicted_market': int(pred_market)
            })
        
        return {
            'linear_regression': {
                'share_slope': share_slope,
                'share_intercept': share_intercept,
                'share_r2': share_r2,
                'charger_slope': charger_slope,
                'charger_r2': charger_r2,
                'market_slope': market_slope
            },
            'statistics': {
                'share_mean': share_mean,
                'share_std': share_std,
                'n_samples': n
            },
            'confidence': {
                'score': round(confidence_score, 2),
                'level': 'HIGH' if confidence_score >= 70 else 'MEDIUM' if confidence_score >= 50 else 'LOW'
            },
            'predictions': predictions
        }
    
    def get_actual_values(self, eval_data: pd.DataFrame, base_month: str, sim_period: int) -> List[Dict]:
        """평가 데이터에서 실제값 추출"""
        base_date = datetime.strptime(base_month, '%Y-%m')
        
        actual_values = []
        for i in range(1, sim_period + 1):
            target_date = base_date + relativedelta(months=i)
            target_month = target_date.strftime('%Y-%m')
            
            gs_row = eval_data[(eval_data['snapshot_month'] == target_month) & 
                               (eval_data['CPO명'] == 'GS차지비')]
            
            if len(gs_row) > 0:
                market_share = gs_row.iloc[0].get('시장점유율', 0)
                if pd.notna(market_share) and market_share < 1:
                    market_share = market_share * 100
                
                actual_values.append({
                    'month': target_month,
                    'months_ahead': i,
                    'actual_share': round(float(market_share), 4) if pd.notna(market_share) else None,
                    'actual_chargers': int(gs_row.iloc[0].get('총충전기', 0))
                })
            else:
                actual_values.append({
                    'month': target_month,
                    'months_ahead': i,
                    'actual_share': None,
                    'actual_chargers': None
                })
        
        return actual_values
    
    def calculate_metrics(self, predictions: List[Dict], actuals: List[Dict]) -> Dict:
        """예측 vs 실제 비교 지표 계산"""
        pred_shares = []
        actual_shares = []
        
        for pred, actual in zip(predictions, actuals):
            if actual['actual_share'] is not None:
                pred_shares.append(pred['predicted_share'])
                actual_shares.append(actual['actual_share'])
        
        if len(pred_shares) == 0:
            return {
                'mae': None, 'rmse': None, 'mape': None,
                'max_abs_error': None, 'direction_accuracy': None,
                'n_valid': 0
            }
        
        pred_arr = np.array(pred_shares)
        actual_arr = np.array(actual_shares)
        errors = pred_arr - actual_arr
        abs_errors = np.abs(errors)
        
        # MAE
        mae = np.mean(abs_errors)
        
        # RMSE
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # MAPE (0 나누기 방지)
        mape_values = []
        for p, a in zip(pred_shares, actual_shares):
            if a != 0:
                mape_values.append(abs(p - a) / abs(a) * 100)
        mape = np.mean(mape_values) if mape_values else None
        
        # 최대 절대 오차
        max_abs_error = np.max(abs_errors)
        
        # 방향 정확도 (상승/하락 방향 일치)
        if len(pred_shares) >= 2:
            pred_directions = np.diff(pred_arr) > 0
            actual_directions = np.diff(actual_arr) > 0
            direction_accuracy = np.mean(pred_directions == actual_directions) * 100
        else:
            direction_accuracy = None
        
        return {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2) if mape else None,
            'max_abs_error': round(max_abs_error, 4),
            'direction_accuracy': round(direction_accuracy, 1) if direction_accuracy else None,
            'n_valid': len(pred_shares)
        }

    
    def run_single_backtest(self, base_month: str, sim_period: int) -> Dict:
        """단일 백테스트 실행"""
        # 데이터 분리
        train_data, eval_data = self.split_data(base_month)
        
        # 히스토리 추출
        gs_history = self.extract_gs_history(train_data)
        market_history = self.extract_market_history(train_data)
        
        if len(gs_history) < 3:
            return {
                'base_month': base_month,
                'sim_period_months': sim_period,
                'error': '학습 데이터 부족'
            }
        
        # ML 분석
        ml_analysis = self.perform_ml_analysis(gs_history, market_history)
        
        if 'error' in ml_analysis:
            return {
                'base_month': base_month,
                'sim_period_months': sim_period,
                'error': ml_analysis['error']
            }
        
        # 예측값 추출
        predictions = ml_analysis['predictions'][:sim_period]
        
        # 실제값 추출
        actuals = self.get_actual_values(eval_data, base_month, sim_period)
        
        # 지표 계산
        metrics = self.calculate_metrics(predictions, actuals)
        
        return {
            'base_month': base_month,
            'sim_period_months': sim_period,
            'confidence_score': ml_analysis['confidence']['score'],
            'confidence_level': ml_analysis['confidence']['level'],
            'share_r2': ml_analysis['linear_regression']['share_r2'],
            'share_slope': ml_analysis['linear_regression']['share_slope'],
            'n_train_months': len(gs_history),
            **metrics
        }
    
    def run_backtest(
        self,
        base_months: List[str] = None,
        sim_period_candidates: List[int] = [1, 3, 6]
    ) -> pd.DataFrame:
        """
        전체 백테스트 실행
        
        Args:
            base_months: 테스트할 기준월 리스트 (None이면 자동 선택)
            sim_period_candidates: 예측 기간 후보 리스트
            
        Returns:
            백테스트 결과 DataFrame
        """
        # 기준월 자동 선택
        if base_months is None:
            max_sim_period = max(sim_period_candidates)
            base_months = self.get_valid_base_months(
                min_train_months=3,
                min_eval_months=max_sim_period
            )
        
        results = []
        total_tests = len(base_months) * len(sim_period_candidates)
        
        print(f"\n{'='*60}")
        print(f"📊 백테스트 시작")
        print(f"   - 기준월: {len(base_months)}개 ({base_months[0]} ~ {base_months[-1]})")
        print(f"   - 예측 기간: {sim_period_candidates}")
        print(f"   - 총 테스트: {total_tests}개")
        print(f"{'='*60}\n")
        
        for i, base_month in enumerate(base_months):
            for sim_period in sim_period_candidates:
                # 해당 기간만큼 미래 데이터가 있는지 확인
                base_idx = self.all_months.index(base_month)
                if base_idx + sim_period >= len(self.all_months):
                    continue
                
                result = self.run_single_backtest(base_month, sim_period)
                results.append(result)
        
        self.backtest_results = pd.DataFrame(results)
        
        # 유효한 결과만 필터링
        valid_results = self.backtest_results[
            self.backtest_results['mae'].notna()
        ].copy()
        
        print(f"✅ 백테스트 완료: {len(valid_results)}개 유효 결과")
        
        return self.backtest_results
    
    def analyze_results(self) -> Dict:
        """백테스트 결과 분석"""
        if self.backtest_results is None:
            raise ValueError("먼저 run_backtest()를 실행하세요")
        
        df = self.backtest_results[self.backtest_results['mae'].notna()].copy()
        
        if len(df) == 0:
            return {'error': '유효한 결과 없음'}
        
        analysis = {}
        
        # 1. Confidence Level별 분석
        print(f"\n{'='*60}")
        print("📈 1. Confidence Level별 오차 분석")
        print(f"{'='*60}")
        
        level_stats = df.groupby('confidence_level').agg({
            'mae': ['mean', 'std', 'count'],
            'rmse': 'mean',
            'mape': 'mean',
            'max_abs_error': 'mean'
        }).round(4)
        
        print(level_stats)
        analysis['level_stats'] = level_stats
        
        # HIGH가 LOW보다 오차가 작은지 확인
        level_mae = df.groupby('confidence_level')['mae'].mean()
        if 'HIGH' in level_mae.index and 'LOW' in level_mae.index:
            high_better = level_mae['HIGH'] < level_mae['LOW']
            print(f"\n✓ HIGH가 LOW보다 오차가 {'작음 ✅' if high_better else '크거나 같음 ⚠️'}")
            print(f"  - HIGH MAE: {level_mae.get('HIGH', 'N/A'):.4f}")
            print(f"  - MEDIUM MAE: {level_mae.get('MEDIUM', 'N/A'):.4f}")
            print(f"  - LOW MAE: {level_mae.get('LOW', 'N/A'):.4f}")
        
        # 2. Confidence Score와 MAPE 상관관계
        print(f"\n{'='*60}")
        print("📈 2. Confidence Score vs 오차 상관관계")
        print(f"{'='*60}")
        
        valid_mape = df[df['mape'].notna()]
        if len(valid_mape) >= 3:
            corr_mape = valid_mape['confidence_score'].corr(valid_mape['mape'])
            corr_mae = valid_mape['confidence_score'].corr(valid_mape['mae'])
            
            print(f"  - confidence_score vs MAPE 상관계수: {corr_mape:.4f}")
            print(f"  - confidence_score vs MAE 상관계수: {corr_mae:.4f}")
            print(f"  → {'음의 상관관계 (점수↑ → 오차↓) ✅' if corr_mape < -0.1 else '상관관계 약함 ⚠️'}")
            
            analysis['correlation'] = {
                'score_vs_mape': round(corr_mape, 4),
                'score_vs_mae': round(corr_mae, 4)
            }
        
        # 3. 예측 기간별 오차
        print(f"\n{'='*60}")
        print("📈 3. 예측 기간별 오차 분석")
        print(f"{'='*60}")
        
        period_stats = df.groupby('sim_period_months').agg({
            'mae': ['mean', 'std'],
            'mape': 'mean',
            'rmse': 'mean',
            'direction_accuracy': 'mean'
        }).round(4)
        
        print(period_stats)
        analysis['period_stats'] = period_stats
        
        # 기간별 오차 증가율
        period_mae = df.groupby('sim_period_months')['mae'].mean()
        if len(period_mae) >= 2:
            periods = sorted(period_mae.index)
            print(f"\n  기간별 MAE 증가:")
            for i in range(1, len(periods)):
                prev, curr = periods[i-1], periods[i]
                increase = (period_mae[curr] - period_mae[prev]) / period_mae[prev] * 100
                print(f"    {prev}개월 → {curr}개월: {increase:+.1f}%")
        
        # 4. 권장 최대 예측 기간 계산
        print(f"\n{'='*60}")
        print("📈 4. 권장 최대 예측 기간")
        print(f"{'='*60}")
        
        # MAPE 5% 이하인 최대 기간
        mape_threshold = 5.0
        recommended_max_period = 1
        
        for period in sorted(df['sim_period_months'].unique()):
            period_mape = df[df['sim_period_months'] == period]['mape'].mean()
            if pd.notna(period_mape) and period_mape <= mape_threshold:
                recommended_max_period = period
        
        print(f"  - MAPE {mape_threshold}% 이하 기준 권장 최대 기간: {recommended_max_period}개월")
        analysis['recommended_max_period'] = recommended_max_period
        
        # 5. Confidence Level 경계 재검토
        print(f"\n{'='*60}")
        print("📈 5. Confidence Level 경계 재검토")
        print(f"{'='*60}")
        
        scores = df['confidence_score'].dropna()
        if len(scores) >= 3:
            p70 = np.percentile(scores, 70)
            p30 = np.percentile(scores, 30)
            
            print(f"  현재 기준: HIGH >= 70, MEDIUM >= 50, LOW < 50")
            print(f"  데이터 기반 제안 (상위 30% / 중간 40% / 하위 30%):")
            print(f"    - HIGH >= {p70:.1f}")
            print(f"    - MEDIUM >= {p30:.1f}")
            print(f"    - LOW < {p30:.1f}")
            
            analysis['suggested_thresholds'] = {
                'high': round(p70, 1),
                'medium': round(p30, 1)
            }
        
        self.backtest_stats = analysis
        return analysis
    
    def get_backtest_summary(self, sim_period: int) -> Dict:
        """특정 예측 기간에 대한 백테스트 요약 (API 응답용)"""
        if self.backtest_results is None:
            return None
        
        df = self.backtest_results[
            (self.backtest_results['sim_period_months'] == sim_period) &
            (self.backtest_results['mae'].notna())
        ]
        
        if len(df) == 0:
            return None
        
        return {
            'sim_period_months': sim_period,
            'n_tests': len(df),
            'avg_mae': round(df['mae'].mean(), 4),
            'avg_mape': round(df['mape'].mean(), 2) if df['mape'].notna().any() else None,
            'avg_rmse': round(df['rmse'].mean(), 4),
            'avg_direction_accuracy': round(df['direction_accuracy'].mean(), 1) if df['direction_accuracy'].notna().any() else None,
            'confidence_correlation': self.backtest_stats.get('correlation', {}).get('score_vs_mape') if self.backtest_stats else None
        }


def load_all_data_from_s3(data_loader) -> pd.DataFrame:
    """S3에서 모든 월별 데이터 로드 (data_loader 인스턴스 사용)"""
    return data_loader.load_all_months()


def run_full_backtest(full_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    전체 백테스트 실행 및 분석
    
    Returns:
        (결과 DataFrame, 분석 결과 Dict)
    """
    backtester = BacktestSimulator(full_data)
    
    # 백테스트 실행
    results = backtester.run_backtest(
        sim_period_candidates=[1, 2, 3, 6]
    )
    
    # 결과 분석
    analysis = backtester.analyze_results()
    
    return results, analysis, backtester


if __name__ == "__main__":
    # 테스트용 실행
    print("백테스트 모듈 로드 완료")
    print("사용법: from backtest_simulator import BacktestSimulator, run_full_backtest")
