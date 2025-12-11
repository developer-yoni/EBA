"""
개선된 ML 예측 모듈 v2 - GS차지비 시장점유율 예측

핵심 개선사항:
1. 전체 시장 추세 + GS차지비 자체 추세를 함께 모델링
2. GS vs 시장 상대 성장률(Relative Growth Rate) 추가
3. 점유율 변화 패턴을 직접 학습 (시장 역학 반영)
4. 앙상블 방식으로 여러 예측 결합
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ImprovedMLPredictorV2:
    """
    개선된 ML 예측기 v2
    
    핵심 원칙:
    1. 전체 시장 추세 (Market Trend)
    2. GS차지비 자체 추세 (GS Trend)  
    3. GS vs 시장 상대 성장률 (Relative Growth)
    4. 점유율 변화 패턴 (Share Dynamics)
    
    이 4가지를 앙상블하여 예측
    """
    
    def __init__(self):
        # 개별 모델들
        self.market_model = None  # 시장 전체 충전기 예측
        self.gs_charger_model = None  # GS 충전기 예측
        self.gs_share_model = None  # GS 점유율 직접 예측
        self.relative_growth_model = None  # 상대 성장률 예측
        
        # 학습 데이터 저장
        self.n_train = 0
        self.last_values = {}
        self.model_weights = {}  # 앙상블 가중치
        
        # 백테스트 결과 저장
        self.backtest_results = []
        
    def fit(
        self, 
        gs_history: List[Dict], 
        market_history: List[Dict]
    ) -> Dict:
        """
        모델 학습 - 4가지 관점에서 분석
        
        Args:
            gs_history: GS차지비 히스토리 [{month, total_chargers, market_share, ...}, ...]
            market_history: 시장 히스토리 [{month, total_chargers, total_cpos}, ...]
            
        Returns:
            학습 결과 및 통계
        """
        n = len(gs_history)
        
        if n < 3:
            return {'error': '데이터 부족 (최소 3개월 필요)', 'n_samples': n}
        
        # 데이터 추출
        gs_chargers = np.array([h['total_chargers'] for h in gs_history])
        gs_shares = np.array([h['market_share'] for h in gs_history])
        market_chargers = np.array([m['total_chargers'] for m in market_history[:n]])
        
        # 시간 인덱스
        X = np.arange(n).reshape(-1, 1)
        
        # ========== 1. 시장 전체 추세 모델 ==========
        self.market_model = HuberRegressor(epsilon=1.35)  # 이상치에 강건
        self.market_model.fit(X, market_chargers)
        market_r2 = self._calculate_r2(X, market_chargers, self.market_model)
        market_slope = self._get_slope(X, market_chargers)
        
        # ========== 2. GS 충전기 추세 모델 ==========
        self.gs_charger_model = HuberRegressor(epsilon=1.35)
        self.gs_charger_model.fit(X, gs_chargers)
        gs_charger_r2 = self._calculate_r2(X, gs_chargers, self.gs_charger_model)
        gs_charger_slope = self._get_slope(X, gs_chargers)
        
        # ========== 3. GS 점유율 직접 예측 모델 ==========
        self.gs_share_model = Ridge(alpha=0.5)  # 약간의 정규화
        self.gs_share_model.fit(X, gs_shares)
        gs_share_r2 = self._calculate_r2(X, gs_shares, self.gs_share_model)
        gs_share_slope = self._get_slope(X, gs_shares)
        
        # ========== 4. 상대 성장률 모델 (핵심 추가) ==========
        # GS 성장률 vs 시장 성장률의 차이를 모델링
        # relative_growth[t] = (gs_chargers[t]/gs_chargers[t-1]) / (market[t]/market[t-1]) - 1
        if n >= 3:
            relative_growth = []
            for i in range(1, n):
                gs_growth = gs_chargers[i] / gs_chargers[i-1] if gs_chargers[i-1] > 0 else 1
                market_growth = market_chargers[i] / market_chargers[i-1] if market_chargers[i-1] > 0 else 1
                rel_growth = (gs_growth / market_growth - 1) * 100  # 퍼센트로 변환
                relative_growth.append(rel_growth)
            
            relative_growth = np.array(relative_growth)
            X_rel = np.arange(len(relative_growth)).reshape(-1, 1)
            
            self.relative_growth_model = Ridge(alpha=1.0)
            self.relative_growth_model.fit(X_rel, relative_growth)
            rel_growth_r2 = self._calculate_r2(X_rel, relative_growth, self.relative_growth_model)
            rel_growth_mean = np.mean(relative_growth)
            rel_growth_std = np.std(relative_growth)
        else:
            rel_growth_r2 = 0
            rel_growth_mean = 0
            rel_growth_std = 0
        
        # ========== 5. 점유율 변화 패턴 분석 ==========
        share_changes = np.diff(gs_shares)
        share_change_mean = np.mean(share_changes) if len(share_changes) > 0 else 0
        share_change_std = np.std(share_changes) if len(share_changes) > 0 else 0
        
        # 최근 추세 vs 전체 추세 비교
        if n >= 4:
            recent_slope = (gs_shares[-1] - gs_shares[-3]) / 2
            trend_consistency = 1.0 if (recent_slope * gs_share_slope) > 0 else 0.5
        else:
            recent_slope = gs_share_slope
            trend_consistency = 0.7
        
        # ========== 6. 앙상블 가중치 결정 ==========
        # R² 기반 가중치 (성능이 좋은 모델에 더 높은 가중치)
        total_r2 = max(0.01, gs_share_r2 + gs_charger_r2 + market_r2)
        
        # 기본 가중치 (ratio 방식 70%, direct 방식 30%)
        # R²가 높으면 해당 방식의 가중치 증가
        ratio_base_weight = 0.7
        direct_base_weight = 0.3
        
        # R² 기반 조정
        if gs_share_r2 > 0.8:  # 점유율 직접 예측이 매우 좋으면
            direct_base_weight = 0.4
            ratio_base_weight = 0.6
        elif gs_share_r2 < 0.5:  # 점유율 직접 예측이 불안정하면
            direct_base_weight = 0.2
            ratio_base_weight = 0.8
        
        self.model_weights = {
            'ratio': ratio_base_weight,
            'direct': direct_base_weight,
            'relative_growth_adjustment': min(0.1, abs(rel_growth_mean) / 10)  # 상대 성장률 조정 비중
        }
        
        # 저장
        self.n_train = n
        self.last_values = {
            'gs_chargers': gs_chargers[-1],
            'market_chargers': market_chargers[-1],
            'gs_share': gs_shares[-1],
            'gs_shares_history': gs_shares.tolist(),
            'market_chargers_history': market_chargers.tolist()
        }
        
        # 통계 계산
        share_mean = np.mean(gs_shares)
        share_std = np.std(gs_shares)
        cv = share_std / share_mean if share_mean > 0 else 1
        
        # 신뢰도 점수 계산 (개선된 공식)
        data_score = min(100, (n / 12) * 100)
        trend_score = gs_share_r2 * trend_consistency * 100
        volatility_score = max(0, (1 - cv * 5)) * 100
        relative_growth_score = max(0, (1 - abs(rel_growth_mean) / 5)) * 100  # 상대 성장률이 안정적일수록 높음
        
        confidence_score = (
            data_score * 0.20 +
            trend_score * 0.30 +
            volatility_score * 0.30 +
            relative_growth_score * 0.20  # 상대 성장률 안정성 추가
        )
        confidence_score = max(0, min(100, confidence_score))
        
        return {
            'n_samples': n,
            'models': {
                'market': {'slope': market_slope, 'r2': market_r2},
                'gs_charger': {'slope': gs_charger_slope, 'r2': gs_charger_r2},
                'gs_share': {'slope': gs_share_slope, 'r2': gs_share_r2},
                'relative_growth': {
                    'mean': rel_growth_mean,
                    'std': rel_growth_std,
                    'r2': rel_growth_r2
                }
            },
            'statistics': {
                'share_mean': share_mean,
                'share_std': share_std,
                'share_change_mean': share_change_mean,
                'share_change_std': share_change_std,
                'cv': cv,
                'trend_consistency': trend_consistency
            },
            'ensemble_weights': self.model_weights,
            'confidence': {
                'score': round(confidence_score, 1),
                'level': 'HIGH' if confidence_score >= 80 else 'MEDIUM' if confidence_score >= 60 else 'LOW',
                'factors': {
                    'data_score': round(data_score, 1),
                    'trend_score': round(trend_score, 1),
                    'volatility_score': round(volatility_score, 1),
                    'relative_growth_score': round(relative_growth_score, 1)
                }
            }
        }
    
    def _calculate_r2(self, X: np.ndarray, y: np.ndarray, model) -> float:
        """R² 계산"""
        try:
            return max(0, model.score(X, y))
        except:
            return 0
    
    def _get_slope(self, X: np.ndarray, y: np.ndarray) -> float:
        """선형 회귀 기울기 계산"""
        lr = LinearRegression()
        lr.fit(X, y)
        return float(lr.coef_[0])
    
    def predict(
        self, 
        months_ahead: int,
        extra_gs_chargers: int = 0
    ) -> List[Dict]:
        """
        미래 예측 - 앙상블 방식
        
        Args:
            months_ahead: 예측 개월 수
            extra_gs_chargers: 추가 설치 충전기 (시나리오용)
                
        Returns:
            월별 예측 결과 리스트
        """
        if self.market_model is None:
            raise ValueError("먼저 fit()을 호출하세요")
        
        predictions = []
        cumulative_extra = 0
        monthly_extra = extra_gs_chargers / months_ahead if months_ahead > 0 else 0
        
        for i in range(1, months_ahead + 1):
            future_idx = self.n_train + i - 1
            X_future = np.array([[future_idx]])
            
            # ========== 방법 1: Ratio 방식 ==========
            # GS충전기와 시장전체 각각 예측 후 점유율 계산
            pred_gs_chargers = self.gs_charger_model.predict(X_future)[0]
            pred_market = self.market_model.predict(X_future)[0]
            
            # 추가 충전기 반영
            cumulative_extra += monthly_extra
            pred_gs_with_extra = pred_gs_chargers + cumulative_extra
            pred_market_with_extra = pred_market + cumulative_extra  # GS가 추가하면 시장도 증가
            
            pred_share_ratio = (pred_gs_with_extra / pred_market_with_extra) * 100 if pred_market_with_extra > 0 else 0
            
            # ========== 방법 2: Direct 방식 ==========
            # 점유율 직접 예측
            pred_share_direct = self.gs_share_model.predict(X_future)[0]
            
            # 추가 충전기 효과 반영 (점유율 증가분 계산)
            if extra_gs_chargers > 0 and pred_market_with_extra > 0:
                # 추가 충전기로 인한 점유율 증가분
                extra_share_effect = (cumulative_extra / pred_market_with_extra) * 100
                # 하지만 시장도 같이 증가하므로 효과 감소
                market_dilution = cumulative_extra / pred_market_with_extra
                net_extra_effect = extra_share_effect * (1 - market_dilution * 0.5)
                pred_share_direct += net_extra_effect
            
            # ========== 방법 3: 상대 성장률 조정 ==========
            # 과거 GS vs 시장 상대 성장률 패턴 반영
            if self.relative_growth_model is not None:
                X_rel = np.array([[self.n_train - 1 + i]])
                pred_rel_growth = self.relative_growth_model.predict(X_rel)[0]
                
                # 상대 성장률이 양수면 GS가 시장보다 빠르게 성장
                # 이를 ratio 예측에 반영
                rel_adjustment = pred_rel_growth / 100  # 퍼센트를 비율로
                pred_share_ratio_adjusted = pred_share_ratio * (1 + rel_adjustment * 0.5)
            else:
                pred_share_ratio_adjusted = pred_share_ratio
            
            # ========== 앙상블 결합 ==========
            w_ratio = self.model_weights.get('ratio', 0.7)
            w_direct = self.model_weights.get('direct', 0.3)
            
            # 가중 평균
            pred_share = (
                pred_share_ratio_adjusted * w_ratio +
                pred_share_direct * w_direct
            )
            
            # 신뢰구간 계산
            uncertainty = 0.1 * i  # 월당 0.1%p 불확실성 증가
            ci_lower = pred_share - 1.96 * uncertainty
            ci_upper = pred_share + 1.96 * uncertainty
            
            predictions.append({
                'months_ahead': i,
                'predicted_gs_chargers': int(pred_gs_with_extra),
                'predicted_market_chargers': int(pred_market_with_extra),
                'predicted_share': round(pred_share, 4),
                'predicted_share_ratio': round(pred_share_ratio_adjusted, 4),
                'predicted_share_direct': round(pred_share_direct, 4),
                'added_chargers': int(cumulative_extra),
                'ci_lower': round(ci_lower, 4),
                'ci_upper': round(ci_upper, 4),
                'method': 'ensemble_v2'
            })
        
        return predictions
    
    def backtest(
        self,
        gs_history: List[Dict],
        market_history: List[Dict],
        test_months: int = 3
    ) -> Dict:
        """
        백테스트 수행
        
        Args:
            gs_history: 전체 GS차지비 히스토리
            market_history: 전체 시장 히스토리
            test_months: 테스트에 사용할 마지막 N개월
            
        Returns:
            백테스트 결과
        """
        n = len(gs_history)
        if n < test_months + 3:
            return {'error': '데이터 부족'}
        
        # 학습/테스트 분리
        train_gs = gs_history[:-test_months]
        train_market = market_history[:-test_months]
        test_gs = gs_history[-test_months:]
        
        # 학습
        fit_result = self.fit(train_gs, train_market)
        if 'error' in fit_result:
            return fit_result
        
        # 예측
        predictions = self.predict(test_months)
        
        # 실제값
        actual_shares = [h['market_share'] for h in test_gs]
        
        # 오차 계산
        errors = []
        for i, (pred, actual) in enumerate(zip(predictions, actual_shares)):
            error = pred['predicted_share'] - actual
            abs_error = abs(error)
            pct_error = abs_error / actual * 100 if actual > 0 else 0
            
            errors.append({
                'month': i + 1,
                'predicted': pred['predicted_share'],
                'actual': actual,
                'error': error,
                'abs_error': abs_error,
                'pct_error': pct_error
            })
        
        # 통계
        mae = np.mean([e['abs_error'] for e in errors])
        mape = np.mean([e['pct_error'] for e in errors])
        rmse = np.sqrt(np.mean([e['error']**2 for e in errors]))
        
        return {
            'test_months': test_months,
            'mae': round(mae, 4),
            'mape': round(mape, 2),
            'rmse': round(rmse, 4),
            'errors': errors,
            'fit_result': fit_result
        }


def compare_v1_vs_v2(full_data: pd.DataFrame) -> Dict:
    """
    기존 방식(v1)과 개선된 방식(v2) 비교
    
    Args:
        full_data: 전체 RAG 데이터
        
    Returns:
        비교 결과
    """
    from ml_predictor import ImprovedMLPredictor
    
    # GS차지비 데이터 추출
    gs_data = full_data[full_data['CPO명'] == 'GS차지비'].copy()
    gs_data = gs_data.sort_values('snapshot_month')
    
    gs_history = []
    for _, row in gs_data.iterrows():
        market_share = row.get('시장점유율', 0)
        if pd.notna(market_share) and market_share < 1:
            market_share = market_share * 100
        
        gs_history.append({
            'month': row.get('snapshot_month'),
            'total_chargers': int(row.get('총충전기', 0)) if pd.notna(row.get('총충전기')) else 0,
            'market_share': round(float(market_share), 4) if pd.notna(market_share) else 0,
            'total_change': int(row.get('총증감', 0)) if pd.notna(row.get('총증감')) else 0
        })
    
    # 시장 히스토리 추출
    all_months = sorted(full_data['snapshot_month'].unique().tolist())
    market_history = []
    for month in all_months:
        month_data = full_data[full_data['snapshot_month'] == month]
        if len(month_data) > 0:
            total_chargers = month_data['총충전기'].sum()
            total_cpos = len(month_data[month_data['총충전기'] > 0])
            market_history.append({
                'month': month,
                'total_chargers': int(total_chargers),
                'total_cpos': int(total_cpos)
            })
    
    results = {
        'total_months': len(gs_history),
        'period': f"{gs_history[0]['month']} ~ {gs_history[-1]['month']}",
        'v1_results': [],
        'v2_results': [],
        'comparison': []
    }
    
    # 다양한 테스트 기간으로 비교
    for test_months in [1, 2, 3, 4, 5, 6]:
        if len(gs_history) >= test_months + 4:
            # V1 (기존)
            predictor_v1 = ImprovedMLPredictor()
            v1_result = predictor_v1.compare_methods(gs_history, market_history, test_months)
            if 'error' not in v1_result:
                results['v1_results'].append({
                    'test_months': test_months,
                    'mae': v1_result['mae_ratio_method'],  # ratio 방식 사용
                    'method': 'v1_ratio'
                })
            
            # V2 (개선)
            predictor_v2 = ImprovedMLPredictorV2()
            v2_result = predictor_v2.backtest(gs_history, market_history, test_months)
            if 'error' not in v2_result:
                results['v2_results'].append({
                    'test_months': test_months,
                    'mae': v2_result['mae'],
                    'mape': v2_result['mape'],
                    'method': 'v2_ensemble'
                })
                
                # 비교
                if 'error' not in v1_result:
                    v1_mae = v1_result['mae_ratio_method']
                    v2_mae = v2_result['mae']
                    improvement = (v1_mae - v2_mae) / v1_mae * 100 if v1_mae > 0 else 0
                    
                    results['comparison'].append({
                        'test_months': test_months,
                        'v1_mae': v1_mae,
                        'v2_mae': v2_mae,
                        'improvement_pct': round(improvement, 2),
                        'better': 'v2' if v2_mae < v1_mae else 'v1'
                    })
    
    # 요약
    if results['comparison']:
        v2_wins = sum(1 for c in results['comparison'] if c['better'] == 'v2')
        avg_improvement = np.mean([c['improvement_pct'] for c in results['comparison']])
        
        results['summary'] = {
            'v2_wins': v2_wins,
            'v1_wins': len(results['comparison']) - v2_wins,
            'avg_improvement_pct': round(avg_improvement, 2),
            'recommendation': 'v2' if v2_wins > len(results['comparison']) / 2 else 'v1'
        }
    
    return results


if __name__ == "__main__":
    print("개선된 ML 예측 모듈 v2")
    print("사용법: from ml_predictor_v2 import ImprovedMLPredictorV2, compare_v1_vs_v2")
