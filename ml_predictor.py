"""
개선된 ML 예측 모듈 - GS차지비 시장점유율 예측

핵심 개선사항:
1. 점유율 = GS충전기 / 시장전체 공식 사용 (직접 예측 대신)
2. 여러 ML 모델 비교 및 최적 모델 자동 선택
3. 시계열 특성을 고려한 예측 (추세 + 계절성)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ImprovedMLPredictor:
    """
    개선된 ML 예측기
    
    핵심 원칙:
    - 점유율을 직접 예측하지 않고, GS충전기와 시장전체를 각각 예측 후 계산
    - 여러 모델을 비교하여 최적 모델 자동 선택
    - 백테스트 기반 신뢰도 평가
    """
    
    # 사용 가능한 모델들
    MODELS = {
        'linear': LinearRegression,
        'ridge': lambda: Ridge(alpha=1.0),
        'ridge_strong': lambda: Ridge(alpha=10.0),
        'lasso': lambda: Lasso(alpha=0.1),
    }
    
    def __init__(self):
        self.best_gs_model = None
        self.best_market_model = None
        self.best_gs_model_name = None
        self.best_market_model_name = None
        self.scaler_gs = StandardScaler()
        self.scaler_market = StandardScaler()
        self.model_scores = {}
        
    def prepare_features(self, n_samples: int, include_poly: bool = False) -> np.ndarray:
        """
        시계열 특성 생성
        
        Args:
            n_samples: 샘플 수
            include_poly: 다항 특성 포함 여부
        """
        # 기본 특성: 시간 인덱스
        X = np.arange(n_samples).reshape(-1, 1)
        
        if include_poly and n_samples >= 6:
            # 2차 다항 특성 추가 (비선형 추세 포착)
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X = poly.fit_transform(X)
        
        return X
    
    def select_best_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        target_name: str = 'target'
    ) -> Tuple[object, str, float]:
        """
        시계열 교차검증으로 최적 모델 선택
        
        Args:
            X: 특성 행렬
            y: 타겟 벡터
            target_name: 타겟 이름 (로깅용)
            
        Returns:
            (best_model, model_name, best_score)
        """
        n_samples = len(y)
        
        # 데이터가 적으면 단순 선형 회귀 사용
        if n_samples < 5:
            model = LinearRegression()
            model.fit(X, y)
            return model, 'linear', 0.0
        
        # 시계열 교차검증 (최소 2개 fold)
        n_splits = min(3, n_samples - 2)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        best_model = None
        best_model_name = None
        best_score = float('inf')
        
        for name, model_class in self.MODELS.items():
            try:
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    if callable(model_class):
                        model = model_class()
                    else:
                        model = model_class()
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    # MAE 사용 (이상치에 덜 민감)
                    mae = mean_absolute_error(y_val, y_pred)
                    scores.append(mae)
                
                avg_score = np.mean(scores)
                self.model_scores[f'{target_name}_{name}'] = avg_score
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_model_name = name
                    
                    # 전체 데이터로 재학습
                    if callable(model_class):
                        best_model = model_class()
                    else:
                        best_model = model_class()
                    best_model.fit(X, y)
                    
            except Exception as e:
                continue
        
        # 모델 선택 실패 시 기본 선형 회귀
        if best_model is None:
            best_model = LinearRegression()
            best_model.fit(X, y)
            best_model_name = 'linear'
            best_score = 0.0
        
        return best_model, best_model_name, best_score
    
    def fit(
        self, 
        gs_history: List[Dict], 
        market_history: List[Dict],
        use_poly: bool = False
    ) -> Dict:
        """
        모델 학습
        
        Args:
            gs_history: GS차지비 히스토리 [{month, total_chargers, market_share, ...}, ...]
            market_history: 시장 히스토리 [{month, total_chargers, total_cpos}, ...]
            use_poly: 다항 특성 사용 여부
            
        Returns:
            학습 결과 및 통계
        """
        n = len(gs_history)
        
        if n < 3:
            return {'error': '데이터 부족 (최소 3개월 필요)', 'n_samples': n}
        
        # 데이터 추출
        gs_chargers = np.array([h['total_chargers'] for h in gs_history])
        gs_shares = np.array([h['market_share'] for h in gs_history])
        market_chargers = np.array([m['total_chargers'] for m in market_history])
        
        # 특성 생성
        X = self.prepare_features(n, include_poly=use_poly and n >= 6)
        
        # GS 충전기 모델 선택 및 학습
        self.best_gs_model, self.best_gs_model_name, gs_cv_score = \
            self.select_best_model(X, gs_chargers, 'gs_chargers')
        
        # 시장 전체 모델 선택 및 학습
        self.best_market_model, self.best_market_model_name, market_cv_score = \
            self.select_best_model(X, market_chargers, 'market_chargers')
        
        # 점유율 직접 예측 모델도 학습 (비교용)
        self.share_model, self.share_model_name, share_cv_score = \
            self.select_best_model(X, gs_shares, 'share_direct')
        
        # 통계 계산
        gs_slope = self._get_slope(X, gs_chargers)
        market_slope = self._get_slope(X, market_chargers)
        share_slope = self._get_slope(X, gs_shares)
        
        # R² 계산
        gs_r2 = self.best_gs_model.score(X, gs_chargers)
        market_r2 = self.best_market_model.score(X, market_chargers)
        share_r2 = self.share_model.score(X, gs_shares)
        
        # 저장
        self.X_train = X
        self.n_train = n
        self.use_poly = use_poly
        self.gs_chargers_last = gs_chargers[-1]
        self.market_chargers_last = market_chargers[-1]
        self.gs_share_last = gs_shares[-1]
        
        return {
            'n_samples': n,
            'gs_model': self.best_gs_model_name,
            'market_model': self.best_market_model_name,
            'share_model': self.share_model_name,
            'gs_cv_mae': gs_cv_score,
            'market_cv_mae': market_cv_score,
            'share_cv_mae': share_cv_score,
            'gs_slope': gs_slope,
            'market_slope': market_slope,
            'share_slope': share_slope,
            'gs_r2': gs_r2,
            'market_r2': market_r2,
            'share_r2': share_r2,
            'model_scores': self.model_scores
        }
    
    def _get_slope(self, X: np.ndarray, y: np.ndarray) -> float:
        """선형 회귀 기울기 계산"""
        lr = LinearRegression()
        lr.fit(X[:, :1] if X.ndim > 1 and X.shape[1] > 1 else X, y)
        return float(lr.coef_[0])
    
    def predict(
        self, 
        months_ahead: int,
        extra_gs_chargers: int = 0,
        method: str = 'ratio'  # 'ratio' or 'direct'
    ) -> List[Dict]:
        """
        미래 예측
        
        Args:
            months_ahead: 예측 개월 수
            extra_gs_chargers: 추가 설치 충전기 (시나리오용)
            method: 예측 방법
                - 'ratio': GS충전기/시장전체 각각 예측 후 점유율 계산 (권장)
                - 'direct': 점유율 직접 예측
                
        Returns:
            월별 예측 결과 리스트
        """
        if self.best_gs_model is None:
            raise ValueError("먼저 fit()을 호출하세요")
        
        predictions = []
        cumulative_extra = 0
        monthly_extra = extra_gs_chargers / months_ahead if months_ahead > 0 else 0
        
        for i in range(1, months_ahead + 1):
            future_idx = self.n_train + i - 1
            
            # 특성 생성
            if self.use_poly and self.n_train >= 6:
                X_future = np.array([[future_idx, future_idx**2]])
            else:
                X_future = np.array([[future_idx]])
            
            if method == 'ratio':
                # 방법 1: GS충전기와 시장전체 각각 예측 후 점유율 계산
                pred_gs = self.best_gs_model.predict(X_future)[0]
                pred_market = self.best_market_model.predict(X_future)[0]
                
                # 추가 충전기 반영 (GS가 추가하면 시장 전체도 증가)
                cumulative_extra += monthly_extra
                pred_gs_with_extra = pred_gs + cumulative_extra
                pred_market_with_extra = pred_market + cumulative_extra
                
                # 점유율 계산
                pred_share = (pred_gs_with_extra / pred_market_with_extra) * 100
                baseline_share = (pred_gs / pred_market) * 100
                
            else:
                # 방법 2: 점유율 직접 예측 (기존 방식)
                pred_share = self.share_model.predict(X_future)[0]
                baseline_share = pred_share
                pred_gs = self.best_gs_model.predict(X_future)[0]
                pred_market = self.best_market_model.predict(X_future)[0]
                pred_gs_with_extra = pred_gs + cumulative_extra
                pred_market_with_extra = pred_market + cumulative_extra
            
            # 신뢰구간 계산 (단순화된 버전)
            # 예측 기간이 길어질수록 불확실성 증가
            uncertainty = 0.1 * i  # 월당 0.1%p 불확실성 증가
            ci_lower = pred_share - 1.96 * uncertainty
            ci_upper = pred_share + 1.96 * uncertainty
            
            predictions.append({
                'months_ahead': i,
                'predicted_gs_chargers': int(pred_gs_with_extra),
                'predicted_market_chargers': int(pred_market_with_extra),
                'predicted_share': round(pred_share, 4),
                'baseline_share': round(baseline_share, 4),
                'added_chargers': int(cumulative_extra),
                'ci_lower': round(ci_lower, 4),
                'ci_upper': round(ci_upper, 4),
                'method': method
            })
        
        return predictions
    
    def compare_methods(
        self, 
        gs_history: List[Dict], 
        market_history: List[Dict],
        test_months: int = 3
    ) -> Dict:
        """
        두 예측 방법 비교 (백테스트)
        
        Args:
            gs_history: 전체 GS차지비 히스토리
            market_history: 전체 시장 히스토리
            test_months: 테스트에 사용할 마지막 N개월
            
        Returns:
            비교 결과
        """
        n = len(gs_history)
        if n < test_months + 3:
            return {'error': '데이터 부족'}
        
        # 학습/테스트 분리
        train_gs = gs_history[:-test_months]
        train_market = market_history[:-test_months]
        test_gs = gs_history[-test_months:]
        
        # 학습
        self.fit(train_gs, train_market)
        
        # 예측
        pred_ratio = self.predict(test_months, method='ratio')
        pred_direct = self.predict(test_months, method='direct')
        
        # 실제값
        actual_shares = [h['market_share'] for h in test_gs]
        
        # 오차 계산
        errors_ratio = []
        errors_direct = []
        
        for i, actual in enumerate(actual_shares):
            err_ratio = abs(pred_ratio[i]['predicted_share'] - actual)
            err_direct = abs(pred_direct[i]['predicted_share'] - actual)
            errors_ratio.append(err_ratio)
            errors_direct.append(err_direct)
        
        mae_ratio = np.mean(errors_ratio)
        mae_direct = np.mean(errors_direct)
        
        return {
            'test_months': test_months,
            'mae_ratio_method': round(mae_ratio, 4),
            'mae_direct_method': round(mae_direct, 4),
            'better_method': 'ratio' if mae_ratio < mae_direct else 'direct',
            'improvement': round(abs(mae_direct - mae_ratio), 4),
            'improvement_pct': round(abs(mae_direct - mae_ratio) / mae_direct * 100, 2) if mae_direct > 0 else 0,
            'details': {
                'ratio_errors': errors_ratio,
                'direct_errors': errors_direct,
                'actual_shares': actual_shares,
                'pred_ratio': [p['predicted_share'] for p in pred_ratio],
                'pred_direct': [p['predicted_share'] for p in pred_direct]
            }
        }


def run_comprehensive_backtest(full_data: pd.DataFrame) -> Dict:
    """
    종합 백테스트 실행
    
    Args:
        full_data: 전체 RAG 데이터
        
    Returns:
        백테스트 결과
    """
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
    
    # 예측기 생성 및 비교
    predictor = ImprovedMLPredictor()
    
    results = {
        'total_months': len(gs_history),
        'period': f"{gs_history[0]['month']} ~ {gs_history[-1]['month']}",
        'comparisons': []
    }
    
    # 다양한 테스트 기간으로 비교
    for test_months in [1, 2, 3, 4, 5, 6]:
        if len(gs_history) >= test_months + 4:
            comparison = predictor.compare_methods(gs_history, market_history, test_months)
            if 'error' not in comparison:
                results['comparisons'].append(comparison)
    
    # 요약 통계
    if results['comparisons']:
        ratio_maes = [c['mae_ratio_method'] for c in results['comparisons']]
        direct_maes = [c['mae_direct_method'] for c in results['comparisons']]
        
        results['summary'] = {
            'avg_mae_ratio': round(np.mean(ratio_maes), 4),
            'avg_mae_direct': round(np.mean(direct_maes), 4),
            'ratio_wins': sum(1 for c in results['comparisons'] if c['better_method'] == 'ratio'),
            'direct_wins': sum(1 for c in results['comparisons'] if c['better_method'] == 'direct'),
            'recommendation': 'ratio' if np.mean(ratio_maes) < np.mean(direct_maes) else 'direct'
        }
    
    return results


if __name__ == "__main__":
    print("개선된 ML 예측 모듈")
    print("사용법: from ml_predictor import ImprovedMLPredictor, run_comprehensive_backtest")
