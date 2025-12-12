"""
Linear Regression 종합 분석 및 시각화

목적:
1. RAG 원본 데이터를 test set으로 사용하여 Linear Regression 평가
2. 오차, 신뢰도 등 ML 지표 계산
3. 실제 데이터 vs Linear Regression 추세 그래프 시각화
4. Linear Regression 함수식 도출
5. 방식 적합성 및 파라미터 적합성 판단

핵심 분석:
- 시뮬레이터에서 사용하는 Ratio 방식: 점유율 = GS충전기 / 시장전체 * 100
- GS 충전기와 시장 전체를 각각 Linear Regression으로 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 모듈 임포트
from data_loader import ChargingDataLoader


class LinearRegressionComprehensiveAnalyzer:
    """Linear Regression 종합 분석기"""
    
    def __init__(self, full_data: pd.DataFrame):
        self.full_data = full_data
        self.all_months = sorted(full_data['snapshot_month'].unique().tolist())
        
        # 데이터 추출
        self.gs_history = self._extract_gs_history()
        self.market_history = self._extract_market_history()
        
        # 모델 저장
        self.lr_gs = None
        self.lr_market = None
        self.lr_share = None
        
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
    
    def fit_linear_regression(self) -> Dict:
        """
        Linear Regression 모델 학습 및 분석
        
        시뮬레이터에서 사용하는 방식:
        1. GS 충전기 수 예측: lr_gs
        2. 시장 전체 충전기 수 예측: lr_market
        3. 점유율 = GS충전기 / 시장전체 * 100 (Ratio 방식)
        """
        n = len(self.gs_history)
        
        # 데이터 준비
        X = np.arange(n).reshape(-1, 1)
        gs_chargers = np.array([h['total_chargers'] for h in self.gs_history])
        market_chargers = np.array([m['total_chargers'] for m in self.market_history])
        gs_shares = np.array([h['market_share'] for h in self.gs_history])
        
        # 1. GS 충전기 Linear Regression
        self.lr_gs = LinearRegression()
        self.lr_gs.fit(X, gs_chargers)
        gs_pred = self.lr_gs.predict(X)
        
        # 2. 시장 전체 Linear Regression
        self.lr_market = LinearRegression()
        self.lr_market.fit(X, market_chargers)
        market_pred = self.lr_market.predict(X)
        
        # 3. 점유율 직접 Linear Regression (비교용)
        self.lr_share = LinearRegression()
        self.lr_share.fit(X, gs_shares)
        share_pred_direct = self.lr_share.predict(X)
        
        # 4. Ratio 방식 점유율 계산
        share_pred_ratio = (gs_pred / market_pred) * 100
        
        # 함수식 도출
        gs_formula = f"GS충전기(t) = {self.lr_gs.coef_[0]:.2f} × t + {self.lr_gs.intercept_:.2f}"
        market_formula = f"시장전체(t) = {self.lr_market.coef_[0]:.2f} × t + {self.lr_market.intercept_:.2f}"
        share_formula = f"점유율(t) = {self.lr_share.coef_[0]:.6f} × t + {self.lr_share.intercept_:.4f}"
        
        # 지표 계산
        results = {
            'n_samples': n,
            'data_period': f"{self.all_months[0]} ~ {self.all_months[-1]}",
            
            # GS 충전기 모델
            'gs_charger_model': {
                'formula': gs_formula,
                'slope': round(self.lr_gs.coef_[0], 2),
                'intercept': round(self.lr_gs.intercept_, 2),
                'r2': round(r2_score(gs_chargers, gs_pred), 4),
                'mae': round(mean_absolute_error(gs_chargers, gs_pred), 2),
                'rmse': round(np.sqrt(mean_squared_error(gs_chargers, gs_pred)), 2),
                'interpretation': f"월평균 {self.lr_gs.coef_[0]:.0f}대 증가"
            },
            
            # 시장 전체 모델
            'market_model': {
                'formula': market_formula,
                'slope': round(self.lr_market.coef_[0], 2),
                'intercept': round(self.lr_market.intercept_, 2),
                'r2': round(r2_score(market_chargers, market_pred), 4),
                'mae': round(mean_absolute_error(market_chargers, market_pred), 2),
                'rmse': round(np.sqrt(mean_squared_error(market_chargers, market_pred)), 2),
                'interpretation': f"월평균 {self.lr_market.coef_[0]:.0f}대 증가"
            },
            
            # 점유율 직접 예측 모델
            'share_direct_model': {
                'formula': share_formula,
                'slope': round(self.lr_share.coef_[0], 6),
                'intercept': round(self.lr_share.intercept_, 4),
                'r2': round(r2_score(gs_shares, share_pred_direct), 4),
                'mae': round(mean_absolute_error(gs_shares, share_pred_direct), 4),
                'rmse': round(np.sqrt(mean_squared_error(gs_shares, share_pred_direct)), 4),
                'mape': round(np.mean(np.abs((gs_shares - share_pred_direct) / gs_shares)) * 100, 2),
                'interpretation': f"월평균 {self.lr_share.coef_[0]*100:.4f}%p 변화"
            },
            
            # Ratio 방식 점유율 (시뮬레이터 사용 방식)
            'share_ratio_model': {
                'formula': "점유율(t) = GS충전기(t) / 시장전체(t) × 100",
                'r2': round(r2_score(gs_shares, share_pred_ratio), 4),
                'mae': round(mean_absolute_error(gs_shares, share_pred_ratio), 4),
                'rmse': round(np.sqrt(mean_squared_error(gs_shares, share_pred_ratio)), 4),
                'mape': round(np.mean(np.abs((gs_shares - share_pred_ratio) / gs_shares)) * 100, 2)
            },
            
            # 원본 데이터
            'actual_data': {
                'gs_chargers': gs_chargers.tolist(),
                'market_chargers': market_chargers.tolist(),
                'gs_shares': gs_shares.tolist(),
                'months': self.all_months
            },
            
            # 예측 데이터
            'predicted_data': {
                'gs_chargers': gs_pred.tolist(),
                'market_chargers': market_pred.tolist(),
                'share_direct': share_pred_direct.tolist(),
                'share_ratio': share_pred_ratio.tolist()
            }
        }
        
        return results
    
    def cross_validation_analysis(self) -> Dict:
        """시계열 교차검증으로 모델 성능 평가"""
        n = len(self.gs_history)
        
        X = np.arange(n).reshape(-1, 1)
        gs_chargers = np.array([h['total_chargers'] for h in self.gs_history])
        market_chargers = np.array([m['total_chargers'] for m in self.market_history])
        gs_shares = np.array([h['market_share'] for h in self.gs_history])
        
        # 시계열 교차검증
        n_splits = min(5, n - 3)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {
            'gs_charger': {'mae': [], 'rmse': [], 'r2': []},
            'market': {'mae': [], 'rmse': [], 'r2': []},
            'share_direct': {'mae': [], 'rmse': [], 'r2': [], 'mape': []},
            'share_ratio': {'mae': [], 'rmse': [], 'r2': [], 'mape': []}
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            
            # GS 충전기
            gs_train, gs_val = gs_chargers[train_idx], gs_chargers[val_idx]
            lr_gs = LinearRegression().fit(X_train, gs_train)
            gs_pred = lr_gs.predict(X_val)
            cv_results['gs_charger']['mae'].append(mean_absolute_error(gs_val, gs_pred))
            cv_results['gs_charger']['rmse'].append(np.sqrt(mean_squared_error(gs_val, gs_pred)))
            cv_results['gs_charger']['r2'].append(r2_score(gs_val, gs_pred) if len(gs_val) > 1 else 0)
            
            # 시장 전체
            market_train, market_val = market_chargers[train_idx], market_chargers[val_idx]
            lr_market = LinearRegression().fit(X_train, market_train)
            market_pred = lr_market.predict(X_val)
            cv_results['market']['mae'].append(mean_absolute_error(market_val, market_pred))
            cv_results['market']['rmse'].append(np.sqrt(mean_squared_error(market_val, market_pred)))
            cv_results['market']['r2'].append(r2_score(market_val, market_pred) if len(market_val) > 1 else 0)
            
            # 점유율 직접
            share_train, share_val = gs_shares[train_idx], gs_shares[val_idx]
            lr_share = LinearRegression().fit(X_train, share_train)
            share_pred_direct = lr_share.predict(X_val)
            cv_results['share_direct']['mae'].append(mean_absolute_error(share_val, share_pred_direct))
            cv_results['share_direct']['rmse'].append(np.sqrt(mean_squared_error(share_val, share_pred_direct)))
            cv_results['share_direct']['r2'].append(r2_score(share_val, share_pred_direct) if len(share_val) > 1 else 0)
            cv_results['share_direct']['mape'].append(np.mean(np.abs((share_val - share_pred_direct) / share_val)) * 100)
            
            # Ratio 방식
            share_pred_ratio = (gs_pred / market_pred) * 100
            cv_results['share_ratio']['mae'].append(mean_absolute_error(share_val, share_pred_ratio))
            cv_results['share_ratio']['rmse'].append(np.sqrt(mean_squared_error(share_val, share_pred_ratio)))
            cv_results['share_ratio']['r2'].append(r2_score(share_val, share_pred_ratio) if len(share_val) > 1 else 0)
            cv_results['share_ratio']['mape'].append(np.mean(np.abs((share_val - share_pred_ratio) / share_val)) * 100)
        
        # 평균 계산
        summary = {}
        for model, metrics in cv_results.items():
            summary[model] = {k: round(np.mean(v), 4) for k, v in metrics.items()}
        
        return {
            'n_splits': n_splits,
            'cv_summary': summary,
            'cv_details': cv_results
        }
    
    def backtest_analysis(self, test_periods: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]) -> Dict:
        """
        다양한 예측 기간에 대한 백테스트
        
        기준월을 변경하며 예측 정확도 측정
        """
        results = {period: [] for period in test_periods}
        
        for period in test_periods:
            # 유효한 기준월 선택 (최소 3개월 학습 + period개월 검증)
            for i in range(3, len(self.all_months) - period):
                base_month = self.all_months[i]
                
                # 학습 데이터 (기준월까지)
                train_gs = self.gs_history[:i+1]
                train_market = self.market_history[:i+1]
                
                # 검증 데이터 (기준월 이후)
                test_gs = self.gs_history[i+1:i+1+period]
                test_market = self.market_history[i+1:i+1+period]
                
                if len(test_gs) < period:
                    continue
                
                # 모델 학습
                n_train = len(train_gs)
                X_train = np.arange(n_train).reshape(-1, 1)
                gs_train = np.array([h['total_chargers'] for h in train_gs])
                market_train = np.array([m['total_chargers'] for m in train_market])
                
                lr_gs = LinearRegression().fit(X_train, gs_train)
                lr_market = LinearRegression().fit(X_train, market_train)
                
                # 예측
                errors = []
                for j in range(period):
                    X_pred = np.array([[n_train + j]])
                    pred_gs = lr_gs.predict(X_pred)[0]
                    pred_market = lr_market.predict(X_pred)[0]
                    pred_share = (pred_gs / pred_market) * 100
                    
                    actual_share = test_gs[j]['market_share']
                    error = abs(pred_share - actual_share)
                    pct_error = (error / actual_share) * 100 if actual_share > 0 else 0
                    
                    errors.append({
                        'predicted': pred_share,
                        'actual': actual_share,
                        'abs_error': error,
                        'pct_error': pct_error
                    })
                
                mae = np.mean([e['abs_error'] for e in errors])
                mape = np.mean([e['pct_error'] for e in errors])
                
                results[period].append({
                    'base_month': base_month,
                    'train_months': n_train,
                    'mae': mae,
                    'mape': mape,
                    'reliability': 100 - mape,
                    'errors': errors
                })
        
        # 요약 통계
        summary = {}
        for period, period_results in results.items():
            if period_results:
                maes = [r['mae'] for r in period_results]
                mapes = [r['mape'] for r in period_results]
                reliabilities = [r['reliability'] for r in period_results]
                
                summary[period] = {
                    'n_tests': len(period_results),
                    'avg_mae': round(np.mean(maes), 4),
                    'std_mae': round(np.std(maes), 4),
                    'avg_mape': round(np.mean(mapes), 2),
                    'std_mape': round(np.std(mapes), 2),
                    'avg_reliability': round(np.mean(reliabilities), 2),
                    'min_reliability': round(min(reliabilities), 2),
                    'max_reliability': round(max(reliabilities), 2)
                }
        
        return {
            'test_periods': test_periods,
            'summary': summary,
            'details': results
        }
    
    def compare_models(self) -> Dict:
        """다양한 회귀 모델 비교 (Linear, Ridge, Lasso)"""
        n = len(self.gs_history)
        X = np.arange(n).reshape(-1, 1)
        gs_shares = np.array([h['market_share'] for h in self.gs_history])
        gs_chargers = np.array([h['total_chargers'] for h in self.gs_history])
        market_chargers = np.array([m['total_chargers'] for m in self.market_history])
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge(alpha=0.1)': Ridge(alpha=0.1),
            'Ridge(alpha=1.0)': Ridge(alpha=1.0),
            'Ridge(alpha=10.0)': Ridge(alpha=10.0),
            'Lasso(alpha=0.01)': Lasso(alpha=0.01),
            'Lasso(alpha=0.1)': Lasso(alpha=0.1)
        }
        
        results = {}
        
        for name, model in models.items():
            # GS 충전기 모델
            model_gs = type(model)(**model.get_params())
            model_gs.fit(X, gs_chargers)
            gs_pred = model_gs.predict(X)
            
            # 시장 전체 모델
            model_market = type(model)(**model.get_params())
            model_market.fit(X, market_chargers)
            market_pred = model_market.predict(X)
            
            # Ratio 방식 점유율
            share_pred = (gs_pred / market_pred) * 100
            
            results[name] = {
                'gs_r2': round(r2_score(gs_chargers, gs_pred), 4),
                'gs_mae': round(mean_absolute_error(gs_chargers, gs_pred), 2),
                'market_r2': round(r2_score(market_chargers, market_pred), 4),
                'market_mae': round(mean_absolute_error(market_chargers, market_pred), 2),
                'share_r2': round(r2_score(gs_shares, share_pred), 4),
                'share_mae': round(mean_absolute_error(gs_shares, share_pred), 4),
                'share_mape': round(np.mean(np.abs((gs_shares - share_pred) / gs_shares)) * 100, 2),
                'gs_slope': round(model_gs.coef_[0], 2),
                'market_slope': round(model_market.coef_[0], 2)
            }
        
        return results

    
    def plot_analysis(self, save_path: str = 'lr_analysis_plots.png'):
        """분석 결과 시각화"""
        # 한글 폰트 설정 시도
        try:
            plt.rcParams['font.family'] = 'AppleGothic'  # macOS
        except:
            try:
                plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
            except:
                pass
        plt.rcParams['axes.unicode_minus'] = False
        
        # 데이터 준비
        n = len(self.gs_history)
        X = np.arange(n)
        gs_chargers = np.array([h['total_chargers'] for h in self.gs_history])
        market_chargers = np.array([m['total_chargers'] for m in self.market_history])
        gs_shares = np.array([h['market_share'] for h in self.gs_history])
        months = [h['month'] for h in self.gs_history]
        
        # 예측값 계산
        X_fit = X.reshape(-1, 1)
        gs_pred = self.lr_gs.predict(X_fit)
        market_pred = self.lr_market.predict(X_fit)
        share_pred_ratio = (gs_pred / market_pred) * 100
        share_pred_direct = self.lr_share.predict(X_fit)
        
        # 미래 예측 (8개월)
        X_future = np.arange(n, n + 8).reshape(-1, 1)
        gs_future = self.lr_gs.predict(X_future)
        market_future = self.lr_market.predict(X_future)
        share_future_ratio = (gs_future / market_future) * 100
        share_future_direct = self.lr_share.predict(X_future)
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. GS 충전기 수 추세
        ax1 = axes[0, 0]
        ax1.scatter(X, gs_chargers, color='blue', label='Actual GS Chargers', s=50, zorder=5)
        ax1.plot(X, gs_pred, 'b--', label=f'Linear Regression (R²={r2_score(gs_chargers, gs_pred):.4f})', linewidth=2)
        ax1.plot(np.arange(n-1, n+8), np.concatenate([[gs_pred[-1]], gs_future.flatten()]), 
                 'b:', label='Future Prediction (8M)', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Month Index')
        ax1.set_ylabel('Number of Chargers')
        ax1.set_title(f'GS Chargers Trend\nFormula: y = {self.lr_gs.coef_[0]:.2f}x + {self.lr_gs.intercept_:.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # X축 레이블 설정
        tick_positions = list(range(0, n, max(1, n//6))) + list(range(n, n+8))
        tick_labels = [months[i] if i < n else f'+{i-n+1}M' for i in tick_positions]
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # 2. 시장 전체 충전기 수 추세
        ax2 = axes[0, 1]
        ax2.scatter(X, market_chargers, color='green', label='Actual Market Total', s=50, zorder=5)
        ax2.plot(X, market_pred, 'g--', label=f'Linear Regression (R²={r2_score(market_chargers, market_pred):.4f})', linewidth=2)
        ax2.plot(np.arange(n-1, n+8), np.concatenate([[market_pred[-1]], market_future.flatten()]), 
                 'g:', label='Future Prediction (8M)', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Month Index')
        ax2.set_ylabel('Number of Chargers')
        ax2.set_title(f'Market Total Chargers Trend\nFormula: y = {self.lr_market.coef_[0]:.2f}x + {self.lr_market.intercept_:.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # 3. 점유율 추세 (Ratio vs Direct 비교)
        ax3 = axes[1, 0]
        ax3.scatter(X, gs_shares, color='red', label='Actual Market Share', s=50, zorder=5)
        ax3.plot(X, share_pred_ratio, 'r--', 
                 label=f'Ratio Method (R²={r2_score(gs_shares, share_pred_ratio):.4f})', linewidth=2)
        ax3.plot(X, share_pred_direct, 'm--', 
                 label=f'Direct Method (R²={r2_score(gs_shares, share_pred_direct):.4f})', linewidth=2, alpha=0.7)
        ax3.plot(np.arange(n-1, n+8), np.concatenate([[share_pred_ratio[-1]], share_future_ratio.flatten()]), 
                 'r:', label='Ratio Future (8M)', linewidth=2, alpha=0.7)
        ax3.plot(np.arange(n-1, n+8), np.concatenate([[share_pred_direct[-1]], share_future_direct.flatten()]), 
                 'm:', label='Direct Future (8M)', linewidth=2, alpha=0.5)
        ax3.set_xlabel('Month Index')
        ax3.set_ylabel('Market Share (%)')
        ax3.set_title('Market Share Trend: Ratio vs Direct Method')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # 4. 예측 오차 분포
        ax4 = axes[1, 1]
        errors_ratio = gs_shares - share_pred_ratio
        errors_direct = gs_shares - share_pred_direct
        
        x_pos = np.arange(n)
        width = 0.35
        ax4.bar(x_pos - width/2, errors_ratio, width, label=f'Ratio Error (MAE={np.mean(np.abs(errors_ratio)):.4f})', color='red', alpha=0.7)
        ax4.bar(x_pos + width/2, errors_direct, width, label=f'Direct Error (MAE={np.mean(np.abs(errors_direct)):.4f})', color='magenta', alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Month Index')
        ax4.set_ylabel('Prediction Error (%p)')
        ax4.set_title('Prediction Error by Month')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 그래프 저장 완료: {save_path}")
        return save_path
    
    def generate_report(self) -> str:
        """종합 분석 리포트 생성"""
        # 모델 학습
        fit_results = self.fit_linear_regression()
        cv_results = self.cross_validation_analysis()
        backtest_results = self.backtest_analysis()
        model_comparison = self.compare_models()
        
        report = []
        report.append("=" * 80)
        report.append("📊 Linear Regression 종합 분석 리포트")
        report.append("=" * 80)
        
        # 1. 데이터 개요
        report.append("\n" + "─" * 80)
        report.append("1. 데이터 개요")
        report.append("─" * 80)
        report.append(f"   데이터 기간: {fit_results['data_period']}")
        report.append(f"   총 데이터 포인트: {fit_results['n_samples']}개월")
        
        # 2. Linear Regression 함수식
        report.append("\n" + "─" * 80)
        report.append("2. Linear Regression 함수식 (시뮬레이터 사용 방식)")
        report.append("─" * 80)
        report.append("\n   [GS 충전기 예측 모델]")
        report.append(f"   {fit_results['gs_charger_model']['formula']}")
        report.append(f"   - 해석: {fit_results['gs_charger_model']['interpretation']}")
        report.append(f"   - R²: {fit_results['gs_charger_model']['r2']}")
        report.append(f"   - MAE: {fit_results['gs_charger_model']['mae']}대")
        
        report.append("\n   [시장 전체 예측 모델]")
        report.append(f"   {fit_results['market_model']['formula']}")
        report.append(f"   - 해석: {fit_results['market_model']['interpretation']}")
        report.append(f"   - R²: {fit_results['market_model']['r2']}")
        report.append(f"   - MAE: {fit_results['market_model']['mae']}대")
        
        report.append("\n   [점유율 계산 (Ratio 방식 - 시뮬레이터 사용)]")
        report.append(f"   {fit_results['share_ratio_model']['formula']}")
        report.append(f"   - R²: {fit_results['share_ratio_model']['r2']}")
        report.append(f"   - MAE: {fit_results['share_ratio_model']['mae']}%p")
        report.append(f"   - MAPE: {fit_results['share_ratio_model']['mape']}%")
        
        report.append("\n   [점유율 직접 예측 (비교용)]")
        report.append(f"   {fit_results['share_direct_model']['formula']}")
        report.append(f"   - R²: {fit_results['share_direct_model']['r2']}")
        report.append(f"   - MAE: {fit_results['share_direct_model']['mae']}%p")
        report.append(f"   - MAPE: {fit_results['share_direct_model']['mape']}%")
        
        # 3. 교차검증 결과
        report.append("\n" + "─" * 80)
        report.append("3. 시계열 교차검증 결과")
        report.append("─" * 80)
        report.append(f"   교차검증 Fold 수: {cv_results['n_splits']}")
        
        cv_summary = cv_results['cv_summary']
        report.append("\n   [점유율 예측 성능 비교]")
        report.append(f"   Ratio 방식 - MAE: {cv_summary['share_ratio']['mae']:.4f}%p, MAPE: {cv_summary['share_ratio']['mape']:.2f}%")
        report.append(f"   Direct 방식 - MAE: {cv_summary['share_direct']['mae']:.4f}%p, MAPE: {cv_summary['share_direct']['mape']:.2f}%")
        
        better_method = "Ratio" if cv_summary['share_ratio']['mae'] < cv_summary['share_direct']['mae'] else "Direct"
        report.append(f"\n   → 교차검증 기준 더 나은 방식: {better_method}")
        
        # 4. 백테스트 결과
        report.append("\n" + "─" * 80)
        report.append("4. 백테스트 결과 (예측 기간별)")
        report.append("─" * 80)
        report.append(f"\n   {'기간':^8} | {'테스트수':^8} | {'평균MAE':^12} | {'평균MAPE':^12} | {'평균신뢰도':^12} | {'신뢰도범위':^15}")
        report.append("   " + "-" * 75)
        
        for period, stats in backtest_results['summary'].items():
            report.append(f"   {period}개월{' '*3} | {stats['n_tests']:^8} | {stats['avg_mae']:^12.4f} | {stats['avg_mape']:^12.2f}% | {stats['avg_reliability']:^12.2f}% | {stats['min_reliability']:.1f}~{stats['max_reliability']:.1f}%")
        
        # 전체 평균
        all_reliabilities = []
        all_mapes = []
        for period, details in backtest_results['details'].items():
            for d in details:
                all_reliabilities.append(d['reliability'])
                all_mapes.append(d['mape'])
        
        if all_reliabilities:
            avg_reliability = np.mean(all_reliabilities)
            avg_mape = np.mean(all_mapes)
            report.append("   " + "-" * 75)
            report.append(f"   {'전체':^8} | {len(all_reliabilities):^8} | {np.mean([d['mae'] for p in backtest_results['details'].values() for d in p]):^12.4f} | {avg_mape:^12.2f}% | {avg_reliability:^12.2f}% | {min(all_reliabilities):.1f}~{max(all_reliabilities):.1f}%")
        
        # 5. 모델 비교
        report.append("\n" + "─" * 80)
        report.append("5. 회귀 모델 비교 (Linear vs Ridge vs Lasso)")
        report.append("─" * 80)
        report.append(f"\n   {'모델':^20} | {'점유율 R²':^12} | {'점유율 MAE':^12} | {'점유율 MAPE':^12}")
        report.append("   " + "-" * 65)
        
        for name, metrics in model_comparison.items():
            report.append(f"   {name:^20} | {metrics['share_r2']:^12.4f} | {metrics['share_mae']:^12.4f} | {metrics['share_mape']:^12.2f}%")
        
        # 최적 모델 찾기
        best_model = min(model_comparison.items(), key=lambda x: x[1]['share_mape'])
        report.append(f"\n   → 최적 모델: {best_model[0]} (MAPE: {best_model[1]['share_mape']:.2f}%)")
        
        # 6. 결론 및 권장사항
        report.append("\n" + "=" * 80)
        report.append("6. 결론 및 권장사항")
        report.append("=" * 80)
        
        # Linear Regression 적합성 판단
        report.append("\n   [Linear Regression 방식 적합성 판단]")
        
        gs_r2 = fit_results['gs_charger_model']['r2']
        market_r2 = fit_results['market_model']['r2']
        share_r2 = fit_results['share_ratio_model']['r2']
        
        if gs_r2 >= 0.95 and market_r2 >= 0.95:
            report.append(f"   ✅ 매우 적합: GS충전기 R²={gs_r2:.4f}, 시장전체 R²={market_r2:.4f}")
            report.append("      → 데이터가 선형 추세를 매우 잘 따르고 있음")
        elif gs_r2 >= 0.85 and market_r2 >= 0.85:
            report.append(f"   ✅ 적합: GS충전기 R²={gs_r2:.4f}, 시장전체 R²={market_r2:.4f}")
            report.append("      → 데이터가 선형 추세를 잘 따르고 있음")
        elif gs_r2 >= 0.7 and market_r2 >= 0.7:
            report.append(f"   ⚠️ 보통: GS충전기 R²={gs_r2:.4f}, 시장전체 R²={market_r2:.4f}")
            report.append("      → 선형 추세가 있으나 변동성 존재, 단기 예측에 적합")
        else:
            report.append(f"   ❌ 부적합: GS충전기 R²={gs_r2:.4f}, 시장전체 R²={market_r2:.4f}")
            report.append("      → 비선형 모델 또는 다른 접근 방식 검토 필요")
        
        # 파라미터 적합성 판단
        report.append("\n   [현재 파라미터 적합성 판단]")
        
        current_model = "LinearRegression"
        current_mape = model_comparison[current_model]['share_mape']
        best_mape = best_model[1]['share_mape']
        
        if current_model == best_model[0]:
            report.append(f"   ✅ 최적: 현재 사용 중인 LinearRegression이 최적 모델")
            report.append(f"      → MAPE: {current_mape:.2f}%")
        else:
            improvement = ((current_mape - best_mape) / current_mape) * 100
            report.append(f"   ⚠️ 개선 가능: {best_model[0]}이 {improvement:.1f}% 더 나은 성능")
            report.append(f"      → 현재 MAPE: {current_mape:.2f}%, 최적 MAPE: {best_mape:.2f}%")
        
        # 예측 기간별 권장사항
        report.append("\n   [예측 기간별 권장사항]")
        for period, stats in backtest_results['summary'].items():
            if stats['avg_reliability'] >= 98:
                status = "✅ 매우 신뢰"
            elif stats['avg_reliability'] >= 95:
                status = "✅ 신뢰"
            elif stats['avg_reliability'] >= 90:
                status = "⚠️ 양호"
            else:
                status = "❌ 주의"
            report.append(f"   {period}개월 예측: {status} (신뢰도 {stats['avg_reliability']:.1f}%, MAPE {stats['avg_mape']:.2f}%)")
        
        # 최종 결론
        report.append("\n   [최종 결론]")
        if avg_reliability >= 95:
            report.append(f"   🎯 Linear Regression (Ratio 방식)은 현재 데이터에 매우 적합합니다.")
            report.append(f"      평균 신뢰도: {avg_reliability:.2f}%, 평균 MAPE: {avg_mape:.2f}%")
        elif avg_reliability >= 90:
            report.append(f"   🎯 Linear Regression (Ratio 방식)은 현재 데이터에 적합합니다.")
            report.append(f"      평균 신뢰도: {avg_reliability:.2f}%, 평균 MAPE: {avg_mape:.2f}%")
            report.append(f"      단, 6개월 이상 장기 예측 시 주의 필요")
        else:
            report.append(f"   ⚠️ Linear Regression의 예측 정확도가 다소 낮습니다.")
            report.append(f"      평균 신뢰도: {avg_reliability:.2f}%, 평균 MAPE: {avg_mape:.2f}%")
            report.append(f"      단기(1-3개월) 예측에만 활용 권장")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("🚀 Linear Regression 종합 분석 시작")
    print("=" * 80)
    
    # 데이터 로드
    print("\n📥 RAG 데이터 로드 중...")
    loader = ChargingDataLoader()
    full_data = loader.load_multiple()
    
    if full_data is None or len(full_data) == 0:
        print("❌ 데이터 로드 실패")
        return
    
    print(f"✅ 데이터 로드 완료: {len(full_data)} 행")
    
    # 분석기 생성
    analyzer = LinearRegressionComprehensiveAnalyzer(full_data)
    
    # 모델 학습
    print("\n📊 Linear Regression 모델 학습 중...")
    fit_results = analyzer.fit_linear_regression()
    
    # 리포트 생성
    print("\n📝 분석 리포트 생성 중...")
    report = analyzer.generate_report()
    print(report)
    
    # 그래프 생성
    print("\n📈 그래프 생성 중...")
    plot_path = analyzer.plot_analysis('lr_analysis_plots.png')
    
    # 결과 저장
    print("\n💾 결과 저장 중...")
    with open('lr_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✅ 리포트 저장 완료: lr_analysis_report.txt")
    
    print("\n" + "=" * 80)
    print("✅ 분석 완료")
    print("=" * 80)
    
    return {
        'fit_results': fit_results,
        'report': report,
        'plot_path': plot_path
    }


if __name__ == "__main__":
    main()
