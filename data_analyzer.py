"""
충전 인프라 데이터 분석
"""
import pandas as pd
import numpy as np

class ChargingDataAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def get_summary_stats(self):
        """전체 요약 통계"""
        # 컬럼명은 실제 데이터 구조에 맞게 조정 필요
        # 일반적인 충전 인프라 데이터 구조 가정
        
        summary = {
            'total_records': int(len(self.df)),
            'snapshot_dates': [str(d) for d in self.df['snapshot_date'].unique()] if 'snapshot_date' in self.df.columns else [],
            'columns': [str(col) for col in self.df.columns],
            'data_types': {str(k): str(v) for k, v in self.df.dtypes.to_dict().items()}
        }
        
        return summary
    
    def get_summary_table(self):
        """핵심 요약 테이블 데이터"""
        if len(self.df) == 0:
            return None
        
        # 전체 CPO 통계
        total_cpos = int(len(self.df))
        total_stations = int(self.df['충전소수'].sum()) if '충전소수' in self.df.columns else 0
        total_slow = int(self.df['완속충전기'].sum()) if '완속충전기' in self.df.columns else 0
        total_fast = int(self.df['급속충전기'].sum()) if '급속충전기' in self.df.columns else 0
        total_chargers = int(self.df['총충전기'].sum()) if '총충전기' in self.df.columns else 0
        
        # 당월 증감량
        change_cpos = int(self.df['순위변동'].sum()) if '순위변동' in self.df.columns else 0
        change_stations = int(self.df['충전소증감'].sum()) if '충전소증감' in self.df.columns else 0
        change_slow = int(self.df['완속증감'].sum()) if '완속증감' in self.df.columns else 0
        change_fast = int(self.df['급속증감'].sum()) if '급속증감' in self.df.columns else 0
        change_total = int(self.df['총증감'].sum()) if '총증감' in self.df.columns else 0
        
        return {
            'total': {
                'cpos': total_cpos,
                'stations': total_stations,
                'slow_chargers': total_slow,
                'fast_chargers': total_fast,
                'total_chargers': total_chargers
            },
            'change': {
                'cpos': change_cpos,
                'stations': change_stations,
                'slow_chargers': change_slow,
                'fast_chargers': change_fast,
                'total_chargers': change_total
            }
        }
    
    def analyze_by_cpo(self):
        """CPO(충전사업자)별 분석"""
        # CPO 컬럼명 찾기
        cpo_col = self._find_column(['CPO명', 'CPO', '사업자', '충전사업자', 'operator'])
        
        if not cpo_col:
            return None
        
        # 충전소수와 총충전기 정보 포함
        charger_col = self._find_column(['총충전기', 'TTL', '총', 'total'])
        station_col = self._find_column(['충전소수', '충전소', 'station'])
        
        if charger_col and station_col:
            analysis = self.df.groupby(cpo_col).agg({
                station_col: 'sum',
                charger_col: 'sum'
            }).reset_index()
            analysis.columns = ['CPO명', '충전소수', '총충전기']
            analysis = analysis.sort_values('총충전기', ascending=False).head(20)
        else:
            analysis = self.df.groupby(cpo_col).size().reset_index(name='count')
        
        # JSON 직렬화 가능하도록 변환
        return {
            'data': analysis.to_dict('records'),
            'summary': f'{len(analysis)}개 사업자',
            'total_cpos': int(self.df[cpo_col].nunique())
        }
    
    def analyze_by_region(self):
        """지역별 분석"""
        region_col = self._find_column(['지역', 'region', '시도', '광역시도'])
        
        if not region_col:
            return None
        
        analysis = self.df.groupby(region_col).size().reset_index(name='count')
        
        # JSON 직렬화 가능하도록 변환
        return {
            'data': analysis.to_dict('records'),
            'summary': f'{len(analysis)}개 지역'
        }
    
    def analyze_charger_types(self):
        """충전기 유형별 분석"""
        # 급속/완속 등 충전기 유형 분석
        type_cols = [col for col in self.df.columns if any(
            keyword in str(col).lower() for keyword in ['급속', '완속', 'fast', 'slow', 'type']
        )]
        
        if not type_cols:
            return None
        
        analysis = {}
        for col in type_cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                analysis[str(col)] = {
                    'total': float(self.df[col].sum()),
                    'mean': float(self.df[col].mean()),
                    'median': float(self.df[col].median()),
                    'std': float(self.df[col].std())
                }
        
        return analysis
    
    def trend_analysis(self):
        """시계열 트렌드 분석"""
        if 'snapshot_month' not in self.df.columns:
            return None
        
        # 월별 집계
        monthly = self.df.groupby('snapshot_month').size().reset_index(name='count')
        monthly.columns = ['month', 'count']
        
        # JSON 직렬화 가능하도록 변환
        return {
            'data': monthly.to_dict('records'),
            'summary': f'{len(monthly)}개월 데이터'
        }
    
    def top_performers(self, n=10):
        """상위 N개 사업자"""
        cpo_col = self._find_column(['CPO명', 'CPO', '사업자', '충전사업자'])
        charger_col = self._find_column(['총충전기', 'TTL', '총', 'total'])
        
        if not cpo_col:
            return None
        
        if charger_col:
            # 총충전기 수 기준 상위 N개
            top_df = self.df.nlargest(n, charger_col)[[cpo_col, charger_col]]
            return {
                'ranking': [
                    {
                        'cpo': str(row[cpo_col]),
                        'chargers': int(row[charger_col]) if pd.notna(row[charger_col]) else 0
                    }
                    for _, row in top_df.iterrows()
                ]
            }
        else:
            # 빈도 기준
            top = self.df[cpo_col].value_counts().head(n)
            return {str(k): int(v) for k, v in top.to_dict().items()}
    
    def _find_column(self, keywords):
        """키워드로 컬럼 찾기"""
        for col in self.df.columns:
            col_lower = str(col).lower()
            for keyword in keywords:
                if keyword.lower() in col_lower:
                    return col
        return None
    
    def get_recent_6months_trend(self, target_month=None, start_month=None, end_month=None, excel_changes=None):
        """선택 기간 충전기 증감량 추이 (완속/급속 - 엑셀 N4, O4 기반)"""
        # 엑셀에서 직접 추출한 데이터가 있으면 사용
        if excel_changes:
            # 기간 필터링
            filtered = excel_changes
            if start_month and end_month:
                filtered = [x for x in excel_changes if start_month <= x['month'] <= end_month]
            elif target_month:
                filtered = [x for x in excel_changes if x['month'] <= target_month]
                filtered = sorted(filtered, key=lambda x: x['month'], reverse=True)[:6]
            
            filtered = sorted(filtered, key=lambda x: x['month'])
            
            return {
                'months': [x['month'] for x in filtered],
                'slow_charger_change': [x['slow_charger_change'] for x in filtered],
                'fast_charger_change': [x['fast_charger_change'] for x in filtered]
            }
        
        # 기존 방식 (DataFrame 기반) - 폴백
        if 'snapshot_month' not in self.df.columns:
            return None
        
        # 월별 집계
        monthly = self.df.groupby('snapshot_month').agg({
            '완속증감': 'sum',
            '급속증감': 'sum'
        }).reset_index()
        
        # 기간 필터링
        if start_month and end_month:
            monthly = monthly[(monthly['snapshot_month'] >= start_month) & 
                            (monthly['snapshot_month'] <= end_month)]
        elif target_month:
            monthly = monthly[monthly['snapshot_month'] <= target_month]
            monthly = monthly.sort_values('snapshot_month', ascending=False).head(6)
        
        monthly = monthly.sort_values('snapshot_month')  # 오름차순 정렬
        
        return {
            'months': monthly['snapshot_month'].tolist(),
            'slow_charger_change': monthly['완속증감'].tolist(),
            'fast_charger_change': monthly['급속증감'].tolist()
        }
    
    def get_gs_chargebee_trend(self, target_month=None, start_month=None, end_month=None):
        """GS차지비 선택 기간 충전기 증감량 추이 (완속/급속만)"""
        if 'snapshot_month' not in self.df.columns or 'CPO명' not in self.df.columns:
            return None
        
        # GS차지비 데이터만 필터링
        gs_data = self.df[self.df['CPO명'] == 'GS차지비']
        
        if len(gs_data) == 0:
            return None
        
        # 기간 필터링
        if start_month and end_month:
            gs_data = gs_data[(gs_data['snapshot_month'] >= start_month) & 
                            (gs_data['snapshot_month'] <= end_month)]
        elif target_month:
            gs_data = gs_data[gs_data['snapshot_month'] <= target_month]
        
        # 월별 집계
        monthly = gs_data.groupby('snapshot_month').agg({
            '완속증감': 'sum',
            '급속증감': 'sum'
        }).reset_index()
        
        if start_month and end_month:
            pass  # 이미 필터링됨
        else:
            monthly = monthly.sort_values('snapshot_month', ascending=False).head(6)
        
        monthly = monthly.sort_values('snapshot_month')
        
        return {
            'months': monthly['snapshot_month'].tolist(),
            'slow_charger_change': monthly['완속증감'].tolist(),
            'fast_charger_change': monthly['급속증감'].tolist()
        }
    
    def get_top5_market_share_trend(self, target_month=None, start_month=None, end_month=None):
        """상위 5개사 시장점유율 변화 추이 (선택 기간)"""
        if 'snapshot_month' not in self.df.columns or 'CPO명' not in self.df.columns:
            return None
        
        # 기준월 설정 (종료월 기준으로 상위 5개사 선정)
        reference_month = end_month if end_month else (target_month if target_month else self.df['snapshot_month'].max())
        
        # 기준월 데이터로 상위 5개사 찾기
        reference_data = self.df[self.df['snapshot_month'] == reference_month]
        if len(reference_data) == 0:
            return None
        
        top5_cpos = reference_data.nlargest(5, '총충전기')['CPO명'].tolist()
        
        # 기간 필터링
        if start_month and end_month:
            filtered_df = self.df[(self.df['snapshot_month'] >= start_month) & 
                                 (self.df['snapshot_month'] <= end_month)]
            unique_months = sorted(filtered_df['snapshot_month'].unique())
        else:
            filtered_df = self.df[self.df['snapshot_month'] <= reference_month]
            unique_months = sorted(filtered_df['snapshot_month'].unique(), reverse=True)[:6]
            unique_months = sorted(unique_months)
        
        # 상위 5개사의 월별 시장점유율
        result = {'months': unique_months, 'cpos': {}}
        
        for cpo in top5_cpos:
            cpo_data = filtered_df[filtered_df['CPO명'] == cpo]
            monthly_share = []
            
            for month in unique_months:
                month_data = cpo_data[cpo_data['snapshot_month'] == month]
                if len(month_data) > 0:
                    share = float(month_data['시장점유율'].iloc[0]) * 100  # 퍼센트로 변환
                    monthly_share.append(share)
                else:
                    monthly_share.append(0)
            
            result['cpos'][cpo] = monthly_share
        
        return result
    
    def get_cumulative_chargers_trend(self, target_month=None, start_month=None, end_month=None):
        """선택 기간 완속/급속 충전기 운영 대수 (월별 막대그래프용)"""
        if 'snapshot_month' not in self.df.columns:
            return None
        
        # 월별 집계
        monthly = self.df.groupby('snapshot_month').agg({
            '완속충전기': 'sum',
            '급속충전기': 'sum',
            '총충전기': 'sum'
        }).reset_index()
        
        # 기간 필터링
        if start_month and end_month:
            monthly = monthly[(monthly['snapshot_month'] >= start_month) & 
                            (monthly['snapshot_month'] <= end_month)]
        elif target_month:
            monthly = monthly[monthly['snapshot_month'] <= target_month]
            monthly = monthly.sort_values('snapshot_month', ascending=False).head(6)
        
        monthly = monthly.sort_values('snapshot_month')  # 오름차순 정렬
        
        return {
            'months': monthly['snapshot_month'].tolist(),
            'slow_chargers': monthly['완속충전기'].tolist(),
            'fast_chargers': monthly['급속충전기'].tolist(),
            'total_chargers': monthly['총충전기'].tolist()
        }
    
    def get_period_summary(self, start_month, end_month):
        """선택 기간의 요약 데이터 (첫 행: 종료월 기준 전체, 두 번째 행: 전월 대비 증감량, 세 번째 행: 기간 증감량)"""
        if 'snapshot_month' not in self.df.columns:
            return None
        
        # 종료월 데이터 (전체 현황)
        end_data = self.df[self.df['snapshot_month'] == end_month]
        if len(end_data) == 0:
            return None
        
        # 시작월 데이터 (증감량 계산용)
        start_data = self.df[self.df['snapshot_month'] == start_month]
        
        # 종료월 기준 전체 현황
        total = {
            'cpos': int(len(end_data)),
            'stations': int(end_data['충전소수'].sum()) if '충전소수' in end_data.columns else 0,
            'slow_chargers': int(end_data['완속충전기'].sum()) if '완속충전기' in end_data.columns else 0,
            'fast_chargers': int(end_data['급속충전기'].sum()) if '급속충전기' in end_data.columns else 0,
            'total_chargers': int(end_data['총충전기'].sum()) if '총충전기' in end_data.columns else 0
        }
        
        # 전월 대비 증감량 계산 (종료월 기준)
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        
        end_date = datetime.strptime(end_month, '%Y-%m')
        prev_date = end_date - relativedelta(months=1)
        prev_month = prev_date.strftime('%Y-%m')
        
        prev_data = self.df[self.df['snapshot_month'] == prev_month]
        monthly_change = None
        
        if len(prev_data) > 0:
            prev_total = {
                'cpos': int(len(prev_data)),
                'stations': int(prev_data['충전소수'].sum()) if '충전소수' in prev_data.columns else 0,
                'slow_chargers': int(prev_data['완속충전기'].sum()) if '완속충전기' in prev_data.columns else 0,
                'fast_chargers': int(prev_data['급속충전기'].sum()) if '급속충전기' in prev_data.columns else 0,
                'total_chargers': int(prev_data['총충전기'].sum()) if '총충전기' in prev_data.columns else 0
            }
            monthly_change = {
                'cpos': total['cpos'] - prev_total['cpos'],
                'stations': total['stations'] - prev_total['stations'],
                'slow_chargers': total['slow_chargers'] - prev_total['slow_chargers'],
                'fast_chargers': total['fast_chargers'] - prev_total['fast_chargers'],
                'total_chargers': total['total_chargers'] - prev_total['total_chargers'],
                'prev_month': prev_month,
                'current_month': end_month
            }
        
        # 기간 증감량 계산 (종료월 - 시작월)
        if len(start_data) > 0:
            start_total = {
                'cpos': int(len(start_data)),
                'stations': int(start_data['충전소수'].sum()) if '충전소수' in start_data.columns else 0,
                'slow_chargers': int(start_data['완속충전기'].sum()) if '완속충전기' in start_data.columns else 0,
                'fast_chargers': int(start_data['급속충전기'].sum()) if '급속충전기' in start_data.columns else 0,
                'total_chargers': int(start_data['총충전기'].sum()) if '총충전기' in start_data.columns else 0
            }
            change = {
                'cpos': total['cpos'] - start_total['cpos'],
                'stations': total['stations'] - start_total['stations'],
                'slow_chargers': total['slow_chargers'] - start_total['slow_chargers'],
                'fast_chargers': total['fast_chargers'] - start_total['fast_chargers'],
                'total_chargers': total['total_chargers'] - start_total['total_chargers']
            }
        else:
            # 시작월 데이터가 없으면 종료월의 당월 증감량 사용
            change = {
                'cpos': 0,
                'stations': int(end_data['충전소증감'].sum()) if '충전소증감' in end_data.columns else 0,
                'slow_chargers': int(end_data['완속증감'].sum()) if '완속증감' in end_data.columns else 0,
                'fast_chargers': int(end_data['급속증감'].sum()) if '급속증감' in end_data.columns else 0,
                'total_chargers': int(end_data['총증감'].sum()) if '총증감' in end_data.columns else 0
            }
        
        return {
            'total': total,
            'monthly_change': monthly_change,
            'change': change,
            'start_month': start_month,
            'end_month': end_month
        }
    
    def generate_insights(self):
        """전체 인사이트 생성"""
        insights = {
            'summary': self.get_summary_stats(),
            'cpo_analysis': self.analyze_by_cpo(),
            'charger_types': self.analyze_charger_types(),
            'trend': self.trend_analysis(),
            'top_performers': self.top_performers(),
            'recent_6months_trend': self.get_recent_6months_trend(),
            'gs_chargebee_trend': self.get_gs_chargebee_trend(),
            'top5_market_share_trend': self.get_top5_market_share_trend(),
            'cumulative_chargers_trend': self.get_cumulative_chargers_trend()
        }
        
        return insights
