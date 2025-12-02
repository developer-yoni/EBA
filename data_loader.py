"""
S3에서 충전인프라 데이터 로드 및 파싱
"""
import boto3
import pandas as pd
import io
import re
from datetime import datetime
from config import Config

class ChargingDataLoader:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
        )
        
    def list_available_files(self):
        """S3에서 사용 가능한 파일 목록 조회"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=Config.S3_BUCKET,
                Prefix=Config.S3_PREFIX
            )
            
            files = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.xlsx'):
                    files.append({
                        'key': key,
                        'filename': key.split('/')[-1],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
            
            return sorted(files, key=lambda x: x['last_modified'], reverse=True)
        
        except Exception as e:
            print(f'❌ S3 파일 목록 조회 오류: {e}')
            return []
    
    def download_file(self, s3_key):
        """S3에서 파일 다운로드"""
        try:
            response = self.s3_client.get_object(
                Bucket=Config.S3_BUCKET,
                Key=s3_key
            )
            return io.BytesIO(response['Body'].read())
        
        except Exception as e:
            print(f'❌ S3 파일 다운로드 오류: {e}')
            return None
    
    def parse_snapshot_date_from_filename(self, filename):
        """파일명에서 스냅샷 날짜 추출
        예: 충전인프라 현황_2508.xlsx -> 2025-08 (2025년 8월)
        """
        try:
            # 파일명에서 YYMM 패턴 추출
            pattern = r'_(\d{4})'
            match = re.search(pattern, filename)
            
            if match:
                yymm = match.group(1)
                year = f'20{yymm[:2]}'
                month = yymm[2:4]
                
                # 월말 날짜로 설정 (해당 월의 마지막 날)
                from datetime import datetime
                import calendar
                
                year_int = int(year)
                month_int = int(month)
                last_day = calendar.monthrange(year_int, month_int)[1]
                
                snapshot_date = f'{year}-{month}-{last_day:02d}'
                snapshot_month = f'{year}-{month}'
                
                return snapshot_date, snapshot_month
            
            return None, None
        
        except Exception as e:
            print(f'❌ 파일명 날짜 파싱 오류: {e}')
            return None, None
    
    def parse_snapshot_date(self, excel_file):
        """엑셀 파일에서 스냅샷 날짜 추출 (백업용)"""
        try:
            # 제목 셀 읽기 (0행, 2열)
            df_title = pd.read_excel(
                excel_file, 
                sheet_name='Sheet1',
                header=None,
                nrows=1
            )
            
            title_text = str(df_title.iloc[Config.TITLE_ROW, Config.TITLE_COL])
            
            # 날짜 패턴 추출: YY.MM.DD 형식
            date_pattern = r'(\d{2})\.(\d{2})\.(\d{2})'
            match = re.search(date_pattern, title_text)
            
            if match:
                year = f'20{match.group(1)}'
                month = match.group(2)
                day = match.group(3)
                
                snapshot_date = f'{year}-{month}-{day}'
                snapshot_month = f'{year}-{month}'
                
                return snapshot_date, snapshot_month, title_text
            
            return None, None, title_text
        
        except Exception as e:
            print(f'❌ 날짜 파싱 오류: {e}')
            return None, None, None
    
    def _safe_int(self, value):
        """안전한 정수 변환"""
        try:
            if pd.isna(value):
                return 0
            # 문자열인 경우 숫자가 아니면 0 반환
            if isinstance(value, str):
                # 숫자가 아닌 문자열이면 0 반환
                if not value.replace(',', '').replace('-', '').replace('+', '').replace('.', '').isdigit():
                    return 0
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def extract_summary_data(self, s3_key):
        """엑셀 파일의 K2:P4 범위에서 요약 데이터 추출"""
        try:
            # 파일 다운로드
            excel_file = self.download_file(s3_key)
            if excel_file is None:
                return None
            
            # K1:P4 범위 읽기 (헤더 포함)
            df_summary = pd.read_excel(
                excel_file,
                sheet_name='Sheet1',
                header=None,
                skiprows=0,  # 0행부터 읽기
                nrows=4,     # 4행 읽기 (0, 1, 2, 3행)
                usecols='K:P'  # K~P 컬럼
            )
            
            # 데이터 구조 확인
            print(f'📊 요약 데이터 추출: {df_summary.shape}')
            print(f'📊 요약 데이터 내용:\n{df_summary}')
            
            # 행 인덱스: 0=빈행, 1=헤더, 2=전체CPO, 3=당월증감량
            if len(df_summary) >= 4:
                # 세 번째 행: 전체CPO (인덱스 2)
                total_row = df_summary.iloc[2]
                # 네 번째 행: 당월증감량 (인덱스 3)
                change_row = df_summary.iloc[3]
                
                result = {
                    'total': {
                        'label': str(total_row.iloc[0]) if pd.notna(total_row.iloc[0]) else '전체CPO',
                        'cpos': self._safe_int(total_row.iloc[1]),
                        'stations': self._safe_int(total_row.iloc[2]),
                        'slow_chargers': self._safe_int(total_row.iloc[3]),
                        'fast_chargers': self._safe_int(total_row.iloc[4]),
                        'total_chargers': self._safe_int(total_row.iloc[5])
                    },
                    'change': {
                        'label': str(change_row.iloc[0]) if pd.notna(change_row.iloc[0]) else '당월증감량',
                        'cpos': self._safe_int(change_row.iloc[1]),
                        'stations': self._safe_int(change_row.iloc[2]),
                        'slow_chargers': self._safe_int(change_row.iloc[3]),
                        'fast_chargers': self._safe_int(change_row.iloc[4]),
                        'total_chargers': self._safe_int(change_row.iloc[5])
                    }
                }
                
                print(f'✅ 요약 데이터 추출 완료: {result}')
                return result
            
            return None
            
        except Exception as e:
            print(f'❌ 요약 데이터 추출 오류: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    def load_data(self, s3_key):
        """S3에서 데이터 로드 및 파싱"""
        print(f'📥 데이터 로드 중: {s3_key}')
        
        # 파일명에서 날짜 추출 (우선)
        filename = s3_key.split('/')[-1]
        snapshot_date, snapshot_month = self.parse_snapshot_date_from_filename(filename)
        
        if snapshot_date:
            print(f'📅 스냅샷 날짜: {snapshot_date} (파일명: {filename})')
        else:
            print(f'⚠️ 파일명에서 날짜 추출 실패, 엑셀 내용에서 추출 시도...')
        
        # 파일 다운로드
        excel_file = self.download_file(s3_key)
        if excel_file is None:
            return None
        
        # 파일명에서 날짜 추출 실패 시 엑셀 내용에서 추출
        if not snapshot_date:
            snapshot_date, snapshot_month, title = self.parse_snapshot_date(excel_file)
            print(f'📅 스냅샷 날짜: {snapshot_date} ({title})')
        
        # 데이터 읽기 (헤더는 4번째 인덱스)
        excel_file.seek(0)  # 파일 포인터 리셋
        df = pd.read_excel(
            excel_file,
            sheet_name='Sheet1',
            header=Config.HEADER_ROW
        )
        
        # 컬럼명 변경 (의미있는 이름으로)
        df = df.rename(columns=Config.COLUMN_MAPPING)
        
        # 스냅샷 정보 추가
        df['snapshot_date'] = snapshot_date
        df['snapshot_month'] = snapshot_month
        df['data_source'] = s3_key
        df['filename'] = filename
        
        # 빈 행 제거
        df = df.dropna(how='all')
        
        # CPO명이 있는 행만 유지 (실제 데이터)
        if 'CPO명' in df.columns:
            # 헤더 중복 행 제거 (CPO명이 'CPO'인 행)
            df = df[df['CPO명'].notna() & (df['CPO명'] != 'CPO')]
            
            # 데이터 타입 변환
            numeric_cols = ['순위', '충전소수', '완속충전기', '급속충전기', '총충전기', 
                          '시장점유율', '순위변동', '충전소증감', '완속증감', '급속증감', '총증감']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f'✅ 데이터 로드 완료: {len(df)} 행')
        print(f'📊 컬럼: {list(df.columns)}')
        
        return df
    
    def extract_charger_change_from_excel(self, s3_key):
        """엑셀 파일의 N4, O4에서 완속/급속 충전기 증감값 추출"""
        try:
            # 파일 다운로드
            excel_file = self.download_file(s3_key)
            if excel_file is None:
                return None
            
            # N4, O4 값 읽기 (0-indexed: N=13, O=14, 행4=인덱스3)
            df_change = pd.read_excel(
                excel_file,
                sheet_name='Sheet1',
                header=None,
                skiprows=3,  # 3행 스킵 (0,1,2행)
                nrows=1,     # 1행만 읽기 (4행 = 인덱스3)
                usecols='N:O'  # N~O 컬럼
            )
            
            if len(df_change) > 0:
                slow_change = self._safe_int(df_change.iloc[0, 0])  # N4
                fast_change = self._safe_int(df_change.iloc[0, 1])  # O4
                total_change = slow_change + fast_change
                
                return {
                    'slow_charger_change': slow_change,
                    'fast_charger_change': fast_change,
                    'total_change': total_change
                }
            
            return None
            
        except Exception as e:
            print(f'❌ 충전기 증감값 추출 오류: {e}')
            return None
    
    def get_all_months_charger_changes(self):
        """모든 월의 충전기 증감값을 엑셀 N4, O4에서 추출"""
        files = self.list_available_files()
        result = []
        
        for file_info in files:
            s3_key = file_info['key']
            filename = file_info['filename']
            
            # 파일명에서 월 추출
            snapshot_date, snapshot_month = self.parse_snapshot_date_from_filename(filename)
            
            if snapshot_month:
                # 엑셀에서 증감값 추출
                change_data = self.extract_charger_change_from_excel(s3_key)
                
                if change_data:
                    result.append({
                        'month': snapshot_month,
                        'slow_charger_change': change_data['slow_charger_change'],
                        'fast_charger_change': change_data['fast_charger_change'],
                        'total_change': change_data['total_change']
                    })
                    print(f'📊 {snapshot_month}: 완속 {change_data["slow_charger_change"]:+}, 급속 {change_data["fast_charger_change"]:+}')
        
        # 월 기준 정렬
        result = sorted(result, key=lambda x: x['month'])
        return result
    
    def load_latest(self):
        """가장 최신 파일 로드"""
        files = self.list_available_files()
        if not files:
            print('❌ 사용 가능한 파일이 없습니다.')
            return None
        
        latest_file = files[0]
        print(f'📂 최신 파일: {latest_file["filename"]}')
        
        return self.load_data(latest_file['key'])
    
    def load_multiple(self, months=None):
        """여러 월의 데이터 로드"""
        files = self.list_available_files()
        
        if months:
            # 특정 월만 필터링
            files = [f for f in files if any(m in f['filename'] for m in months)]
        
        all_data = []
        for file_info in files:
            df = self.load_data(file_info['key'])
            if df is not None:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f'\n✅ 총 {len(all_data)}개 파일, {len(combined_df)} 행 로드 완료')
            return combined_df
        
        return None
