import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    # AWS 자격 증명
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # AWS 설정
    AWS_REGION = os.getenv('AWS_REGION', 'ap-northeast-2')
    
    # S3 설정
    S3_BUCKET = os.getenv('S3_BUCKET', 's3-eba-team3')
    S3_PREFIX = os.getenv('S3_PREFIX', '충전인프라현황DB/')
    
    # Bedrock 설정
    MODEL_ID = os.getenv('MODEL_ID', 'global.anthropic.claude-sonnet-4-5-20250929-v1:0')
    ANTHROPIC_VERSION = 'bedrock-2023-05-31'
    MAX_TOKENS = 8192
    TEMPERATURE = 0.7
    
    # Knowledge Base 설정
    KNOWLEDGE_BASE_ID = os.getenv('KNOWLEDGE_BASE_ID', 'XHG5MMFIYK')
    KB_NUMBER_OF_RESULTS = 5
    
    # 데이터 설정
    HEADER_ROW = 4  # 0-based index (4번째 행이 헤더)
    TITLE_ROW = 0
    TITLE_COL = 2
    
    # 컬럼 매핑 (실제 의미있는 컬럼명으로 변경)
    COLUMN_MAPPING = {
        'Unnamed: 3': 'CPO명',
        'Unnamed: 4': '순위',
        'Unnamed: 5': '충전소수',
        'Unnamed: 6': '완속충전기',
        'Unnamed: 7': '급속충전기',
        'Unnamed: 8': '총충전기',
        'Unnamed: 9': '시장점유율',
        'Unnamed: 10': '순위변동',
        'Unnamed: 11': '충전소증감',
        'Unnamed: 12': '완속증감',
        'Unnamed: 13': '급속증감',
        'Unnamed: 14': '총증감',
        'CH기준': '특이사항'
    }
