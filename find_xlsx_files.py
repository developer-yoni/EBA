"""
S3에서 xlsx 파일 찾기
"""
import boto3
from config import Config

s3 = boto3.client(
    's3',
    region_name=Config.AWS_REGION,
    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
)

print('🔍 S3에서 xlsx 파일 찾기\n')

# 여러 프리픽스 시도
prefixes = ['', '충전인프라현황DB/', 'charging-infrastructure/', 'data/']

for prefix in prefixes:
    print(f'📁 프리픽스: "{prefix}"')
    try:
        response = s3.list_objects_v2(
            Bucket=Config.S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' in response:
            xlsx_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.xlsx')]
            if xlsx_files:
                print(f'✅ xlsx 파일 발견: {len(xlsx_files)}개')
                for obj in xlsx_files[:5]:
                    print(f'  - {obj["Key"]}')
            else:
                print(f'  ℹ️ xlsx 파일 없음 (전체 {len(response["Contents"])}개 파일)')
        else:
            print('  ℹ️ 파일 없음')
    except Exception as e:
        print(f'  ❌ 오류: {e}')
    print()
