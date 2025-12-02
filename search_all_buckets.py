"""
모든 버킷에서 xlsx 파일 찾기
"""
import boto3
from config import Config

s3 = boto3.client(
    's3',
    region_name=Config.AWS_REGION,
    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
)

print('🔍 모든 버킷에서 xlsx 파일 찾기\n')

# 버킷 목록 조회
buckets_response = s3.list_buckets()
buckets = [b['Name'] for b in buckets_response['Buckets']]

# 충전 관련 버킷만 필터링
target_buckets = [b for b in buckets if 'eba' in b.lower() or 'chargev' in b.lower() or 'charge' in b.lower()]

print(f'📂 검색 대상 버킷: {len(target_buckets)}개\n')

for bucket_name in target_buckets:
    print(f'📁 버킷: {bucket_name}')
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=100)
        
        if 'Contents' in response:
            xlsx_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.xlsx')]
            if xlsx_files:
                print(f'✅ xlsx 파일 발견: {len(xlsx_files)}개')
                for obj in xlsx_files[:3]:
                    print(f'  - {obj["Key"]}')
                print()
            else:
                print(f'  ℹ️ xlsx 파일 없음\n')
        else:
            print('  ℹ️ 파일 없음\n')
    except Exception as e:
        print(f'  ❌ 접근 불가: {e}\n')
