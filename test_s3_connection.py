"""
S3 연결 테스트
"""
import boto3
from config import Config

print('🔍 S3 연결 테스트\n')
print(f'AWS Region: {Config.AWS_REGION}')
print(f'S3 Bucket: {Config.S3_BUCKET}')
print(f'S3 Prefix: {Config.S3_PREFIX}')
print(f'Access Key ID: {Config.AWS_ACCESS_KEY_ID[:10]}...\n')

try:
    s3_client = boto3.client(
        's3',
        region_name=Config.AWS_REGION,
        aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
    )
    
    print('✅ S3 클라이언트 생성 성공\n')
    
    # 버킷 목록 조회
    print('📂 버킷 목록 조회 중...')
    buckets = s3_client.list_buckets()
    print(f'✅ 총 {len(buckets["Buckets"])}개 버킷:')
    for bucket in buckets['Buckets']:
        print(f'  - {bucket["Name"]}')
    
    print(f'\n📁 {Config.S3_BUCKET} 버킷의 파일 목록 조회 중...')
    
    # 프리픽스 없이 전체 조회
    response = s3_client.list_objects_v2(
        Bucket=Config.S3_BUCKET,
        MaxKeys=10
    )
    
    if 'Contents' in response:
        print(f'✅ 파일 발견 (최대 10개):')
        for obj in response['Contents']:
            print(f'  - {obj["Key"]}')
    else:
        print('❌ 파일이 없습니다.')
    
    # 프리픽스로 조회
    print(f'\n📁 프리픽스 "{Config.S3_PREFIX}" 로 조회 중...')
    response = s3_client.list_objects_v2(
        Bucket=Config.S3_BUCKET,
        Prefix=Config.S3_PREFIX
    )
    
    if 'Contents' in response:
        print(f'✅ 파일 발견: {len(response["Contents"])}개')
        for obj in response['Contents'][:5]:
            print(f'  - {obj["Key"]}')
    else:
        print('❌ 파일이 없습니다.')

except Exception as e:
    print(f'❌ 오류: {e}')
    import traceback
    traceback.print_exc()
