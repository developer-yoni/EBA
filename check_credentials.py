"""
현재 AWS 자격 증명 설정 확인
"""
import os
from pathlib import Path

print('=' * 80)
print('🔍 AWS 자격 증명 설정 확인')
print('=' * 80)
print()

# 1. 환경변수 확인
print('1️⃣ 환경변수 (PowerShell/시스템)')
print('-' * 80)
env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_REGION']
env_found = False

for var in env_vars:
    value = os.environ.get(var)
    if value:
        env_found = True
        if 'SECRET' in var or 'TOKEN' in var:
            masked = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '***'
            print(f'  ✅ {var}: {masked}')
        else:
            print(f'  ✅ {var}: {value}')
    else:
        print(f'  ❌ {var}: 설정되지 않음')

if not env_found:
    print('  ⚠️ 환경변수가 설정되어 있지 않습니다.')

print()

# 2. .env 파일 확인
print('2️⃣ .env 파일')
print('-' * 80)
env_file = Path('.env')

if env_file.exists():
    print(f'  ✅ .env 파일 존재: {env_file.absolute()}')
    
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for var in env_vars:
        if f'{var}=' in content and not content.split(f'{var}=')[1].split('\n')[0].strip().startswith('#'):
            value = content.split(f'{var}=')[1].split('\n')[0].strip()
            if value:
                if 'SECRET' in var or 'TOKEN' in var:
                    masked = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '***'
                    print(f'  ✅ {var}: {masked}')
                else:
                    print(f'  ✅ {var}: {value}')
else:
    print('  ❌ .env 파일이 없습니다.')

print()

# 3. AWS CLI 설정 확인
print('3️⃣ AWS CLI 설정')
print('-' * 80)
aws_credentials = Path.home() / '.aws' / 'credentials'
aws_config = Path.home() / '.aws' / 'config'

if aws_credentials.exists():
    print(f'  ✅ credentials 파일 존재: {aws_credentials}')
else:
    print('  ❌ credentials 파일 없음')

if aws_config.exists():
    print(f'  ✅ config 파일 존재: {aws_config}')
else:
    print('  ❌ config 파일 없음')

print()

# 4. boto3 인식 확인
print('4️⃣ boto3 자격 증명 인식')
print('-' * 80)
try:
    import boto3
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    
    print('  ✅ boto3가 자격 증명을 인식했습니다!')
    print(f'  Account: {identity["Account"]}')
    print(f'  User/Role: {identity["Arn"]}')
except Exception as e:
    print(f'  ❌ boto3가 자격 증명을 찾을 수 없습니다.')
    print(f'  오류: {e}')

print()
print('=' * 80)
print('💡 권장사항')
print('=' * 80)

if env_found:
    print('✅ 환경변수가 설정되어 있습니다. 바로 사용 가능합니다!')
elif env_file.exists():
    print('✅ .env 파일이 있습니다. 바로 사용 가능합니다!')
elif aws_credentials.exists():
    print('✅ AWS CLI 설정이 있습니다. 바로 사용 가능합니다!')
else:
    print('⚠️ 자격 증명이 설정되지 않았습니다.')
    print()
    print('다음 중 하나를 선택하세요:')
    print('  1. .env 파일: python setup_credentials.py')
    print('  2. PowerShell 영구: .\\set_env_permanent.ps1')
    print('  3. AWS CLI: aws configure')

print('=' * 80)
