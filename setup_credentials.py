"""
AWS 자격 증명 설정 도우미
"""
import os

def setup_credentials():
    print('=' * 80)
    print('🔐 AWS 자격 증명 설정')
    print('=' * 80)
    print()
    print('AWS 자격 증명을 입력해주세요.')
    print('(입력하지 않으려면 Enter를 누르세요)')
    print()
    
    # 현재 .env 파일 읽기
    env_file = '.env'
    env_content = {}
    
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_content[key.strip()] = value.strip()
    
    # 자격 증명 입력
    print('1️⃣ AWS Access Key ID')
    print('   예: AKIAIOSFODNN7EXAMPLE')
    access_key = input('   입력: ').strip()
    
    print()
    print('2️⃣ AWS Secret Access Key')
    print('   예: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY')
    secret_key = input('   입력: ').strip()
    
    print()
    print('3️⃣ AWS Session Token (선택사항 - 임시 자격 증명 사용 시)')
    session_token = input('   입력 (없으면 Enter): ').strip()
    
    # .env 파일 업데이트
    if access_key:
        env_content['AWS_ACCESS_KEY_ID'] = access_key
    if secret_key:
        env_content['AWS_SECRET_ACCESS_KEY'] = secret_key
    if session_token:
        env_content['AWS_SESSION_TOKEN'] = session_token
    
    # 기본 설정 유지
    env_content.setdefault('AWS_REGION', 'ap-northeast-2')
    env_content.setdefault('S3_BUCKET', 's3-eba-team3')
    env_content.setdefault('S3_PREFIX', '충전인프라현황DB/')
    env_content.setdefault('KNOWLEDGE_BASE_ID', 'XHG5MMFIYK')
    env_content.setdefault('MODEL_ID', 'global.anthropic.claude-sonnet-4-5-20250929-v1:0')
    
    # .env 파일 저장
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write('# AWS 자격 증명\n')
        if 'AWS_ACCESS_KEY_ID' in env_content:
            f.write(f'AWS_ACCESS_KEY_ID={env_content["AWS_ACCESS_KEY_ID"]}\n')
        if 'AWS_SECRET_ACCESS_KEY' in env_content:
            f.write(f'AWS_SECRET_ACCESS_KEY={env_content["AWS_SECRET_ACCESS_KEY"]}\n')
        if 'AWS_SESSION_TOKEN' in env_content:
            f.write(f'AWS_SESSION_TOKEN={env_content["AWS_SESSION_TOKEN"]}\n')
        
        f.write('\n# AWS 리전\n')
        f.write(f'AWS_REGION={env_content["AWS_REGION"]}\n')
        
        f.write('\n# S3 설정\n')
        f.write(f'S3_BUCKET={env_content["S3_BUCKET"]}\n')
        f.write(f'S3_PREFIX={env_content["S3_PREFIX"]}\n')
        
        f.write('\n# Knowledge Base ID\n')
        f.write(f'KNOWLEDGE_BASE_ID={env_content["KNOWLEDGE_BASE_ID"]}\n')
        
        f.write('\n# Bedrock 모델 설정\n')
        f.write(f'MODEL_ID={env_content["MODEL_ID"]}\n')
    
    print()
    print('=' * 80)
    print('✅ .env 파일이 업데이트되었습니다!')
    print('=' * 80)
    print()
    print('다음 단계:')
    print('  1. 연결 테스트: python test_connection.py')
    print('  2. 시스템 실행: python cli_runner.py')
    print()

if __name__ == '__main__':
    try:
        setup_credentials()
    except KeyboardInterrupt:
        print('\n\n⚠️ 설정이 취소되었습니다.')
    except Exception as e:
        print(f'\n❌ 오류: {e}')
