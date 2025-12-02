"""
AWS 연결 테스트 스크립트
"""
import sys

def test_imports():
    """필수 패키지 import 테스트"""
    print('📦 패키지 import 테스트...')
    try:
        import boto3
        import pandas
        import openpyxl
        from dotenv import load_dotenv
        import flask
        print('✅ 모든 패키지 import 성공\n')
        return True
    except ImportError as e:
        print(f'❌ 패키지 import 실패: {e}')
        print('   pip install -r requirements.txt 를 실행하세요\n')
        return False

def test_aws_credentials():
    """AWS 자격 증명 테스트"""
    print('🔐 AWS 자격 증명 테스트...')
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f'✅ AWS 인증 성공')
        print(f'   Account: {identity["Account"]}')
        print(f'   User/Role: {identity["Arn"]}\n')
        return True
    except Exception as e:
        print(f'❌ AWS 인증 실패: {e}')
        print('   .env 파일을 확인하거나 aws configure를 실행하세요\n')
        return False

def test_s3_access():
    """S3 접근 테스트"""
    print('📦 S3 접근 테스트...')
    try:
        from config import Config
        import boto3
        
        s3 = boto3.client('s3', region_name=Config.AWS_REGION)
        
        # 버킷 존재 확인
        s3.head_bucket(Bucket=Config.S3_BUCKET)
        print(f'✅ S3 버킷 접근 성공: {Config.S3_BUCKET}')
        
        # 파일 목록 조회
        response = s3.list_objects_v2(
            Bucket=Config.S3_BUCKET,
            Prefix=Config.S3_PREFIX,
            MaxKeys=5
        )
        
        file_count = response.get('KeyCount', 0)
        print(f'   파일 수: {file_count}개')
        
        if file_count > 0:
            print('   최근 파일:')
            for obj in response.get('Contents', [])[:3]:
                print(f'     - {obj["Key"]}')
        
        print()
        return True
        
    except Exception as e:
        print(f'❌ S3 접근 실패: {e}')
        print('   버킷 이름과 권한을 확인하세요\n')
        return False

def test_bedrock_access():
    """Bedrock 접근 테스트"""
    print('🤖 Bedrock 접근 테스트...')
    try:
        from config import Config
        import boto3
        import json
        
        client = boto3.client('bedrock-runtime', region_name=Config.AWS_REGION)
        
        # 간단한 테스트 호출
        payload = {
            'anthropic_version': Config.ANTHROPIC_VERSION,
            'max_tokens': 100,
            'temperature': 0.7,
            'messages': [
                {
                    'role': 'user',
                    'content': '안녕하세요. 간단히 인사해주세요.'
                }
            ]
        }
        
        response = client.invoke_model(
            modelId=Config.MODEL_ID,
            contentType='application/json',
            accept='application/json',
            body=json.dumps(payload)
        )
        
        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']
        
        print(f'✅ Bedrock 모델 호출 성공')
        print(f'   모델: {Config.MODEL_ID}')
        print(f'   응답: {answer[:100]}...\n')
        return True
        
    except Exception as e:
        print(f'❌ Bedrock 접근 실패: {e}')
        print('   모델 ID와 권한을 확인하세요\n')
        return False

def test_knowledge_base():
    """Knowledge Base 접근 테스트"""
    print('📚 Knowledge Base 접근 테스트...')
    try:
        from config import Config
        import boto3
        
        client = boto3.client('bedrock-agent-runtime', region_name=Config.AWS_REGION)
        
        response = client.retrieve(
            knowledgeBaseId=Config.KNOWLEDGE_BASE_ID,
            retrievalQuery={'text': '충전 인프라'},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 2
                }
            }
        )
        
        results = response.get('retrievalResults', [])
        
        print(f'✅ Knowledge Base 검색 성공')
        print(f'   KB ID: {Config.KNOWLEDGE_BASE_ID}')
        print(f'   검색 결과: {len(results)}개\n')
        return True
        
    except Exception as e:
        print(f'❌ Knowledge Base 접근 실패: {e}')
        print('   Knowledge Base ID와 권한을 확인하세요\n')
        return False

def main():
    print('=' * 80)
    print('⚡ 한국 전기차 충전 인프라 분석 시스템 - 연결 테스트')
    print('=' * 80)
    print()
    
    results = []
    
    # 1. 패키지 테스트
    results.append(('패키지', test_imports()))
    
    # 2. AWS 자격 증명 테스트
    results.append(('AWS 인증', test_aws_credentials()))
    
    # 3. S3 테스트
    results.append(('S3', test_s3_access()))
    
    # 4. Bedrock 테스트
    results.append(('Bedrock', test_bedrock_access()))
    
    # 5. Knowledge Base 테스트
    results.append(('Knowledge Base', test_knowledge_base()))
    
    # 결과 요약
    print('=' * 80)
    print('📊 테스트 결과 요약')
    print('=' * 80)
    
    for name, success in results:
        status = '✅ 성공' if success else '❌ 실패'
        print(f'{name:20s}: {status}')
    
    print()
    
    all_success = all(result[1] for result in results)
    
    if all_success:
        print('🎉 모든 테스트 통과! 시스템을 사용할 준비가 되었습니다.')
        print()
        print('다음 명령으로 시작하세요:')
        print('  - CLI 모드: python cli_runner.py')
        print('  - 웹 앱: python app.py')
    else:
        print('⚠️ 일부 테스트 실패. SETUP_GUIDE.md를 참고하여 설정을 확인하세요.')
        sys.exit(1)
    
    print('=' * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n⚠️ 테스트가 중단되었습니다.')
        sys.exit(0)
    except Exception as e:
        print(f'\n❌ 예상치 못한 오류: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
