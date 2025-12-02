"""
Knowledge Base에서 데이터 구조 분석
"""
import boto3
import json
from config import Config

def invoke_bedrock(prompt, context=''):
    """Bedrock 모델 호출"""
    client = boto3.client('bedrock-runtime', region_name=Config.AWS_REGION)
    
    system_prompt = f"{context}\n\n{prompt}" if context else prompt
    
    payload = {
        'anthropic_version': Config.ANTHROPIC_VERSION,
        'max_tokens': Config.MAX_TOKENS,
        'temperature': Config.TEMPERATURE,
        'messages': [
            {
                'role': 'user',
                'content': system_prompt
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
    return response_body['content'][0]['text']

def retrieve_from_kb(query):
    """Knowledge Base에서 검색"""
    client = boto3.client('bedrock-agent-runtime', region_name=Config.AWS_REGION)
    
    response = client.retrieve(
        knowledgeBaseId=Config.KNOWLEDGE_BASE_ID,
        retrievalQuery={'text': query},
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': Config.KB_NUMBER_OF_RESULTS
            }
        }
    )
    
    results = response.get('retrievalResults', [])
    context = '\n\n---\n\n'.join([
        f"[문서 {i+1}] (관련도: {r.get('score', 0):.3f})\n{r.get('content', {}).get('text', '')}"
        for i, r in enumerate(results)
    ])
    
    return context

def analyze_data_structure():
    """데이터 구조 분석"""
    print('📊 충전인프라 현황 데이터 구조 분석 중...\n')
    
    queries = [
        '충전인프라 현황_2510.xlsx 파일의 전체 구조를 자세히 설명해주세요. 특히 헤더가 몇 번째 행에 있는지, 어떤 컬럼들이 있는지 정확히 알려주세요.',
        '엑셀 파일의 상단에 있는 제목 문자열(예: "KR CHARING INFRASTRUCTURE STATUS_24.10.01")의 정확한 위치(행 번호, 열 번호)를 알려주세요.',
        '데이터에 포함된 모든 컬럼명을 정확히 나열하고, 각 컬럼의 의미를 설명해주세요.',
        '데이터의 샘플 행 2-3개를 보여주세요. 실제 값들을 포함해서 보여주세요.'
    ]
    
    for i, query in enumerate(queries, 1):
        print('=' * 80)
        print(f'❓ 질문 {i}: {query}\n')
        
        try:
            # Knowledge Base에서 검색
            context = retrieve_from_kb(query)
            
            # Bedrock으로 답변 생성
            answer = invoke_bedrock(query, context)
            
            print('💡 답변:')
            print(answer)
            print('\n')
            
        except Exception as e:
            print(f'❌ 오류: {e}\n')

if __name__ == '__main__':
    analyze_data_structure()
