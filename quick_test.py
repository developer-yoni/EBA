"""
빠른 Bedrock 테스트
"""
from ai_report_generator import AIReportGenerator
import time

print('🤖 Bedrock 연결 테스트 시작...\n')

gen = AIReportGenerator()

print('📝 간단한 질문 테스트...')
start = time.time()

try:
    answer = gen.invoke_bedrock('안녕하세요. 한 문장으로 인사해주세요.')
    elapsed = time.time() - start
    
    print(f'✅ 성공! ({elapsed:.1f}초)')
    print(f'응답: {answer}\n')
    
    print('📚 Knowledge Base 검색 테스트...')
    start = time.time()
    
    context = gen.retrieve_from_kb('충전 인프라')
    elapsed = time.time() - start
    
    print(f'✅ 성공! ({elapsed:.1f}초)')
    print(f'검색 결과 길이: {len(context)} 문자\n')
    
    print('🎉 모든 테스트 통과!')
    
except Exception as e:
    print(f'❌ 오류: {e}')
    import traceback
    traceback.print_exc()
