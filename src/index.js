import { processRAGQuery } from './rag-service.js';

async function main() {
  try {
    // 예제 질문
    const query = '이 문서의 주요 내용은 무엇인가요?';
    
    const result = await processRAGQuery(query);
    
    console.log('\n' + '='.repeat(80));
    console.log('📝 응답:');
    console.log('='.repeat(80));
    console.log(result.answer);
    console.log('\n' + '='.repeat(80));
    console.log('📎 참고 문서:');
    console.log('='.repeat(80));
    result.sources.forEach(source => {
      console.log(`\n[${source.index}] 관련도: ${source.score?.toFixed(3) || 'N/A'}`);
      console.log(source.content);
    });
    
  } catch (error) {
    console.error('\n❌ 오류 발생:', error.message);
    process.exit(1);
  }
}

main();
