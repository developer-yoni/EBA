import { processRAGQuery } from './src/rag-service.js';

async function analyzeDataStructure() {
  try {
    console.log('📊 충전인프라 현황 데이터 구조 분석 중...\n');
    
    const queries = [
      '충전인프라 현황_2510.xlsx 파일의 전체 구조를 자세히 설명해주세요. 특히 헤더가 몇 번째 행에 있는지, 어떤 컬럼들이 있는지 알려주세요.',
      '엑셀 파일의 상단에 있는 제목 문자열의 정확한 위치(행, 열)와 형식을 알려주세요. 예: "KR CHARING INFRASTRUCTURE STATUS_24.10.01"',
      '데이터에 포함된 모든 컬럼명을 나열하고, 각 컬럼의 의미를 설명해주세요.',
      '데이터의 샘플 행 2-3개를 보여주세요.'
    ];
    
    for (const query of queries) {
      console.log('='.repeat(80));
      console.log(`❓ ${query}\n`);
      
      const result = await processRAGQuery(query);
      
      console.log('💡 답변:');
      console.log(result.answer);
      console.log('\n');
    }
    
  } catch (error) {
    console.error('❌ 오류:', error.message);
  }
}

analyzeDataStructure();
