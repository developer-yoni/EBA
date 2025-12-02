import { retrieveFromKnowledgeBase, formatRetrievalResults } from './knowledge-base.js';
import { invokeBedrockModel } from './bedrock-client.js';

/**
 * RAG 파이프라인: Knowledge Base 검색 + Bedrock 응답 생성
 * @param {string} userQuery - 사용자 질문
 * @returns {Promise<Object>} 응답 및 검색 결과
 */
export async function processRAGQuery(userQuery) {
  console.log(`\n🔍 질문: ${userQuery}`);
  
  // 1. Knowledge Base에서 관련 문서 검색
  console.log('\n📚 Knowledge Base 검색 중...');
  const retrievalResults = await retrieveFromKnowledgeBase(userQuery);
  console.log(`✅ ${retrievalResults.length}개의 관련 문서를 찾았습니다.`);
  
  // 2. 검색 결과를 컨텍스트로 포맷
  const context = formatRetrievalResults(retrievalResults);
  
  // 3. Bedrock 모델로 응답 생성
  console.log('\n🤖 Bedrock 모델 응답 생성 중...');
  const answer = await invokeBedrockModel(userQuery, context);
  
  return {
    query: userQuery,
    answer,
    sources: retrievalResults.map((r, i) => ({
      index: i + 1,
      score: r.score,
      content: r.content?.text?.substring(0, 200) + '...',
    })),
  };
}
