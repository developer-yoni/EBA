import { BedrockAgentRuntimeClient, RetrieveCommand } from '@aws-sdk/client-bedrock-agent-runtime';
import { config } from './config.js';

// AWS SDK가 자동으로 자격 증명을 처리 (환경변수 또는 IAM Role)
const client = new BedrockAgentRuntimeClient({ region: config.aws.region });

/**
 * Knowledge Base에서 관련 문서 검색
 * @param {string} query - 검색 쿼리
 * @returns {Promise<Array>} 검색 결과
 */
export async function retrieveFromKnowledgeBase(query) {
  const command = new RetrieveCommand({
    knowledgeBaseId: config.knowledgeBase.id,
    retrievalQuery: {
      text: query,
    },
    retrievalConfiguration: {
      vectorSearchConfiguration: {
        numberOfResults: config.knowledgeBase.numberOfResults,
      },
    },
  });

  try {
    const response = await client.send(command);
    return response.retrievalResults || [];
  } catch (error) {
    console.error('Knowledge Base 검색 오류:', error);
    throw error;
  }
}

/**
 * 검색 결과를 컨텍스트 문자열로 변환
 * @param {Array} results - 검색 결과
 * @returns {string} 포맷된 컨텍스트
 */
export function formatRetrievalResults(results) {
  return results
    .map((result, index) => {
      const content = result.content?.text || '';
      const score = result.score?.toFixed(3) || 'N/A';
      return `[문서 ${index + 1}] (관련도: ${score})\n${content}`;
    })
    .join('\n\n---\n\n');
}
