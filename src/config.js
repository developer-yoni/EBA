import dotenv from 'dotenv';

dotenv.config();

export const config = {
  aws: {
    region: process.env.AWS_REGION || 'ap-northeast-2',
    // AWS SDK가 자동으로 환경변수나 IAM Role에서 자격 증명을 가져옴
  },
  bedrock: {
    modelId: process.env.MODEL_ID || 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
    anthropicVersion: 'bedrock-2023-05-31',
    maxTokens: 2048,
    temperature: 0.7,
  },
  knowledgeBase: {
    id: process.env.KNOWLEDGE_BASE_ID || 'XHG5MMFIYK',
    numberOfResults: 100,  // RAG 모든 데이터 활용 (AWS 최대값: 100)
  },
};
