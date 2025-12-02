import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime';
import { config } from './config.js';

// AWS SDK가 자동으로 자격 증명을 처리 (환경변수 또는 IAM Role)
const client = new BedrockRuntimeClient({ region: config.aws.region });

/**
 * Bedrock 모델을 호출하여 응답 생성
 * @param {string} prompt - 사용자 프롬프트
 * @param {string} context - Knowledge Base에서 가져온 컨텍스트 (선택사항)
 * @returns {Promise<string>} 모델 응답
 */
export async function invokeBedrockModel(prompt, context = '') {
  const systemPrompt = context 
    ? `다음 컨텍스트를 참고하여 질문에 답변해주세요:\n\n${context}\n\n질문: ${prompt}`
    : prompt;

  const payload = {
    anthropic_version: config.bedrock.anthropicVersion,
    max_tokens: config.bedrock.maxTokens,
    temperature: config.bedrock.temperature,
    messages: [
      {
        role: 'user',
        content: systemPrompt,
      },
    ],
  };

  const command = new InvokeModelCommand({
    modelId: config.bedrock.modelId,
    contentType: 'application/json',
    accept: 'application/json',
    body: JSON.stringify(payload),
  });

  try {
    const response = await client.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));
    return responseBody.content[0].text;
  } catch (error) {
    console.error('Bedrock 모델 호출 오류:', error);
    throw error;
  }
}
