#!/usr/bin/env python3
"""
간단한 슬랙 프록시 서버
CORS 문제를 해결하기 위한 프록시
"""
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# CORS 헤더 수동 추가
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 슬랙 Webhook URL (환경 변수에서 가져오기)
import os
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/services/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')

def send_to_slack_webhook(message: str) -> dict:
    """
    슬랙 Webhook으로 메시지 전송
    
    Args:
        message: 전송할 메시지
    
    Returns:
        전송 결과
    """
    result = {
        'success': False,
        'message': ''
    }
    
    try:
        # 슬랙 Webhook으로 메시지 전송
        payload = {
            "text": message,
            "mrkdwn": True
        }
        
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200 and response.text == 'ok':
            result['success'] = True
            result['message'] = '슬랙 전송 성공!'
        else:
            result['message'] = f'슬랙 전송 실패: {response.status_code} - {response.text}'
        
        return result
        
    except requests.exceptions.Timeout:
        result['message'] = '슬랙 전송 시간 초과'
        return result
    except Exception as e:
        result['message'] = f'슬랙 전송 실패: {str(e)}'
        return result

@app.route('/slack-proxy', methods=['POST', 'OPTIONS'])
def slack_proxy():
    """슬랙 Webhook 프록시"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # 클라이언트에서 받은 메시지
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({
                'success': False,
                'error': '메시지가 없습니다'
            }), 400
        
        print(f"📤 슬랙 프록시: 메시지 전송 중... ({len(message)} 자)")
        
        # 기존의 성공한 슬랙 전송 함수 사용
        result = send_to_slack_webhook(message)
        
        if result['success']:
            print("✅ 슬랙 전송 성공!")
            return jsonify(result)
        else:
            print(f"❌ 슬랙 전송 실패: {result['message']}")
            return jsonify(result), 500
        
    except Exception as e:
        print(f"❌ 프록시 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({'status': 'ok', 'service': 'slack-proxy'})

if __name__ == '__main__':
    print("🚀 슬랙 프록시 서버 시작")
    print("   - 포트: 5002")
    print("   - 엔드포인트: /slack-proxy")
    print("   - CORS: 활성화")
    
    app.run(debug=True, host='0.0.0.0', port=5002, use_reloader=False)