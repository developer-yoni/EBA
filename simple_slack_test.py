#!/usr/bin/env python3
"""
간단한 슬랙 전송 테스트
"""
import requests
import json
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 슬랙 Webhook URL (환경변수에서 로드)
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')

def test_slack_message():
    """슬랙 메시지 전송 테스트"""
    print("🧪 슬랙 메시지 전송 테스트...")
    
    # 테스트 메시지
    message = """📊 *EV 충전 인프라 분석 리포트 (테스트)*
━━━━━━━━━━━━━━━━━━━━━━
📅 분석 기간: *2025-11* (전월 대비 현황)

🔋 *GS차지비 현황*
• 시장점유율: *16.1%* (-0.1%p)
• 충전소: *7,431개* (+3)
• 총충전기: *73,851기* (+1,423)

📈 *전체 시장 현황*
• 총 충전소: 92,670개
• 총 충전기: *459,523기* (+6,457)

━━━━━━━━━━━━━━━━━━━━━━
🕐 생성: 2025-12-13 13:10
🤖 DataReporter 슬랙 연동 테스트"""
    
    try:
        # 슬랙 Webhook으로 메시지 전송
        payload = {
            "text": message,
            "mrkdwn": True
        }
        
        print(f"📤 메시지 전송 중... ({len(message)} 자)")
        
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"📡 응답 상태: {response.status_code}")
        print(f"📡 응답 내용: {response.text}")
        
        if response.status_code == 200 and response.text == 'ok':
            print("✅ 슬랙 전송 성공!")
            return True
        else:
            print(f"❌ 슬랙 전송 실패: {response.status_code} - {response.text}")
            return False
        
    except requests.exceptions.Timeout:
        print("❌ 슬랙 전송 시간 초과")
        return False
    except Exception as e:
        print(f"❌ 슬랙 전송 실패: {str(e)}")
        return False

if __name__ == '__main__':
    print("🚀 슬랙 연동 테스트 시작")
    print("=" * 40)
    
    success = test_slack_message()
    
    print("=" * 40)
    if success:
        print("🎉 슬랙 연동 성공!")
        print("💡 DataReporter에서 슬랙 전송 기능을 사용할 수 있습니다.")
    else:
        print("❌ 슬랙 연동 실패")
        print("🔧 Webhook URL이나 네트워크 연결을 확인해주세요.")