#!/usr/bin/env python3
"""
새로운 슬랙 API 테스트
"""
import requests
import json

def test_new_slack_api():
    """새로운 슬랙 API 테스트"""
    print("🧪 새로운 슬랙 API 테스트...")
    
    url = "http://localhost:5001/api/slack-send"
    
    # 테스트 메시지
    test_message = """📊 *EV 충전 인프라 분석 리포트 (새 API 테스트)*
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
🕐 생성: 2025-12-13 13:30
🤖 새로운 Flask API 테스트"""
    
    try:
        print(f"📡 API 호출: {url}")
        print(f"📤 메시지 길이: {len(test_message)} 자")
        
        response = requests.post(
            url,
            json={'message': test_message},
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        print(f"📡 응답 상태: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ JSON 응답: {result}")
            
            if result.get('success'):
                print("🎉 새로운 슬랙 API 테스트 성공!")
                return True
            else:
                print(f"❌ API 오류: {result.get('error')}")
                return False
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            print(f"📄 응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        return False

if __name__ == '__main__':
    print("🚀 새로운 슬랙 API 테스트")
    print("=" * 50)
    
    success = test_new_slack_api()
    
    print("=" * 50)
    if success:
        print("🎉 새로운 API 테스트 성공!")
        print("💡 이제 웹 대시보드에서 슬랙 전송이 작동할 것입니다.")
    else:
        print("❌ 새로운 API 테스트 실패")
        print("🔧 Flask 서버나 네트워크 연결을 확인해주세요.")