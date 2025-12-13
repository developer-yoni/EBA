#!/usr/bin/env python3
"""
슬랙 연동 기능 테스트 스크립트
"""
import sys
import os
import json
import requests

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slack_sender import SlackDashboardSender, send_to_slack_webhook

def test_slack_webhook():
    """슬랙 Webhook 테스트"""
    print("🧪 슬랙 Webhook 테스트 시작...")
    
    # 테스트 메시지
    test_message = """📊 *EV 충전 인프라 분석 리포트 (테스트)*
━━━━━━━━━━━━━━━━━━━━━━
📅 분석 기간: *2025-11* (전월 대비 현황)

🔋 *GS차지비 현황*
• 시장점유율: *16.1%* (-0.1%p)
• 충전소: *7,431개* (+3)
• 총충전기: *73,851기* (+1,423)
• 완속충전기: *59,437기* (+896)
• 급속충전기: *14,414기* (+527)

📈 *전체 시장 현황*
• 총 충전소: 92,670개
• 총 충전기: *459,523기* (+6,457)
• 완속충전기: *410,205기* (+5,930)
• 급속충전기: *49,318기* (+527)

━━━━━━━━━━━━━━━━━━━━━━
🕐 생성: 2025-12-13 13:09
"""
    
    # 슬랙으로 전송
    result = send_to_slack_webhook(test_message)
    
    if result['success']:
        print("✅ 슬랙 전송 성공!")
        print(f"   메시지: {result['message']}")
    else:
        print("❌ 슬랙 전송 실패!")
        print(f"   오류: {result['message']}")
    
    return result

def test_dashboard_sender():
    """대시보드 전송 클래스 테스트"""
    print("\n🧪 대시보드 전송 클래스 테스트...")
    
    # 테스트 대시보드 데이터
    test_dashboard_data = {
        'gs_kpi': {
            'current': {
                'market_share': 16.1,
                'stations': 7431,
                'slow_chargers': 59437,
                'fast_chargers': 14414,
                'total_chargers': 73851
            },
            'monthly_change': {
                'market_share_change': -0.1,
                'stations': 3,
                'slow_chargers': 896,
                'fast_chargers': 527,
                'total_chargers': 1423
            }
        },
        'summary_table': {
            'total': {
                'cpos': 145,
                'stations': 92670,
                'slow_chargers': 410205,
                'fast_chargers': 49318,
                'total_chargers': 459523
            },
            'change': {
                'cpos': 1,
                'stations': 649,
                'slow_chargers': 5930,
                'fast_chargers': 527,
                'total_chargers': 6457
            }
        }
    }
    
    sender = SlackDashboardSender()
    
    # 슬랙 메시지 생성
    message = sender.create_slack_message(test_dashboard_data, '2025-10', '2025-11')
    print(f"📝 생성된 메시지 길이: {len(message)} 자")
    print(f"📝 메시지 미리보기:\n{message[:500]}...")
    
    # HTML 파일 생성
    html_file = sender.save_dashboard_html(test_dashboard_data, '2025-10', '2025-11')
    print(f"📄 HTML 파일 생성: {html_file}")
    
    return message

def test_flask_api():
    """Flask API 테스트 (서버가 실행 중인 경우)"""
    print("\n🧪 Flask API 테스트...")
    
    try:
        # 대시보드 API 호출 테스트
        response = requests.get('http://localhost:5001/api/dashboard', timeout=5)
        print(f"📡 대시보드 API 상태: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Flask 서버 실행 중")
        else:
            print("⚠️ Flask 서버 응답 이상")
            
    except requests.exceptions.ConnectionError:
        print("❌ Flask 서버가 실행되지 않음 (localhost:5001)")
    except Exception as e:
        print(f"❌ API 테스트 오류: {e}")

if __name__ == '__main__':
    print("🚀 슬랙 연동 기능 통합 테스트")
    print("=" * 50)
    
    # 1. 슬랙 Webhook 테스트
    webhook_result = test_slack_webhook()
    
    # 2. 대시보드 전송 클래스 테스트
    message = test_dashboard_sender()
    
    # 3. Flask API 테스트
    test_flask_api()
    
    print("\n" + "=" * 50)
    print("🎉 테스트 완료!")
    
    if webhook_result['success']:
        print("✅ 슬랙 연동이 정상적으로 작동합니다!")
        print("💡 이제 웹 대시보드에서 '슬랙으로 전송' 버튼을 사용할 수 있습니다.")
    else:
        print("❌ 슬랙 연동에 문제가 있습니다.")
        print("🔧 Webhook URL이나 네트워크 연결을 확인해주세요.")