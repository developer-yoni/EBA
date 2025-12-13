#!/usr/bin/env python3
"""
독립적인 슬랙 리포트 전송 스크립트
웹 대시보드에서 직접 호출 가능
"""
import sys
import json
import requests
from datetime import datetime
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from slack_sender import SlackDashboardSender, send_to_slack_webhook

def send_dashboard_report(start_month, end_month):
    """대시보드 리포트를 슬랙으로 전송"""
    
    print(f"📤 슬랙 리포트 전송 시작: {start_month} ~ {end_month}")
    
    try:
        # Flask API에서 대시보드 데이터 가져오기
        dashboard_url = "http://localhost:5001/api/dashboard"
        
        # 기간 내 모든 월 계산
        from datetime import datetime
        start_date = datetime.strptime(start_month, '%Y-%m')
        end_date = datetime.strptime(end_month, '%Y-%m')
        
        months = []
        current = start_date
        while current <= end_date:
            months.append(current.strftime('%Y-%m'))
            # 다음 달로 이동
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        print(f"📅 포함 월: {months}")
        
        # 대시보드 API 호출
        response = requests.post(
            dashboard_url,
            json={
                'months': months,
                'startMonth': start_month,
                'endMonth': end_month
            },
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ 대시보드 API 오류: {response.status_code}")
            return False
        
        data = response.json()
        if not data.get('success'):
            print(f"❌ 대시보드 데이터 오류: {data.get('error')}")
            return False
        
        dashboard_data = data.get('dashboard', {})
        print(f"✅ 대시보드 데이터 수신 완료")
        
        # 슬랙 메시지 생성 및 전송
        sender = SlackDashboardSender()
        message = sender.create_slack_message(dashboard_data, start_month, end_month)
        
        print(f"📝 슬랙 메시지 생성 완료 ({len(message)} 자)")
        
        # 슬랙으로 전송
        result = send_to_slack_webhook(message)
        
        if result['success']:
            print(f"✅ 슬랙 전송 성공!")
            return True
        else:
            print(f"❌ 슬랙 전송 실패: {result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("사용법: python send_slack_report.py <시작월> <종료월>")
        print("예시: python send_slack_report.py 2025-10 2025-11")
        sys.exit(1)
    
    start_month = sys.argv[1]
    end_month = sys.argv[2]
    
    print("🚀 독립 슬랙 리포트 전송 스크립트")
    print("=" * 50)
    
    success = send_dashboard_report(start_month, end_month)
    
    print("=" * 50)
    if success:
        print("🎉 슬랙 리포트 전송 완료!")
    else:
        print("❌ 슬랙 리포트 전송 실패")
    
    sys.exit(0 if success else 1)