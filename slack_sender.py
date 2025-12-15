#!/usr/bin/env python3
"""
슬랙 메시지 전송 모듈
환경변수에서 SLACK_WEBHOOK_URL을 읽어 사용
"""
import os
import requests
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 슬랙 Webhook URL (환경변수에서 로드)
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')


def send_to_slack_webhook(message: str) -> dict:
    """
    슬랙 Webhook으로 메시지 전송
    
    Args:
        message: 전송할 메시지
    
    Returns:
        전송 결과 dict {'success': bool, 'message': str}
    """
    result = {
        'success': False,
        'message': ''
    }
    
    if not SLACK_WEBHOOK_URL:
        result['message'] = 'SLACK_WEBHOOK_URL 환경변수가 설정되지 않았습니다'
        return result
    
    try:
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


class SlackDashboardSender:
    """대시보드 데이터를 슬랙 메시지로 변환하여 전송"""
    
    def __init__(self):
        self.webhook_url = SLACK_WEBHOOK_URL
    
    def format_dashboard_message(self, dashboard_data: dict, start_month: str, end_month: str) -> str:
        """대시보드 데이터를 슬랙 메시지 형식으로 변환"""
        gs_kpi = dashboard_data.get('gs_kpi', {})
        summary = dashboard_data.get('summary', {})
        
        current = gs_kpi.get('current', {})
        monthly_change = gs_kpi.get('monthly_change', {})
        
        message = f"""📊 *EV 충전 인프라 분석 리포트*
━━━━━━━━━━━━━━━━━━━━━━
📅 분석 기간: *{start_month} ~ {end_month}*

🔋 *GS차지비 현황*
• 시장점유율: *{current.get('market_share', 0)}%* ({monthly_change.get('market_share_change', 0):+.1f}%p)
• 충전소: *{current.get('stations', 0):,}개* ({monthly_change.get('stations', 0):+,})
• 총충전기: *{current.get('total_chargers', 0):,}기* ({monthly_change.get('total_chargers', 0):+,})

📈 *전체 시장 현황*
• 총 충전소: {summary.get('total_stations', 0):,}개
• 총 충전기: *{summary.get('total_chargers', 0):,}기*

━━━━━━━━━━━━━━━━━━━━━━
🕐 생성: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
🤖 DataReporter"""
        
        return message
    
    def send(self, dashboard_data: dict, start_month: str, end_month: str) -> dict:
        """대시보드 데이터를 슬랙으로 전송"""
        message = self.format_dashboard_message(dashboard_data, start_month, end_month)
        return send_to_slack_webhook(message)
