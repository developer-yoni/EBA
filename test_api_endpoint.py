#!/usr/bin/env python3
"""
슬랙 전송 API 엔드포인트 테스트
"""
import requests
import json

def test_slack_api():
    """슬랙 전송 API 테스트"""
    print("🧪 슬랙 전송 API 테스트...")
    
    url = "http://localhost:5001/api/send-to-slack"
    
    # 테스트 데이터
    test_data = {
        "startMonth": "2025-10",
        "endMonth": "2025-11",
        "months": ["2025-10", "2025-11"]
    }
    
    try:
        print(f"📡 API 호출: {url}")
        print(f"📤 데이터: {test_data}")
        
        response = requests.post(
            url,
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"📡 응답 상태: {response.status_code}")
        print(f"📡 응답 헤더: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ JSON 응답: {result}")
                
                if result.get('success'):
                    print("🎉 슬랙 전송 API 성공!")
                else:
                    print(f"❌ API 오류: {result.get('error')}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")
                print(f"📄 응답 내용 (처음 500자): {response.text[:500]}")
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            print(f"📄 응답 내용: {response.text[:500]}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: Flask 서버가 실행되지 않음")
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")

def test_server_status():
    """서버 상태 확인"""
    print("🔍 서버 상태 확인...")
    
    try:
        # 기본 대시보드 페이지 확인
        response = requests.get("http://localhost:5001/dashboard", timeout=5)
        print(f"📡 대시보드 페이지: {response.status_code}")
        
        # API 상태 확인
        response = requests.get("http://localhost:5001/api/months", timeout=5)
        print(f"📡 API 엔드포인트: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Flask 서버 정상 실행 중")
        else:
            print("⚠️ Flask 서버 응답 이상")
            
    except Exception as e:
        print(f"❌ 서버 상태 확인 실패: {e}")

if __name__ == '__main__':
    print("🚀 슬랙 API 엔드포인트 테스트")
    print("=" * 50)
    
    # 1. 서버 상태 확인
    test_server_status()
    
    print()
    
    # 2. 슬랙 API 테스트
    test_slack_api()
    
    print("=" * 50)
    print("🎉 테스트 완료!")