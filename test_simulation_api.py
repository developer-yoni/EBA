#!/usr/bin/env python3
"""
시뮬레이션 API 테스트
"""
import requests
import json

def test_simulation_api():
    url = "http://localhost:5001/api/simulation/predict"
    
    payload = {
        "baseMonth": "2025-11",
        "simulationMonths": 12,
        "additionalChargers": 1000
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("🧪 시뮬레이션 API 테스트 시작...")
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API 테스트 성공!")
            print(f"예측 결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ API 테스트 실패: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")

if __name__ == "__main__":
    test_simulation_api()