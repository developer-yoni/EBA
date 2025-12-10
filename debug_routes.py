#!/usr/bin/env python3
"""
Flask 앱의 라우트 확인
"""
from app import app

print("등록된 라우트:")
for rule in app.url_map.iter_rules():
    print(f"  {rule.rule} -> {rule.endpoint} ({rule.methods})")