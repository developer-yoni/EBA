"""
코드 인터프리터를 활용한 동적 차트 생성 모듈
"""
import json
import base64
import io
import os
from datetime import datetime

class ChartGenerator:
    """동적 차트 생성기 - 코드 인터프리터 방식"""
    
    def __init__(self):
        self.chart_output_dir = 'static/charts'
        os.makedirs(self.chart_output_dir, exist_ok=True)
    
    def generate_chart_code(self, chart_type: str, data: dict, title: str, options: dict = None) -> str:
        """차트 생성을 위한 Python 코드 생성"""
        
        # 다중 시리즈 차트인 경우
        if data.get('multi_series'):
            return self._generate_multi_series_chart_code(data, title, options)
        
        if chart_type == 'line':
            return self._generate_line_chart_code(data, title, options)
        elif chart_type == 'bar':
            return self._generate_bar_chart_code(data, title, options)
        elif chart_type == 'pie':
            return self._generate_pie_chart_code(data, title, options)
        elif chart_type == 'area':
            return self._generate_area_chart_code(data, title, options)
        else:
            return self._generate_line_chart_code(data, title, options)
    
    def _generate_multi_series_chart_code(self, data: dict, title: str, options: dict = None) -> str:
        """다중 시리즈 라인 차트 코드 생성"""
        labels = data.get('labels', [])
        series = data.get('series', [])
        y_axis_label = data.get('y_axis_label', '값')
        
        # 시리즈 데이터를 Python 코드로 변환
        series_code = ""
        annotation_code = ""
        colors = ['#667eea', '#48bb78', '#ed8936', '#e53e3e', '#9f7aea']
        
        for i, s in enumerate(series):
            color = colors[i % len(colors)]
            series_code += f'''
series_{i}_values = {s['values']}
ax.plot(labels, series_{i}_values, marker='o', linewidth=2, markersize=6, 
        color='{color}', label='{s['name']}')
'''
            # 각 포인트에 값 표시 (시리즈별로 위/아래 오프셋 다르게)
            offset = 8 if i % 2 == 0 else -15
            annotation_code += f'''
for j, v in enumerate(series_{i}_values):
    ax.annotate(f'{{v:,.0f}}', (labels[j], v), textcoords="offset points",
                xytext=(0, {offset}), ha='center', fontsize=7, color='{color}', alpha=0.8)
'''
        
        return f'''
import matplotlib.pyplot as plt
import numpy as np
import base64
import io

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

labels = {labels}

fig, ax = plt.subplots(figsize=(14, 7))

{series_code}

# 각 데이터 포인트에 값 표시
{annotation_code}

ax.set_title('{title}', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('기간', fontsize=12)
ax.set_ylabel('{y_axis_label}', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', fontsize=10)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()

print(f"data:image/png;base64,{{img_base64}}")
'''
    
    def _generate_line_chart_code(self, data: dict, title: str, options: dict = None) -> str:
        """라인 차트 코드 생성"""
        return f'''
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import base64
import io

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False

# 데이터
labels = {data.get('labels', [])}
values = {data.get('values', [])}

# 차트 생성
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(labels, values, marker='o', linewidth=2, markersize=8, color='#667eea')
ax.fill_between(labels, values, alpha=0.3, color='#667eea')

# 스타일링
ax.set_title('{title}', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('기간', fontsize=12)
ax.set_ylabel('값', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

# 값 표시
for i, v in enumerate(values):
    ax.annotate(f'{{v:,}}', (labels[i], v), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Base64로 인코딩
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()

print(f"data:image/png;base64,{{img_base64}}")
'''
    
    def _generate_bar_chart_code(self, data: dict, title: str, options: dict = None) -> str:
        """바 차트 코드 생성"""
        y_axis_label = data.get('y_axis_label', '값')
        y_axis_type = data.get('y_axis_type', 'value')
        
        # 타입에 따른 포맷 변경
        if y_axis_type == 'percentage':
            value_format = "f'{v:.1f}%'"
        elif y_axis_type == 'calculated_rate':
            value_format = "f'{v:.1f}%'"  # 증가률도 퍼센트로 표시
        else:
            value_format = "f'{v:,}'"
        
        return f'''
import matplotlib.pyplot as plt
import numpy as np
import base64
import io

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

labels = {data.get('labels', [])}
values = {data.get('values', [])}

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(labels)))
bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.2)

ax.set_title('{title}', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('CPO', fontsize=12)
ax.set_ylabel('{y_axis_label}', fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

for bar, v in zip(bars, values):
    ax.annotate({value_format}, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()

print(f"data:image/png;base64,{{img_base64}}")
'''
    
    def _generate_pie_chart_code(self, data: dict, title: str, options: dict = None) -> str:
        """파이 차트 코드 생성"""
        y_axis_type = data.get('y_axis_type', 'value')
        y_axis_label = data.get('y_axis_label', '값')
        
        # 값 표시 방식에 따른 autopct 설정
        if y_axis_type == 'percentage':
            # 점유율 모드: 퍼센트로 표시
            autopct_code = "autopct='%1.1f%%'"
            legend_format = "f'{l}: {v:.1f}%'"
        elif y_axis_type == 'calculated_rate':
            # 증가률 모드: 퍼센트로 표시
            autopct_code = "autopct='%1.1f%%'"
            legend_format = "f'{l}: {v:.1f}%'"
        else:
            # 개수 모드: 실제 값과 비율 함께 표시
            autopct_code = "autopct=lambda pct: f'{int(pct/100.*sum(values)):,}'"
            legend_format = "f'{l}: {v:,}'"
        
        return f'''
import matplotlib.pyplot as plt
import base64
import io

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

labels = {data.get('labels', [])}
values = {data.get('values', [])}

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Set3(range(len(labels)))

# 파이 차트 생성
wedges, texts, autotexts = ax.pie(values, labels=None, {autopct_code},
                                   colors=colors, startangle=90, pctdistance=0.75)

# 범례 추가 (라벨 + 값)
legend_labels = [{legend_format} for l, v in zip(labels, values)]
ax.legend(wedges, legend_labels, title="{y_axis_label}", loc="center left", 
          bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)

ax.set_title('{title}', fontsize=16, fontweight='bold', pad=20)
plt.setp(autotexts, size=9, weight='bold')

plt.tight_layout()

buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()

print(f"data:image/png;base64,{{img_base64}}")
'''
    
    def _generate_area_chart_code(self, data: dict, title: str, options: dict = None) -> str:
        """영역 차트 코드 생성"""
        return f'''
import matplotlib.pyplot as plt
import numpy as np
import base64
import io

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

labels = {data.get('labels', [])}
values = {data.get('values', [])}

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(range(len(labels)), values, alpha=0.6, color='#48bb78')
ax.plot(range(len(labels)), values, color='#2f855a', linewidth=2)

ax.set_title('{title}', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.grid(True, linestyle='--', alpha=0.7)

for i, v in enumerate(values):
    ax.annotate(f'{{v:,}}', (i, v), textcoords="offset points",
                xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()

buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()

print(f"data:image/png;base64,{{img_base64}}")
'''
    
    def execute_chart_code(self, code: str) -> dict:
        """차트 코드 실행 및 이미지 반환"""
        try:
            # 로컬 실행 (matplotlib 사용)
            import subprocess
            import tempfile
            
            # 임시 파일에 코드 저장
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # 코드 실행
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # 임시 파일 삭제
            os.unlink(temp_file)
            
            if result.returncode == 0:
                # stdout에서 base64 이미지 추출
                output = result.stdout.strip()
                if output.startswith('data:image'):
                    return {
                        'success': True,
                        'image': output
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Invalid output: {output[:100]}'
                    }
            else:
                return {
                    'success': False,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': '차트 생성 시간 초과'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
