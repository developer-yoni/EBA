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
            values = s['values']
            series_name = s['name']  # 시리즈 이름을 변수로 추출
            
            # 값 길이가 labels와 다르면 None으로 패딩
            if len(values) != len(labels):
                # 값이 부족하면 None으로 채움
                padded_values = values + [None] * (len(labels) - len(values))
                values = padded_values[:len(labels)]
            
            series_code += f'''
series_{i}_values = {values}
# None 값을 제외하고 플롯
valid_indices_{i} = [j for j, v in enumerate(series_{i}_values) if v is not None]
valid_labels_{i} = [labels[j] for j in valid_indices_{i}]
valid_values_{i} = [series_{i}_values[j] for j in valid_indices_{i}]
if valid_values_{i}:
    ax.plot(valid_labels_{i}, valid_values_{i}, marker='o', linewidth=2, markersize=6, 
            color='{color}', label='{series_name}')
'''
            # 각 포인트에 값 표시 (시리즈별로 위/아래 오프셋 다르게)
            offset = 10 if i % 2 == 0 else -18
            annotation_code += f'''
for idx, (label, v) in enumerate(zip(valid_labels_{i}, valid_values_{i})):
    ax.annotate(f'{{v:,.0f}}', (label, v), textcoords="offset points",
                xytext=(0, {offset}), ha='center', fontsize=8, color='{color}', 
                fontweight='bold', alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='{color}', alpha=0.7))
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
        # 주의: 파이 차트의 autopct는 전체 합계 대비 비율을 계산함
        # 이미 퍼센트 값인 경우 원래 값을 그대로 표시해야 함
        if y_axis_type == 'percentage':
            # 점유율 모드: 이미 퍼센트 값이므로 원래 값을 그대로 표시
            # autopct 대신 수동으로 원래 값을 표시
            autopct_code = "autopct=lambda pct: f'{values[int(round(pct/100.*len(values)))-1] if int(round(pct/100.*len(values))) > 0 else values[0]:.1f}%'"
            legend_format = "f'{l}: {v:.1f}%'"
            use_original_values = True
        elif y_axis_type == 'calculated_rate':
            # 증가률 모드: 이미 퍼센트 값이므로 원래 값을 그대로 표시
            autopct_code = "autopct=lambda pct: f'{values[int(round(pct/100.*len(values)))-1] if int(round(pct/100.*len(values))) > 0 else values[0]:.1f}%'"
            legend_format = "f'{l}: {v:.1f}%'"
            use_original_values = True
        else:
            # 개수 모드: 실제 값과 비율 함께 표시
            autopct_code = "autopct=lambda pct: f'{int(pct/100.*sum(values)):,}'"
            legend_format = "f'{l}: {v:,}'"
            use_original_values = False
        
        # 퍼센트 값인 경우 원래 값을 직접 표시하는 방식 사용
        if y_axis_type in ['percentage', 'calculated_rate']:
            return f'''
import matplotlib.pyplot as plt
import numpy as np
import base64
import io

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

labels = {data.get('labels', [])}
values = {data.get('values', [])}

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Set3(range(len(labels)))

# 파이 차트 생성 (autopct 없이)
wedges, texts = ax.pie(values, labels=None, colors=colors, startangle=90)

# 각 조각에 원래 퍼센트 값 표시
for i, (wedge, val) in enumerate(zip(wedges, values)):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = 0.7 * wedge.r * np.cos(np.radians(angle))
    y = 0.7 * wedge.r * np.sin(np.radians(angle))
    ax.text(x, y, f'{{val:.1f}}%', ha='center', va='center', fontsize=9, fontweight='bold')

# 범례 추가 (라벨 + 값)
legend_labels = [{legend_format} for l, v in zip(labels, values)]
ax.legend(wedges, legend_labels, title="{y_axis_label}", loc="center left", 
          bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)

ax.set_title('{title}', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()

buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()

print(f"data:image/png;base64,{{img_base64}}")
'''
        
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
            import sys
            
            # 디버깅: 생성된 코드 출력
            print(f'   └─ 📝 생성된 차트 코드 (처음 1500자):\n{code[:1500]}...', flush=True)
            
            # 임시 파일에 코드 저장
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # 현재 Python 인터프리터 사용 (conda 환경 유지)
            result = subprocess.run(
                [sys.executable, temp_file],
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
