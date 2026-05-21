"""
EDA 결과를 HTML 리포트로 생성하는 스크립트
이미지 파일을 포함하여 하나의 HTML 파일로 통합합니다.
"""

import os
import re
from datetime import datetime
from pathlib import Path

def markdown_to_html(md_text):
    """간단한 마크다운을 HTML로 변환"""
    html = md_text
    
    # 헤더 변환
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # 볼드
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # 리스트
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(\d+)\. (.+)$', r'<li>\2</li>', html, flags=re.MULTILINE)
    
    # 코드 블록
    html = re.sub(r'```python\n(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```\n(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    
    # 인라인 코드
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # 줄바꿈
    html = html.replace('\n', '<br>\n')
    
    # 리스트 감싸기
    html = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', html, flags=re.DOTALL)
    html = html.replace('</ul>\n<br>\n<ul>', '')
    
    return html

def read_markdown_file(file_path):
    """마크다운 파일을 읽어서 HTML로 변환하여 반환"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
            return markdown_to_html(md_content)
    except Exception as e:
        return f"<p>파일을 읽을 수 없습니다: {str(e)}</p>"

def get_image_path(image_name):
    """이미지 파일의 상대 경로 반환"""
    return f"시각화/{image_name}"

def generate_html_report():
    """HTML 리포트 생성"""
    
    # HTML 헤더 및 스타일
    html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Train CSV EDA 분석 리포트</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Malgun Gothic', '맑은 고딕', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-left: 10px;
            border-left: 4px solid #3498db;
        }}
        
        h3 {{
            color: #555;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        .executive-summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
            border-left: 5px solid #3498db;
        }}
        
        .executive-summary ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}
        
        .executive-summary li {{
            margin-bottom: 8px;
        }}
        
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 5px;
        }}
        
        .section h2 {{
            margin-top: 0;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        
        .image-container img {{
            max-width: 90%;
        }}
        
        .image-title {{
            font-weight: bold;
            color: #555;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        table th, table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        table th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        
        table tr:hover {{
            background-color: #f5f5f5;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #e74c3c;
        }}
        
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
        }}
        
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}
        
        .toc {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        
        .toc ul {{
            list-style-type: none;
            margin-left: 20px;
        }}
        
        .toc li {{
            margin: 8px 0;
        }}
        
        .toc a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}
        
        .recommendations {{
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin-top: 30px;
            border-radius: 5px;
        }
        
        .recommendations h3 {{
            color: #856404;
            margin-top: 0;
        }
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Train CSV 심층 EDA 분석 리포트</h1>
        <p style="color: #7f8c8d; margin-bottom: 30px;">
            생성 시간: {generation_time}
        </p>
        
        <div class="toc">
            <h2>목차</h2>
            <ul>
                <li><a href="#executive-summary">1. 실행 요약</a></li>
                <li><a href="#data-overview">2. 데이터 개요</a></li>
                <li><a href="#target-analysis">3. 타겟 변수 분석</a></li>
                <li><a href="#categorical-analysis">4. 범주형 변수 분석</a></li>
                <li><a href="#missing-analysis">5. 결측치 분석</a></li>
                <li><a href="#feature-analysis">6. 특징 변수 분석</a></li>
                <li><a href="#outlier-analysis">7. 이상치 탐지</a></li>
                <li><a href="#correlation-analysis">8. 상관관계 분석</a></li>
                <li><a href="#group-analysis">9. 그룹별 심층 분석</a></li>
                <li><a href="#recommendations">10. 모델링 권장사항</a></li>
            </ul>
        </div>
        
        {content}
        
        <div class="footer">
            <p>EDA 분석 리포트 | 생성일: {generation_time}</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 실행 요약 읽기
    executive_summary = read_markdown_file('../docs/개발일지/EDA_최종리포트.md')
    
    # 각 단계별 내용 읽기
    sections = {}
    step_files = {
        'data-overview': '단계1_기본데이터정보분석.md',
        'target-analysis': '단계2_타겟변수분석.md',
        'categorical-analysis': '단계3_범주형변수분석.md',
        'missing-analysis': '단계4_결측치분석.md',
        'feature-analysis': '단계5_특징변수분석.md',
        'outlier-analysis': '단계6_이상치탐지.md',
        'correlation-analysis': '단계7_상관관계분석.md',
        'group-analysis': '단계8_그룹별심층분석.md',
    }
    
    for section_id, filename in step_files.items():
        filepath = f'../docs/개발일지/{filename}'
        if os.path.exists(filepath):
            sections[section_id] = read_markdown_file(filepath)
    
    # 이미지 파일 매핑
    image_mapping = {
        'target-analysis': '2_타겟변수분석.png',
        'categorical-analysis': '3_범주형변수분석.png',
        'missing-analysis': '4_결측치분석.png',
        'feature-analysis': '5_특징변수분석.png',
        'correlation-analysis': '7_상관관계분석.png',
        'group-analysis': '8_그룹별심층분석.png',
    }
    
    # 실행 요약 HTML 생성
    exec_html = f"""
        <div id="executive-summary" class="executive-summary">
            {markdown_to_html(executive_summary)}
        </div>
    """
    
    # 각 섹션 HTML 생성
    sections_html = []
    
    # 데이터 개요
    if 'data-overview' in sections:
        sections_html.append(f"""
            <div id="data-overview" class="section">
                <h2>2. 데이터 개요</h2>
                {sections['data-overview']}
            </div>
        """)
    
    # 타겟 변수 분석
    if 'target-analysis' in sections:
        img_path = get_image_path(image_mapping.get('target-analysis', ''))
        sections_html.append(f"""
            <div id="target-analysis" class="section">
                <h2>3. 타겟 변수 분석</h2>
                {sections['target-analysis']}
                <div class="image-container">
                    <div class="image-title">타겟 변수 분포 시각화</div>
                    <img src="{img_path}" alt="타겟 변수 분석">
                </div>
            </div>
        """)
    
    # 범주형 변수 분석
    if 'categorical-analysis' in sections:
        img_path = get_image_path(image_mapping.get('categorical-analysis', ''))
        sections_html.append(f"""
            <div id="categorical-analysis" class="section">
                <h2>4. 범주형 변수 분석</h2>
                {sections['categorical-analysis']}
                <div class="image-container">
                    <div class="image-title">범주형 변수 분석 시각화</div>
                    <img src="{img_path}" alt="범주형 변수 분석">
                </div>
            </div>
        """)
    
    # 결측치 분석
    if 'missing-analysis' in sections:
        img_path = get_image_path(image_mapping.get('missing-analysis', ''))
        sections_html.append(f"""
            <div id="missing-analysis" class="section">
                <h2>5. 결측치 분석</h2>
                {sections['missing-analysis']}
                <div class="image-container">
                    <div class="image-title">결측치 분석 시각화</div>
                    <img src="{img_path}" alt="결측치 분석">
                </div>
            </div>
        """)
    
    # 특징 변수 분석
    if 'feature-analysis' in sections:
        img_path = get_image_path(image_mapping.get('feature-analysis', ''))
        sections_html.append(f"""
            <div id="feature-analysis" class="section">
                <h2>6. 특징 변수 분석</h2>
                {sections['feature-analysis']}
                <div class="image-container">
                    <div class="image-title">특징 변수 분석 시각화</div>
                    <img src="{img_path}" alt="특징 변수 분석">
                </div>
            </div>
        """)
    
    # 이상치 탐지
    if 'outlier-analysis' in sections:
        sections_html.append(f"""
            <div id="outlier-analysis" class="section">
                <h2>7. 이상치 탐지</h2>
                {sections['outlier-analysis']}
            </div>
        """)
    
    # 상관관계 분석
    if 'correlation-analysis' in sections:
        img_path = get_image_path(image_mapping.get('correlation-analysis', ''))
        sections_html.append(f"""
            <div id="correlation-analysis" class="section">
                <h2>8. 상관관계 분석</h2>
                {sections['correlation-analysis']}
                <div class="image-container">
                    <div class="image-title">상관관계 분석 시각화</div>
                    <img src="{img_path}" alt="상관관계 분석">
                </div>
            </div>
        """)
    
    # 그룹별 심층 분석
    if 'group-analysis' in sections:
        img_path = get_image_path(image_mapping.get('group-analysis', ''))
        sections_html.append(f"""
            <div id="group-analysis" class="section">
                <h2>9. 그룹별 심층 분석 (LINE × PRODUCT_CODE)</h2>
                {sections['group-analysis']}
                <div class="image-container">
                    <div class="image-title">그룹별 심층 분석 시각화</div>
                    <img src="{img_path}" alt="그룹별 심층 분석">
                </div>
            </div>
        """)
    
    # 모델링 권장사항
    recommendations_html = """
        <div id="recommendations" class="recommendations">
            <h3>모델링 권장사항</h3>
            <ol>
                <li><strong>결측치 처리</strong>
                    <ul>
                        <li>결측치가 50% 이상인 컬럼은 제거 고려</li>
                        <li>결측치 패턴 분석 후 적절한 대체 방법 선택 (평균, 중앙값, KNN 등)</li>
                        <li>결측치 자체가 정보를 담고 있을 수 있으므로 결측치 표시 변수 생성 고려</li>
                    </ul>
                </li>
                <li><strong>특징 선택</strong>
                    <ul>
                        <li>결측치가 적고 타겟 변수와 상관관계가 높은 변수 우선 선택</li>
                        <li>차원 축소 기법 (PCA, Feature Selection) 고려</li>
                        <li>상수/준상수 변수 제거 (597개)</li>
                    </ul>
                </li>
                <li><strong>클래스 불균형 처리</strong>
                    <ul>
                        <li>Y_Class 불균형 문제 해결 (SMOTE, 가중치 조정 등)</li>
                        <li>클래스 1이 67.7%로 다수를 차지하므로 적절한 샘플링 전략 필요</li>
                    </ul>
                </li>
                <li><strong>그룹별 모델링</strong>
                    <ul>
                        <li>LINE × PRODUCT_CODE 조합별로 별도 모델 학습 고려</li>
                        <li>조합별 품질 특성이 다르므로 그룹별 특화 모델이 효과적일 수 있음</li>
                    </ul>
                </li>
                <li><strong>이상치 처리</strong>
                    <ul>
                        <li>변수별 분포 특성을 고려하여 신중하게 판단</li>
                        <li>일괄 제거보다는 도메인 지식 기반 판단 필요</li>
                        <li>이상치가 중요한 정보를 담고 있을 수 있으므로 제거 전후 성능 비교</li>
                    </ul>
                </li>
            </ol>
        </div>
    """
    
    # 전체 내용 합치기
    content = exec_html + '\n'.join(sections_html) + recommendations_html
    
    # HTML 생성
    generation_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")
    # CSS 중괄호를 단일 중괄호로 변환 (이중 중괄호 -> 단일 중괄호)
    html_template_fixed = html_template.replace('{{', '{').replace('}}', '}')
    # 내용 삽입
    html_content = html_template_fixed.replace('{content}', content).replace('{generation_time}', generation_time)
    
    # HTML 파일 저장
    output_path = '../docs/개발일지/EDA_분석리포트.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML 리포트 생성 완료: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_html_report()

