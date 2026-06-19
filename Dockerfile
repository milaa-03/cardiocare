# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 먼저 복사 및 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 및 데이터 복사
COPY src/ ./src/
COPY data/heart.csv ./data/
COPY data/best_model.pkl ./data/

# 추론 실행
CMD ["python", "src/inference.py"]