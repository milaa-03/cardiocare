# CardioCare 🫀
심장병 예측을 위한 종단간 머신러닝 시스템

## 프로젝트 개요
UCI Heart Disease 데이터셋을 활용하여 심장병 발병 가능성을 예측하는 ML 시스템입니다.
심장 전문의의 의사결정을 보조하는 도구이며, 단독으로 결정하지 않습니다.

## 재현 절차

### 1. 저장소 클론
```bash
git clone https://github.com/milaa-03/cardiocare.git
cd cardiocare
```

### 2. 가상환경 설정 및 의존성 설치
```bash
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### 3. 데이터 다운로드
```bash
python data/download_data.py
```

### 4. 모델 학습 (MLflow 실험 추적 포함)
```bash
python src/train.py
```

### 5. 단위 테스트 실행
```bash
python -m unittest tests/test_pipeline.py -v
```

### 6. 추론 실행
```bash
python src/inference.py
```

### 7. 드리프트 모니터링
```bash
python src/monitor.py
```

### 8. Docker 빌드 및 실행
```bash
docker build -t cardiocare:1.0 .
docker run cardiocare:1.0
```

## 모델 성능
| 모델 | Balanced Accuracy | F1 | Recall |
|------|------------------|-----|--------|
| Logistic Regression | 0.873 | 0.867 | 0.929 |
| SVC | 0.823 | 0.814 | 0.857 |
| **Random Forest** | **0.919** | **0.912** | **0.929** |

## 프로젝트 구조
```
cardiocare/
├── data/               # 데이터 및 모델
├── notebooks/          # EDA 노트북
├── src/                # 소스코드
├── tests/              # 단위 테스트
├── Dockerfile
├── requirements.txt
└── .github/workflows/  # CI 설정
```