"""
추론 모듈 - 저장된 모델로 예측 수행
"""
import os
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    filename="inference.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = "data/best_model.pkl"
MODEL_VERSION = "1.0"


def load_model(model_path: str = MODEL_PATH):
    """저장된 모델 로드"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"모델 로드 완료: {model_path}, 버전: {MODEL_VERSION}")
    return model


def predict(model, X: pd.DataFrame):
    """예측 수행 및 로깅"""
    logger.info(f"추론 시작 - 입력 shape: {X.shape}, 모델 버전: {MODEL_VERSION}")

    # 예측
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    logger.info(f"예측 완료 - 예측값: {y_pred.tolist()}")

    return y_pred, y_prob


def validate_input(X: pd.DataFrame):
    """입력값 범위 검증"""
    errors = []

    if "chol" in X.columns:
        if not X["chol"].between(0, 600).all():
            errors.append("chol 값이 [0, 600] 범위를 벗어남")

    if "age" in X.columns:
        if not X["age"].between(0, 120).all():
            errors.append("age 값이 [0, 120] 범위를 벗어남")

    if "trestbps" in X.columns:
        if not X["trestbps"].between(0, 300).all():
            errors.append("trestbps 값이 [0, 300] 범위를 벗어남")

    if errors:
        raise ValueError(f"입력값 검증 실패: {errors}")

    return True


if __name__ == "__main__":
    # 샘플 입력으로 테스트
    sample = pd.DataFrame([{
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
    }])

    model = load_model()
    validate_input(sample)
    y_pred, y_prob = predict(model, sample)

    print(f"예측 결과: {'심장병' if y_pred[0] == 1 else '정상'}")
    print(f"확률: 정상={y_prob[0][0]:.3f}, 심장병={y_prob[0][1]:.3f}")