"""
전처리 파이프라인 모듈
노트북 EDA 결과를 바탕으로 재사용 가능한 파이프라인 정의
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42

# 연속형 / 범주형 컬럼 정의
CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def build_preprocessor():
    """전처리 파이프라인 생성 및 반환"""

    continuous_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer([
        ("continuous", continuous_pipeline, CONTINUOUS_COLS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS)
    ])

    return preprocessor


def load_data(path: str):
    """CSV 데이터 로드 후 X, y 분리"""
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


if __name__ == "__main__":
    X, y = load_data("data/heart.csv")
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    print(f"전처리 완료: {X_transformed.shape}")