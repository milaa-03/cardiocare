"""
모델 학습 및 MLflow 실험 추적 모듈
"""
import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import joblib

from preprocessing import build_preprocessor, load_data

RANDOM_STATE = 42
DATA_PATH = "data/heart.csv"
MODEL_PATH = "data/best_model.pkl"


def evaluate(model, X_test, y_test):
    """모델 평가 지표 계산"""
    y_pred = model.predict(X_test)
    return {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def run_experiment(name, model, X_train, X_test, y_train, y_test, params):
    """MLflow 실험 실행"""
    with mlflow.start_run(run_name=name):
        # 태그
        mlflow.set_tag("model_family", name)

        # 파라미터 기록
        mlflow.log_params(params)

        # 학습
        model.fit(X_train, y_train)

        # 교차 검증
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring="balanced_accuracy")
        mlflow.log_metric("cv_balanced_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_balanced_accuracy_std", cv_scores.std())

        # 평가
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # 혼동행렬 출력
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n[{name}] 혼동행렬:\n{cm}")
        print(f"지표: {metrics}")

        # 모델 아티팩트 저장
        mlflow.sklearn.log_model(model, "model",
            skops_trusted_types=["numpy.dtype", "numpy.ndarray"])

        return metrics, model


def main():
    # 데이터 로드 및 분할
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"학습: {X_train.shape}, 테스트: {X_test.shape}")

    preprocessor = build_preprocessor()

    mlflow.set_experiment("CardioCare")

    results = {}

    # 1. Logistic Regression
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000))
    ])
    metrics, _ = run_experiment(
        "LogisticRegression", lr_pipeline,
        X_train, X_test, y_train, y_test,
        {"model": "LogisticRegression", "max_iter": 1000}
    )
    results["LogisticRegression"] = metrics

    # 2. SVC
    preprocessor2 = build_preprocessor()
    svc_pipeline = Pipeline([
        ("preprocessor", preprocessor2),
        ("classifier", SVC(random_state=RANDOM_STATE, probability=True))
    ])
    metrics, _ = run_experiment(
        "SVC", svc_pipeline,
        X_train, X_test, y_train, y_test,
        {"model": "SVC", "kernel": "rbf"}
    )
    results["SVC"] = metrics

    # 3. Random Forest + 하이퍼파라미터 튜닝
    preprocessor3 = build_preprocessor()
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor3),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5]
    }
    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=5,
        scoring="balanced_accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    with mlflow.start_run(run_name="RandomForest_best"):
        mlflow.set_tag("model_family", "RandomForest")
        mlflow.log_params(grid_search.best_params_)
        metrics = evaluate(best_rf, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "model",
            skops_trusted_types=["numpy.dtype", "numpy.ndarray"])
        y_pred = best_rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n[RandomForest_best] 혼동행렬:\n{cm}")
        print(f"지표: {metrics}")
    results["RandomForest"] = metrics

    # 결과 비교
    print("\n=== 모델 비교 ===")
    for model_name, m in results.items():
        print(f"{model_name}: balanced_acc={m['balanced_accuracy']:.3f}, "
              f"f1={m['f1']:.3f}, recall={m['recall']:.3f}")

    # 최종 모델 저장 (F1 기준)
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\n최종 선택 모델: {best_name}")
    joblib.dump(best_rf, MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()