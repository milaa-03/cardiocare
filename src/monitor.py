"""
모니터링 및 드리프트 탐지 모듈
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score
import joblib

logging.basicConfig(
    filename="monitor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = "data/best_model.pkl"
DATA_PATH = "data/heart.csv"
RANDOM_STATE = 42
CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def simulate_drift(X_test: pd.DataFrame) -> pd.DataFrame:
    """chol 분포를 인위적으로 이동시켜 드리프트 시뮬레이션"""
    X_drifted = X_test.copy()
    np.random.seed(RANDOM_STATE)
    X_drifted["chol"] = X_drifted["chol"] + 80 + np.random.normal(0, 50, len(X_drifted))
    X_drifted["trestbps"] = X_drifted["trestbps"] + 30 + np.random.normal(0, 15, len(X_drifted))
    return X_drifted


def detect_drift(X_train: pd.DataFrame, X_test_drifted: pd.DataFrame):
    """KS 검정으로 드리프트 탐지"""
    print("\n=== 드리프트 탐지 결과 (KS 검정) ===")
    drift_results = {}

    for col in CONTINUOUS_COLS:
        stat, p_value = ks_2samp(X_train[col].dropna(), X_test_drifted[col].dropna())
        flagged = p_value < 0.05
        drift_results[col] = {"statistic": stat, "p_value": p_value, "flagged": flagged}
        status = "❌ 드리프트 감지!" if flagged else "✅ 정상"
        print(f"{col}: p={p_value:.4f} {status}")
        logger.info(f"KS test {col}: stat={stat:.4f}, p={p_value:.4f}, flagged={flagged}")

    return drift_results


def compare_accuracy(model, X_test, y_test, X_drifted):
    """원본 vs 드리프트 데이터 정확도 비교"""
    y_pred_orig = model.predict(X_test)
    y_pred_drift = model.predict(X_drifted)

    acc_orig = balanced_accuracy_score(y_test, y_pred_orig)
    acc_drift = balanced_accuracy_score(y_test, y_pred_drift)

    print(f"\n=== 성능 비교 ===")
    print(f"원본 테스트셋 Balanced Accuracy:  {acc_orig:.4f}")
    print(f"드리프트 테스트셋 Balanced Accuracy: {acc_drift:.4f}")
    print(f"성능 하락: {acc_orig - acc_drift:.4f}")

    logger.info(f"원본 accuracy={acc_orig:.4f}, 드리프트 accuracy={acc_drift:.4f}")

    return acc_orig, acc_drift


def plot_drift_timeline(acc_orig: float, acc_drift: float):
    """시간에 따른 성능 변화 시계열 그래프"""
    timestamps = list(range(1, 11))
    # 합성 타임스탬프로 점진적 드리프트 시뮬레이션
    accuracies = [acc_orig] * 5 + [
        acc_orig - (acc_orig - acc_drift) * i / 5
        for i in range(1, 6)
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, accuracies, marker="o", color="steelblue", linewidth=2)
    plt.axhline(y=acc_orig, color="green", linestyle="--", label="기준 성능")
    plt.axhline(y=acc_drift, color="red", linestyle="--", label="드리프트 후 성능")
    plt.axvline(x=5.5, color="orange", linestyle="--", label="드리프트 발생 시점")
    plt.title("시간에 따른 Balanced Accuracy 변화")
    plt.xlabel("시간 (배포 후 주차)")
    plt.ylabel("Balanced Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/drift_timeline.png")
    plt.show()
    print("그래프 저장 완료: data/drift_timeline.png")


def main():
    from sklearn.model_selection import train_test_split

    # 데이터 로드
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 모델 로드
    model = joblib.load(MODEL_PATH)

    # 드리프트 시뮬레이션
    X_drifted = simulate_drift(X_test)

    # KS 검정
    detect_drift(X_train, X_drifted)

    # 성능 비교
    acc_orig, acc_drift = compare_accuracy(model, X_test, y_test, X_drifted)

    # 시계열 그래프
    plot_drift_timeline(acc_orig, acc_drift)


if __name__ == "__main__":
    main()