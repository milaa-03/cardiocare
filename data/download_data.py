"""
UCI Heart Disease 데이터셋 다운로드 스크립트
출처: https://archive.ics.uci.edu/dataset/45/heart+disease
"""
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "heart.csv")


def download_and_save():
    print("데이터 다운로드 중...")
    heart = fetch_ucirepo(id=45)

    X = heart.data.features
    y = heart.data.targets

    df = pd.concat([X, y], axis=1)

    # 타깃 이진화: 0 = 정상, 1 = 심장병
    target_col = y.columns[0]
    df[target_col] = (df[target_col] > 0).astype(int)
    df = df.rename(columns={target_col: "target"})

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"행: {len(df)}, 열: {len(df.columns)}")
    print(df["target"].value_counts(normalize=True).rename("비율"))


if __name__ == "__main__":
    download_and_save()