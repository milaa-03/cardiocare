"""
파이프라인 단위 테스트 - 최소 4개 테스트 메서드
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import build_preprocessor, load_data
from inference import load_model, predict, validate_input

RANDOM_STATE = 42
DATA_PATH = "data/heart.csv"
MODEL_PATH = "data/best_model.pkl"


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """테스트 전 공통 준비"""
        cls.X, cls.y = load_data(DATA_PATH)
        cls.model = load_model(MODEL_PATH)
        cls.sample = pd.DataFrame([{
            "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
            "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
            "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
        }])

    def test_prediction_shape(self):
        """테스트 1: 예측 결과 shape가 입력 shape와 일치하는지"""
        y_pred, y_prob = predict(self.model, self.X)
        self.assertEqual(y_pred.shape[0], self.X.shape[0])
        self.assertEqual(y_prob.shape[0], self.X.shape[0])
        print("테스트 1 통과: prediction shape 일치")

    def test_probability_range(self):
        """테스트 2: 예측 확률이 [0,1] 범위이고 행마다 합이 1인지"""
        _, y_prob = predict(self.model, self.X)
        self.assertTrue((y_prob >= 0).all())
        self.assertTrue((y_prob <= 1).all())
        prob_sums = y_prob.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, 1.0, decimal=5)
        print("테스트 2 통과: 확률 범위 정상")

    def test_input_validation(self):
        """테스트 3: 입력값 범위 검증 (chol [0,600])"""
        # 정상 입력
        self.assertTrue(validate_input(self.sample))

        # 비정상 입력
        bad_sample = self.sample.copy()
        bad_sample["chol"] = 999
        with self.assertRaises(ValueError):
            validate_input(bad_sample)
        print("테스트 3 통과: 입력값 범위 검증 정상")

    def test_deterministic_output(self):
        """테스트 4: 고정 시드에서 동일 입력 -> 동일 출력"""
        y_pred1, y_prob1 = predict(self.model, self.sample)
        y_pred2, y_prob2 = predict(self.model, self.sample)
        np.testing.assert_array_equal(y_pred1, y_pred2)
        np.testing.assert_array_almost_equal(y_prob1, y_prob2)
        print("테스트 4 통과: 결정론적 출력 확인")


if __name__ == "__main__":
    unittest.main(verbosity=2)