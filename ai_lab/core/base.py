# 모든 모델은 똑같은 인터페이스 (fit, predict)를 가짐
# Scikit-learn 구조 벤치마킹함

import numpy as np
from ai_lab.metrics.regression import r2_score
from ai_lab.metrics.classification import accuracy_score

class BaseEstimator:
    """
    모든 머신러닝 모델의 최상위 부모 클래스.
    Scikit-learn의 구조를 벤치마킹함.
    하이퍼파라미터는 __init__에서 받고, 데이터 (X,y)는 fit 메서드에서만 처리하도록 설계함.
    """

    def fit(self, X : np.ndarray, y: np.ndarray | None = None):
        """
        학습 데이터(X)와 정답(y)를 받아 모델을 학습시킴.

        Args:
            X (np.ndarray): (n_samples, n_features) 학습 데이터
            y (np.ndarray, optional): (n_samples,) 타겟 데이터. 비지도 학습 시 None

        Returns:
            self: 학습된 모델 인스턴스 자신
        """

        raise NotImplementedError("자식 클래스에서 fit 메서드를 반드시 구현해야 합니다.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        새로운 데이터(X)에 대한 예측값을 반환함.
        
        Args:
            X (np.ndarray): (n_samples, n_features) 예측할 데이터

        Returns:
            np.ndarray: (n_samples,) 예측 결과값
        """
        raise NotImplementedError("자식 클래스에서 predict 메서드를 반드시 구현해야 합니다.")
    
class RegressorMixin:
    """
    Regression 모델에 공통으로 들어갈 기능을 제공하는 Mixin 클래스
    """
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        회귀 모델의 성능을 평가함 (R^2 Score 결정계수 사용).

        Args:
            X (np.ndarray): (n_samples, n_features) 평가할 입력 데이터, 형태
            y (np.ndarray): (n_samples,) 평가할 실제 정답 데이터, 형태

        Returns:
            float: R^2 Score. 1.0에 가까울수록 완벽한 예측 의미
        """

        y_pred = self.predict(X)

        return r2_score(y, y_pred)
    
class ClassifierMixin:
    """
    Classification 모델에 공통으로 포함되는 기능을 제공하는 Mixin 클래스
    """

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        분류 모델의 성능을 평가함 (Accuracy 정확도 사용).

        Args:
            X (np.ndarray): (n_samples, n_features) 평가할 입력 데이터, 형태
            y (np.ndarray): (n_samples,) 평가할 실제 정답 데이터, 형태

        Returns:
            float: 정확도 (Accuracy). 0.0 ~ 1.0 사이 값
        """

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)