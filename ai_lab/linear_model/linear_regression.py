import numpy as np
from ai_lab.core.base import BaseEstimator, RegressorMixin
from ai_lab.core.exceptions import NotFittedError
from ai_lab.utils.validation import check_array, check_X_y

class LinearRegression(BaseEstimator, RegressorMixin):
    """
    Linear Regression model
    Gradien Descent를 사용하여 최적화.
    """

    def __init__(
        self, 
        penalty: str | None = None, 
        alpha: float = 0.01, 
        lr: float = 0.01, 
        epochs: int = 1000, 
        batch_size: int | None = None,
        random_state: int | None = None
    ):
        """
        Linear Regression 모델의 하이퍼파라미터를 초기화.

        Args:
            penalty (str | None): 'l1' (Lasso), 'l2' (Ridge), 또는 None (규제 없음)
            alpha (float): 규제 강도 (penalty가 None이 아닐 때 적용)
            lr (float): 학습률 (Learning Rate) - GD 방식에서만 사용
            epochs (int): 전체 데이터 학습 반복 횟수 - GD 방식에서만 사용
            batch_size (int | None): 미니배치 크기. None이면 전체 배치(Batch GD), 1이면 확률적(SGD)
            random_state (int | None): 미니배치 시 데이터 셔플링을 위한 난수 시드
        """

        self.penalty = penalty
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        모델을 데이터에 적합시킴. (GD를 통한 가중치 학습)

        Args:
            X (np.ndarray): (n_samples, n_features) 학습할 입력 데이터
            y (np.ndarray): (n_samples, ) 학습할 정답 데이터
        """
        # 입력 데이터 X, y검증 및 차원 보정
        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape
        # 가중치 배열을 (n_features, ) 형태로 초기화
        # forward 진행시 X dot W 연산을 통해 feature별로 가중치를 곱해 더하기 위함
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0 # bias 초기화

        batch_size = self.batch_size if self.batch_size is not None else n_samples

        rng = np.random.RandomState(self.random_state)

        for epoch in range(self.epochs):
            # 매 epoch마다 데이터 순서에 따른 학습 편향을 막기 위해 인덱스를 무작위로 섞음
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices] # Fancy Indexing
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                m = X_batch.shape[0]

                # 예측 및 오차 계산
                y_pred = X_batch.dot(self.weights_) + self.bias_
                error = y_pred - y_batch

                # Gradient 계산
                dw = (1 / m) * X_batch.T.dot(error)
                db = (1 / m) * np.sum(error)

                # Regularization
                if self.penalty == 'l2':
                    dw += self.alpha * self.weights_
                elif self.penalty == 'l1':
                    dw += self.alpha * np.sign(self.weights_)

                # 가중치 업데이트
                self.weights_ -= self.lr * dw
                self.bias_ -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측값 반환.
        
        Args:
            X (np.ndarray): (n_samples, n_features) 예측할 입력 데이터

        Returns:
            np.ndarray: (n_samples, ) 각 데이터 샘플에 대한 예측 결과값이 담긴 배열
        """
        if not hasattr(self, 'weights_') or not hasattr(self, 'bias_'):
            raise NotFittedError("모델이 아직 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        X = check_array(X)
        
        # Y = XW + b
        return X.dot(self.weights_) + self.bias_