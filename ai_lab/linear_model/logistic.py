import numpy as np
from ai_lab.core.base import BaseEstimator, ClassifierMixin
from ai_lab.core.exceptions import NotFittedError
from ai_lab.utils.validation import check_array, check_X_y

class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression model
    타겟(y)의 클래스 개수에 따라 이진 분류(Sigmoid) 또는 다중 분류(Softmax)를 자동으로 수행함.
    """

    def __init__(
            self,
            penalty: str | None = None,
            alpha: float = 0.0,
            lr: float = 0.01,
            epochs: int = 1000,
            batch_size: int | None = None,
            random_state: int | None = None
    ):
        """
        Logistic Regression 모델의 하이퍼파라미터를 초기화.

        Args:
            penalty (str | None): 규제 종류 ('l1', 'l2', 또는 None). 기본값 None.
            alpha (float): 규제 강도. 클수록 규제가 강해짐. 기본값 0.0.
            lr (float): 경사 하강법의 학습률(Learning Rate). 기본값 0.01.
            epochs (int): 전체 데이터 학습 반복 횟수. 기본값 1000.
            batch_size (int | None): 미니배치 크기. 
                                     - None(기본값)이면 전체 데이터를 한 번에 학습(Batch GD).
                                     - 1이면 데이터 1개씩 학습(Stochastic GD).
                                     - 32, 64 등 정수면 해당 크기로 학습(Mini-batch GD).
            random_state (int | None): 데이터 섞기를 위한 난수 시드.
        """
        self.penalty = penalty
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        이진 분류를 위한 Sigmoid 활성화 함수.
        입력값(Logit)을 0과 1사이의 확률값으로 변환.

        Args:
            z (np.ndarray): (n_samples, ) Logit (가중합)

        Returns:
            np.ndarray: (n_samples, ) 형태의 0~1 사이 확률값 배열
        """
        z = np.clip(z, -250, 250) # 오버플로우 방지 (너무 크거나 작은 값 제한)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        다중 분류를 위한 Softmax 활성화 함수.
        각 클래스의 확률 합이 1(100%)이 되도록 변환.
        오버플로우 방지를 위해 각 행의 최댓값을 빼고 계산함.

        Args:
            z (np.ndarray): (n_samples, n_classes) Logit (가중합)

        Returns:
            np.ndarray (n_samples, n_classes) 형태의 각 클래스별 확률값 행렬
        """
        z = np.clip(z, -250, 250)
        # np.exp 계산시 오버플로우를 방지하기 위해, 각 행(데이터) 별로 최댓값을 빼주고 계산
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        모델을 데이터에 적합시킴. (가중치 학습)
        데이터의 클래스 개수를 파악하여 이진 분류와 다중 분류를 자동으로 선택.

        Args:
            X (np.ndarray): (n_samples, n_features) 학습할 입력 데이터
            y (np.ndarray): (n_samples,) 학습할 정답 데이터
        """

        # 입력 데이터 X, y검증 및 차원 보정
        X, y = check_X_y(X, y)
            
        # y에 있는 고유한 클래스 이름들(중복 제외)을 self.classes_에 배열로 저장.
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 클래스 개수에 따라 내부 로직 분기
        if n_classes == 2:
            self.multi_class_ = 'binary'
            self._fit_binary(X, y)
        elif n_classes > 2:
            self.multi_class_ = 'multinomial'
            self._fit_multinomial(X, y, n_classes)
        else:
            raise ValueError("클래스가 1개뿐인 데이터는 분류할 수 없습니다.")
        
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        이진 분류 (Sigmoid) 모델 학습.

        Args:
            X (np.ndarray): (n_samples, n_features) 학습할 입력 데이터
            y (np.ndarray): (n_samples,) 학습할 정답 데이터
        """

        n_samples, n_features = X.shape
        # 가중치 배열을 (n_features,) 형태로 초기화.
        self.weights_ = np.zeros(n_features) 
        self.bias_ = 0.0 # bias 초기화

        # 문자열 등의 데이터를 수학 연산이 가능하도록 0과 1로 변환
        y_binary = np.where(y == self.classes_[1], 1, 0)

        batch_size = self.batch_size if self.batch_size is not None else n_samples

        # GD 진행시 매 epoch마다 데이터를 섞어주는데, random_state 값이 주어졌다면 그 값으로 시드 고정 (실험 재현성 위해)
        # self.random_state가 None 이면 컴퓨터의 현재 시간을 시드로 무작위로 지정
        rng = np.random.RandomState(self.random_state)

        # binary 분류 GD 알고리즘 루프
        for epoch in range(self.epochs):
            # 데이터 순서 학습에 따른 편향을 막기 위해 매 epoch 마다 인덱스를 새로 섞음
            indices = rng.permutation(n_samples)

            # Fancy Indexing을 통해 섞인 인덱스를 실제 데이터 행렬에 매핑
            X_shuffled, y_shuffled = X[indices], y_binary[indices] # 변환된 y_binary 사용

            # mini-batch 학습 -> 전체 데이터를 batch_size 만큼씩 잘라서 학습 (가중치 업데이트)
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size] # (m, n_features)
                y_batch = y_shuffled[i:i + batch_size] # (m, )
                m = X_batch.shape[0] # 행의 개수 (현재 배치에서의 데이터 샘플 수)

                # Forward 가중합
                # Z = XW + b ((m, n_features) dot (n_features,) -> (m,))
                linear_output = X_batch.dot(self.weights_) + self.bias_
                
                # 예측값 (m, )
                # 가중합에 활성화 함수 Sigmoid 씌운 확률값
                y_pred_proba = self._sigmoid(linear_output)
                 
                # 오차 계산 (m, )
                error = y_pred_proba - y_batch # 오차 (m, )

                # Backward (Gradient 계산)
                # 데이터의 평균 오차를 구하기 위해 1/m을 곱해줌
                dw = (1 / m) * X_batch.T.dot(error) # 가중치 Gradient (n_features, )
                db = (1 / m) * np.sum(error) # bias Gradient 단일 숫자(float)

                # Regularization 적용 -> 과적합 방지
                # bias에는 규제를 적용하지 않기에 dw에만 패널티를 부여함
                if self.penalty == 'l2':
                    dw += self.alpha * self.weights_ # 특정 feature의 가중치가 비정상적으로 높다면 더 큰 수정을 하도록 벌점 추가
                elif self.penalty == 'l1':
                    dw += self.alpha * np.sign(self.weights_) # 불필요한 가중치를 0으로 만듦

                # 가중치 업데이트 (Gradient는 최대 증가 방향을 가리키기에 - 붙여서)
                self.weights_ -= self.lr * dw
                self.bias_ -= self.lr * db


    def _fit_multinomial(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> None:
        """
        다중 분류 (Softmax) 모델 학습.
        내부적으로 One-hot Encoding을 적용함.

        Args:
            X (np.ndarray): (n_samples, n_features) 학습할 입력 데이터
            y (np.ndarray): (n_samples,) 학습할 정답 데이터. 1차원으로 받고 이를 내부에서 one-hot-encoding으로 변환
            n_classes (int): 정답 데이터의 클래스 개수
        """
        n_samples, n_features = X.shape

        # 다중 분류는 각 클래스마다 가중치가 따로 필요하기에 2차원 가중치 행렬 구성
        # 가중치 행렬: (n_features, n_classes)
        self.weights_ = np.zeros((n_features, n_classes))
        
        # bias 배열: (n_classes, ) - 각 클래스마다 1개씩 존재
        self.bias_ = np.zeros(n_classes)

        # Target 데이터를 one-hot Encoding 행렬 (n_samples, n_classes)로 변환
        y_one_hot = np.zeros((n_samples, n_classes))
        for idx, c in enumerate(self.classes_):
            y_one_hot[y == c, idx] = 1
        
        batch_size = self.batch_size if self.batch_size is not None else n_samples

        rng = np.random.RandomState(self.random_state)

        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y_one_hot[indices] # 섞을 때 one-hot 데이터 사용

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size] # (m, n_features)
                y_batch = y_shuffled[i:i + batch_size] # (m, n_classes)
                m = X_batch.shape[0] # 행의 개수 (현재 배치에서의 데이터 샘플 수)

                # Forward 가중합
                # Z = XW + b ((m, n_features) dot (n_features, n_classes) -> (m,n_classes))
                linear_output = X_batch.dot(self.weights_) + self.bias_

                # 예측값 (m, n_classes)
                # 가중합에 활성화 함수 Softmax를 씌운 확률값 (전체 합 1)
                y_pred_probs = self._softmax(linear_output)

                # 오차 행렬 (m, n_classes)
                error = y_pred_probs - y_batch

                # Backward (Gradient 계산)
                dw = (1 / m) * X_batch.T.dot(error) # (n_features, m) dot (m, n_classes) -> (n_features, n_classes)
                db = (1 / m) * np.sum(error, axis=0) # (n_classes,)

                # Regularization
                if self.penalty == 'l2':
                    dw += self.alpha * self.weights_
                elif self.penalty == 'l1':
                    dw += self.alpha * np.sign(self.weights_)

                # 가중치 업데이트
                self.weights_ -= self.lr * dw
                self.bias_ -= self.lr * db
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        학습이 완료된 모델에 새로운 데이터 X를 넣어 각 클래스에 속할 확률을 예측해 반환.

        Args:
            X (np.ndarray): (n_samples, n_features) 예측할 입력 데이터

        Returns:
            np.ndarray: (n_samples,) or (n_samples, n_classes) 예측된 확률값 배열
        """
        # 학습되지 않은 모델(self.weights_ 부재)에 대한 예외 처리
        if not hasattr(self, 'weights_'):
            raise NotFittedError("모델이 아직 학습되지 않았습니다")
        
        # 입력 데이터 X 검증 및 변환
        X = check_array(X)

        # 가중합 계산
        # 결과값이 이진 분류면 (n_samples, ) 다중 분류면 (n_samples, n_classes)
        linear_output = X.dot(self.weights_) + self.bias_

        # 학습된 모델이 이진 분류인지 다중 분류인지 판단하여 다른 함수 적용
        if self.multi_class_ == 'binary':
            return self._sigmoid(linear_output) # 1일 확률값 배열 반환 (n_samples,)
        else:
            return self._softmax(linear_output) # 각 클래스에 속할 확률값 배열 반환 (n_samples, n_classes)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict_proba를 바탕으로 가장 확률이 높은 클래스의 이름을 반환.

        Args:
            X (np.ndarray): (n_samples, n_features) 예측할 입력 데이터
            
        Returns:
            np.ndarray: (n_samples,) 최종 예측된 클래스 배열
        """
        proba = self.predict_proba(X)

        if self.multi_class_ == 'binary':
            # 0.5 이상이면 1, 아니면 0
            # _fit_binary에서 self.classes_[1]을 1로 매핑했으므로 이 로직이 정확히 맞음
            indices = (proba >= 0.5).astype(int)
        else:
            # 확률 행렬에서 행(axis=1)마다 가장 값이 큰 열의 인덱스 번호 추출
            indices = np.argmax(proba, axis=1)

        # 내부 인덱스를 실제 클래스 이름으로 변환하여 반환
        return self.classes_[indices]
