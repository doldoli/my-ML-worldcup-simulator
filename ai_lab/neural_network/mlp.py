import numpy as np
from ai_lab.core.base import BaseEstimator, ClassifierMixin
from ai_lab.core.exceptions import NotFittedError
from ai_lab.utils.validation import check_array, check_X_y

class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    다층 퍼셉트론 분류기 (Multi-Layer Perceptron)
    타겟(y)의 클래스 개수에 따라 이진 분류(Sigmoid)와 다중 분류(Softmax)를 자동 수행.
    """
    def __init__(
        self, 
        hidden_layer_sizes: tuple[int, ...] = (100,),
        activation: str = 'relu',
        penalty: str | None = None, 
        alpha: float = 0.01,
        lr: float = 0.01,
        epochs: int = 1000, 
        batch_size: int | None = None,
        random_state: int | None = None
    ):
        """
        MLP 모델의 하이퍼파라미터를 초기화합니다.

        Args:
            hidden_layer_sizes (tuple): 각 은닉층의 뉴런 개수를 담은 튜플. 은닉층 수와 뉴런개수 동시 지정 가능 (예: (64, 32) -> 2개 층)
            activation (str): 은닉층 활성화 함수 ('relu' 또는 'tanh'). 기본값 'relu'.
            penalty (str | None): 규제 종류 ('l1', 'l2', 또는 None).
            alpha (float): 규제 강도. 기본값 0.01
            lr (float): 학습률(Learning Rate). 기본값 0.01
            epochs (int): 학습 반복 횟수. 기본값 1000
            batch_size (int | None): 미니배치 크기. None이면 Batch GD.
            random_state (int | None): 가중치 초기화 및 데이터 섞기에 사용할 난수 시드.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation.lower()
        self.penalty = penalty
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    # ---은닉층 활성화 함수 모음---
    def _relu(self, Z: np.ndarray) -> np.ndarray:
        # Z는 가중합 배열 (배치 크기, 현재 층의 뉴런 수)
        return np.maximum(0, Z)

    def _relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        # Boolean 배열을 float64로 변환하여 미분값 반환
        return (Z > 0).astype(np.float64)
    
    def _tanh(self, Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)
    
    def _tanh_derivative(self, Z: np.ndarray) -> np.ndarray:
        # tanh 함수 미분 공식: 1- tanh(x)^2
        return 1.0 - np.tanh(Z)**2

    # ---출력층 활성화 함수 모음---
    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        Z = np.clip(Z, -250, 250)
        return 1 / (1 + np.exp(-Z))

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        Z = np.clip(Z, -250, 250)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        순전파(Forward)와 오차 역전파(Backward)를 통해 가중치 학습.

        Args:
            X (np.ndarray): (n_samples, n_features) 학습할 입력 특성 데이터
            y (np.ndarray): (n_samples, ) 학습할 정답 데이터
        """
        # 입력 데이터 검증 및 차원 보정
        X, y = check_X_y(X, y)

        # 하이퍼파라미터로 입력받은 활성화 함수에 따라 학습에 쓸 내부 함수 매핑
        if self.activation == 'relu':
            self._act = self._relu
            self._act_deriv = self._relu_derivative
        elif self.activation == 'tanh':
            self._act = self._tanh
            self._act_deriv = self._tanh_derivative
        else:
            raise ValueError(f"지원하지 않는 활성화 함수입니다: {self.activation}")

        # y에 있는 고유한 클래스 이름들(중복 제외)을 self.classes_에 배열로 저장.
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_) # 클래스 개수
        n_samples, n_features = X.shape

        # Target 데이터의 구조 세팅 (이진 분류 vs 다중 분류)
        if n_classes == 2:
            self.multi_class_ = 'binary'
            output_neurons = 1 # 이진 분류는 출력 뉴런이 1개만 있으면 됨 (0,5보다 크면 1 아니면 0으로 판단)

            # 오차 계산을 위해 (n_samples, 1) 차원으로 변환
            Y_formatted = np.where(y == self.classes_[1], 1, 0).reshape(-1, 1)
        elif n_classes > 2:
            self.multi_class_ = 'multinomial'
            output_neurons = n_classes # 다중 분류는 각 클래스일 확률을 출력하기 위해 클래스 개수만큼의 출력 뉴런 필요

            # 다중 분류의 경우 정답을 one-hot Encoding 행렬로 변환 (n_samples, n_classes)
            Y_formatted = np.zeros((n_samples, n_classes))
            for idx, c in enumerate(self.classes_):
                # Boolean Indexing을 통해 현재 idx(열)의 true 부분을 1로 바꿈. 결과적으로 n개의 샘플 중 idx번 클래스가 정답인 샘플만 1로 바뀜
                Y_formatted[y == c, idx] = 1
        else:
            raise ValueError("클래스가 1개뿐인 데이터는 분류할 수 없습니다.")

        # 레이어 구성 및 가중치 초기화
        
        # 리스트 연산을 통해 전체 층의 크기를 하나의 리스트로 만듦: [입력 노드 개수(특성 수), 은닉1, 은닉2, ..., 출력 노드 개수]
        # 각 층의 가중치 행렬을 생성하기 위함
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [output_neurons]

        # 각 층의 가중치와 bias를 담을 빈 리스트 생성
        # self.weights_[l]은 l층 노드와 l+1층 노드를 연결하는 edges
        self.weights_ = []
        self.biases_ = []

        # i번쨰 층과 (i+1)번째 층을 연결하는 가중치 행렬 생성
        for i in range(len(layer_sizes) - 1):
            # 가중치 초기화 (대칭성 깨고 기울기 소실 방지를 위해 표준정규분포 * 0.01 로 초기화)
            # 현재 층의 노드 수를 a, 다음 층의 노드 수를 b라고 할 때 가중치 행렬 W의 차원은 R^(a*b)
            # (현재 층 노드 수, 다음 층 노드 수)
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros(layer_sizes[i+1]) # 다음 층의 각 뉴런마다 하나씩 더해지는 상수 (bias) (다음 층 노드 수, )

            # 리스트에 Numpy 배열 하나씩 추가. 반복문 종료 후 self.weights_[i]에는 i번째 층의 가중치 행렬이 저장. 순전파 연산 시 인덱스로 접근하기 위함
            self.weights_.append(W)
            self.biases_.append(b)

        batch_size = self.batch_size if self.batch_size is not None else n_samples

        # GD 진행시 매 epoch마다 데이터를 섞어주는데, random_state 값이 주어졌다면 그 값으로 시드 고정 (실험 재현성 위해)
        # self.random_state가 None 이면 컴퓨터의 현재 시간을 시드로 무작위로 지정
        rng = np.random.RandomState(self.random_state)

        # mini-batch 학습 루프
        for epoch in range(self.epochs):
            # 매 epoch마다 순서에 따른 학습 편향을 막기 위해 인덱스를 무작위로 섞음
            # 원본 데이터는 놔두고 인덱스를 섞어서 X, y의 짝이 유지되도록 함
            indices = rng.permutation(n_samples)
            X_shuffled, Y_shuffled = X[indices], Y_formatted[indices]
            # Fancy Indexing으로 섞인 인덱스 배열을 실제 데이터에 매핑해 데이터를 섞음

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]

                # Forward로 예측값(activations)과 가중합(Zs)을 구하고, Backpropagation로 가중치를 업데이트 함.
                activations, Zs = self._forward(X_batch)
                self._backward(Y_batch, activations, Zs)

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        입력 데이터를 모든 층에 순서대로 통과시켜 예측값을 계산함. (Forward Propagation)
        l층 -> l+1층

        Args:
            X (np.ndarray): (n_samples, n_features) 미니배치 입력 데이터

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]:
                - activations: 각 층의 활성화 함수 출력값(A) 리스트
                - Zs: 각 층의 활성화 함수 통과 전 가중합(Z) 리스트
        """

        # 각 층을 통과한 후의 출력값 (활성화함수 통과한 값)을 순서대로 저장할 리스트. 
        # 역전파 단계에서 가중치 행렬을 업데이트하기 위한 Gradient를 구할 때 들어온 입력값(x_l)이 필요하기에 이를 기록해야 함.
        # 첫 입력 데이터 X도 필요하기에 이를 0번째 층의 출력값으로 담아둠.
        activations = [X]
        
        # 활성화 함수를 통과하기 전의 각 층의 가중합 행렬(Z)들을 저장할 빈 리스트. 
        # 역전파 단계에서 오차를 계산할 때, 현재 층의 활성화 함수 미분값(ex: _relu_derivative(Z))이 필요함
        # 따라서 활성화 함수를 통과하기 전 가중합 행렬 Z를 층별로 기록해야 함.
        Zs = [] 

        A = X # 연산에 들어갈 l층의 출력값 (초기값은 0층인 입력 데이터 X)

        L = len(self.weights_) # 전체 가중치 행렬의 개수 (은닉층 개수 + 1(출력층))

        # 은닉층들 통과
        # 출력층은 다른 활성화 함수를 사용해야 하기에 출력층 바로 전 은닉층 까지만 반복문 실행
        for l in range(L - 1):

            # Z = AW + b ((m, l층 노드 수) * (l층 노드수, l+1층 노드수) -> (m, l+1층 노드수))
            Z = np.dot(A, self.weights_[l]) + self.biases_[l]

            # l+1층의 가중합(Z)을 은닉층 활성화 함수에 통과시킴
            # A는 l+1층의 출력값으로 업데이트 됨
            A = self._act(Z)

            # 역전파 계산을 위해 l+1층의 가중합(Z)과 출력값(A)을 리스트에 추가
            Zs.append(Z)
            activations.append(A)

        # 마지막 출력층 통과 (l = L-1) (Sigmoid or Softmax 적용)
        # Z: (배치 데이터 수, 출력층 노드 수)
        # Z = AW + b
        Z = np.dot(A, self.weights_[-1]) + self.biases_[-1]

        # 이진 분류 vs 다중 분류에 따라 출력층 활성화 함수 적용
        if self.multi_class_ == 'binary':
            A = self._sigmoid(Z)
        else:
            A = self._softmax(Z)
        
        # 역전파 계산을 위해 출력층의 가중합, 출력값도 리스트에 저장
        Zs.append(Z)
        activations.append(A)

        # 역전파 계산을 위해 중간 결과물(A, Z)을 담은 튜플 반환
        return activations, Zs
    
    def _backward(self, Y_batch: np.ndarray, activations: list[np.ndarray], Zs:list[np.ndarray]) -> None:
        """
        오차를 뒤로 전달하며 가중치를 업데이트. (Backward Propagation)
        l+1 층의 오차를 l층으로 전달

        Args:
            Y_batch (np.ndarray): (배치 데이터 수, n_classes) 정답 행렬. one-hot-Encoding
            activations (list): 각 층의 활성화 출력값(A) 리스트. 길이: 은닉층 수 + 2 (ex: [X, 은1, 은2, 출])
            Zs (list): 각 층의 가중합(Z) 리스트. 길이: 은닉층 수 + 1 (ex: [은1, 은2, 출])
        """
        m = Y_batch.shape[0] # 배치 데이터 수
        L = len(self.weights_) # 전체 층의 개수 (은닉층 개수 + 1 (출력층))

        # 각 층의 가중치(W)와 편향(b)의 Gradient를 담을 빈 리스트 생성 (0으로 초기화)
        # 가중치 행렬 수(은닉층 + 1) 만큼
        dWs = [np.zeros_like(W) for W in self.weights_]
        dbs = [np.zeros_like(b) for b in self.biases_]

        # 출력층 오차 계산 (delta_L)
        # activations[-1]: (배치 데이터 수, 출력 노드 수) 출력층의 예측값 행렬
        # Y_batch: (배치 데이터 수, 출력 노드 수) 정답 행렬
        # dz: (배치 데이터 수, 출력 노드 수) 출력층의 오차 행렬 (eta_(l+1))
        dZ = activations[-1] - Y_batch

        # 뒤에서부터 앞으로 가면서 오차 전달 (L-1 부터 0 까지 역순으로 반복)
        for l in reversed(range(L)):
            # A_prev: (m, l층 노드수) 앞선 층의 출력값 (입력 신호)
            A_prev = activations[l]

            # Gradient
            # dws[l]: (l층 노드 수, l+1층 노드 수) 가중치 행렬의 Gradient
            # (eta_(l+1) * x_l)
            dWs[l] = (1 / m) * np.dot(A_prev.T, dZ)
            
            # dbs[l]: (l+1층 노드 수, ) bias 배열의 Gradient
            dbs[l] = (1 / m) * np.sum(dZ, axis=0)

            # Regularization 적용
            if self.penalty == 'l2':
                dWs[l] += self.alpha * self.weights_[l]
            elif self.penalty == 'l1':
                dWs[l] += self.alpha * np.sign(self.weights_[l])

            # 입력층(l==0)이 아니라면, 이전 은닉층으로 오차 (dZ)를 전달 (새 오차 신호 eta_l 계산)
            if l > 0:
                
                # dZ(delta_l+1): (배치 데이터 수, l+1층 노드 수) 앞선 층(l+1)에서 넘어온 오차
                # self.weights_[l].T: (l+1층 노드 수, l층 노드 수) 현재 l층 노드와 l+1층 노드를 연결했던 가중치. 현재 층에 오차를 거꾸로 전달하기 위해 전치
                # dA_prev: (배치 데이터 수, l층 노드 수) 앞선 층(l+1)에서 넘어온 오차행렬
                dA_prev = np.dot(dZ, self.weights_[l].T)

                # Zs[l-1]: (배치 데이터 수, l층 노드 수) l층 노드의 활성화 함수 통과 전 가중합(Z)
                # * 요소별 곱(Element-wise)을 해줌
                # dZ: (배치 데이터 수, l층 노드 수) 새롭게 계산된 l층의 오차 신호 (eta_l)
                # 이 값이 다음 루프(l-1)에서는 앞선 층의 오차인 eta_(l+1) 역할을 하게 됨
                dZ = dA_prev * self._act_deriv(Zs[l-1])

        # 계산된 Gradient(dW, db)로 가중치 업데이트
        for l in range(L):
            self.weights_[l] -= self.lr * dWs[l]
            self.biases_[l] -= self.lr * dbs[l]
                
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        입력 데이터가 각 클래스에 속할 확률을 반환.

        Args:
            X (np.ndarray): (n_samples, n_features) 예측할 입력 데이터

        Returns:
            np.ndarray: 예측된 확률값 배열
                        이진 분류: (n_samples, 1)
                        다중 분류: (n_samples, n_classes)
        """
        # 학습(fit())을 하지 않고 예측을 시도할 경우 에러 발생
        if not hasattr(self, 'weights_'):
            raise NotFittedError("모델이 아직 학습되지 않았습니다.")
            
        # 데이터 검증 및 차원 보정
        X = check_array(X)
        # 예측 단계이기에 가중치 업데이트르 위한 역전파 과정이 필요 없음
        # 따라서 반환되는 Zs는 사용하지 않음
        activations, _ = self._forward(X)
        return activations[-1] # 마지막 층의 결과물이 최종 예측 확률

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        확률을 바탕으로 가장 높은 확률을 가진 최종 클래스를 반환함.

        Args:
            X (np.ndarray): (n_samples, n_features) 예측할 입력 데이터

        Returns:
            np.ndarray: (n_samples,) 최종 예측된 클래스 배열
        """
        proba = self.predict_proba(X) # 이진 분류: (n_samples, 1) / 다중 분류: (n_samples, n_classes)
        
        if self.multi_class_ == 'binary':
            # 확률이 0.5 이상이면 클래스 인덱스 1, 아니면 0으로 변환 (true -> 1, false -> 0)
            # .flatten()으로 1차원 배열로 바꿔줌
            indices = (proba >= 0.5).astype(int).flatten()
        else:
            # 다중 분류의 경우 각 행(데이터, axis=1)에서 가장 확률이 높은 열의 인덱스를 반환
            # (n_samples, ): 예측한 인덱스 번호만 남음
            indices = np.argmax(proba, axis=1)
        
        # Fancy Indexing을 통해 내부 인덱스를 실제 클래스 이름으로 변환
        return self.classes_[indices] 
