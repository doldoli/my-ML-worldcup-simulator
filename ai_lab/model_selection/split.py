import numpy as np
from ai_lab.utils.validation import check_X_y

def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.25, 
    random_state: int | None = None, 
    shuffle: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    데이터를 Train/Test set으로 분리하는 함수.

    Args:
        X (np.ndarray): (n_samples, n_features) 특성(Feature) 데이터
        y (np.ndarray): (n_samples,) 정답(Target) 데이터
        test_size (float): 테스트 데이터의 비율 (기본값 0.25)
        random_state (int, optional): 난수 시드값
        shuffle (bool): 데이터를 분할하기 전에 섞을지 여부

    Returns:
        tuple: (X_train, X_test, y_train, y_test) 형태의 분할된 배열
    """

    # 입력 데이터 X, y검증 및 변환
    X, y = check_X_y(X, y)
    
    n_samples = len(X)

    # 인덱스 배열 생성
    indices = np.arange(n_samples)
    
    if shuffle: # 원본 데이터의 정렬 순서에 따른 학습 편향을 막기 위해 Shuffle 수행

        # 원본 데이터를 직접 섞지 않고 인덱스만 섞어 X와 y의 짝을 유지함
        rng = np.random.RandomState(random_state)

        # permutation(N) 은 시드에 따라 0부터 N-1 까지의 정수를 랜덤하게 섞은 1차원 배열 반환
        indices = rng.permutation(n_samples)
        
    test_samples = int(n_samples * test_size)
    
    # 섞인 인덱스를 기준으로 테스트용과 학습용 분리
    # test_samples 비율만큼 무작위로 섞인 인덱스를 train/test용으로 나눠줌
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    # 추출된 인덱스를 실제 데이터에 매핑하여 반환 (Fancy Indexing)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
