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
    
    if shuffle: # 데이터는 보통 특정 순서대로 정렬되어 있는 경우가 많음 (예: 앞 50개 고양이 뒤 50개 강아지). 따라서 섞어주는 게 필요함
        
        # NumPy의 random 모듈의 RandomState (난수 생성기) 클래스 객체 rng 생성.
        # random_state(난수 시드)가 주어지면 항상 똑같은 순서로 섞이게 고정됨
        # random_state가 None이면 컴퓨터의 현재 시간을 기준으로 매번 다른 난수를 생성함.
        rng = np.random.RandomState(random_state)

        # permutation(N) 은 시드에 따라 0부터 N-1 까지의 정수를 랜덤하게 섞은 1차원 배열 반환
        # 원본 데이터는 놔두고 인덱스를 섞어서 X, y의 짝이 유지되도록 함
        indices = rng.permutation(n_samples)
        
    test_samples = int(n_samples * test_size) # 100개 중 25%면 25개
    
    # 섞인 인덱스를 기준으로 테스트용과 학습용 분리
    # test_samples 비율만큼 무작위로 섞인 인덱스를 train/test용으로 나눠줌
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    # NumPy의 Fancy Indexing 을 이용해서 무작위로 섞인 인덱스 번호에 해당하는 실제 데이터를 추출하여 배열을 반환
    # 결과적으로 비율에 맞게 무작위로 X,y 데이터가 train/test 용으로 나뉘어짐
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]