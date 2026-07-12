# 데이터 검증. 사용자가 리스트를 넣든 내부적으로는 Numpy 배열로 변환해야 함
# 결측치 (NaN)가 있는지 X와 y의 데이터 길이가 맞는지 검사
# check_X_y(), check_array 만들어서 모델에서 데이터 검증할 수 있도록

import numpy as np

def check_array(X: np.ndarray | list) -> np.ndarray:
    """
    입력된 데이터 X가 모델 학습에 적합한 2차원 배열인지 검사함.
    학습된 모델에 입력 데이터를 넣어 예측값을 뽑아낼 때의 입력 데이터 검사하기 위한 함수.

    Args:  
        X (np.ndarray | list): (n_samples, n_features) 검사 및 변환할 입력 특성 데이터

    Returns:
        np.ndarray: (n_samples, n_features) 검증이 완료된 2차원 Numpy 배열
    """
    # X가 np.ndarray 타입이 아니면 np.ndarray 타입으로 변환해줌
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    #문자열 등 연산 불가능한 타입이 왔을때 에러가 나는 것을 방지
    try:
        X = X.astype(np.float64)
        # float 대신 np.float64를 써서 데이터가 많아질 때 연산 속도를 향상함
    except ValueError:
        raise ValueError("입력 데이터 X는 숫자형(float or int)로 변환할 수 있어야 합니다.")
    
    if X.ndim == 1:
        # 모델의 입력 데이터 X는 반드시 (n_samples, n_features) 형태의 2차원 행렬이어야 함. 모델의 수학 공식(행렬 곱셈), 로직을 이를 기준으로 구현했기 때문
        # 데이터가 1차원이면 강제로 2차원(열 1개)으로 변환
        # 열을 1개로 행렬 모양을 맞춤.
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        raise ValueError(f"X는 2차원 배열이어야 합니다. 현재 차원: {X.ndim}")
        
    # 결측치(NaN)나 무한대(Inf) 값이 있는지 검사
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("입력 데이터에 NaN(결측치) 또는 Inf(무한대)가 포함되어 있습니다.")
        
    return X

def check_X_y(X: np.ndarray | list, y: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:
    """
    X와 y의 샘플 수가 일치하는지 검사하고, 모델 학습에 적합한 형태로 변환합니다.
    모델 학습시의 입력 데이터 X, y를 검사하기 위한 함수.

    Args:
        X (np.ndarray | list): (n_samples, n_features) 특성(Feature) 데이터
        y (np.ndarray | list): (n_samples,) 정답(Target) 데이터

    Returns:
        tuple[np.ndarray, np.ndarray]: ((n_samples, n_features), (n_samples,)) 검증 및 변환이 완료된 (X, y) 튜플
    """
    # 특성 데이터 먼저 검사 및 필요 시 2차원 행렬 변환
    X = check_array(X)
    
    # y가 np.ndarray 타입이 아니면 np.ndarray 타입으로 변환해줌
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    
    # y는 1차원 배열이어야 함. 구현한 분류 모델에서는 샘플 당 정답이 하나의 값만 주어지기 때문. 또한 모델의 수학적 계산 시 이를 기준으로 구현했기에 형태를 맞춰줌
    if y.ndim > 1:
        if y.shape[1] == 1:
            # y가 2차원 이상일 때 [[a], [b], ...] 형태면 1차원으로 변환하고 아니면 에러 발생
            y = np.ravel(y)
            # (n, 1) 형태의 2차원 배열을 (n, ) 형태의 1차원 배열로 바꿔주는 문법.
        else:
            raise ValueError("다중 출력(Multi-output) y는 현재 지원하지 않습니다. 1차원 배열을 입력하세요.")
    
    # X, y의 샘플 수가 일치하는 지 확인
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X의 샘플 수({X.shape[0]})와 y의 샘플 수({y.shape[0]})가 일치하지 않습니다.")

    # 검증 및 변환 완료된 데이터 반환        
    return X, y