import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE를 계산함.
    0에 가까울수록 모델의 성능이 좋음을 의미.

    Args:
        y_true (np.ndarray): 실제 정답 데이터 (샘플 수,)
        y_pred (np.ndarray): 모델이 예측한 데이터 (샘플 수,)

    Returns:
        float: 계산된 MSE 값
    """

    return float(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    결정 계수 (R^2 score)를 계산함.
    모델이 데이터의 분산을 얼마나 잘 설명하는지 나타내며, 1.0에 가까울수록 모델의 성능이 좋음을 의미함.

    Args:
        y_true (np.ndarray): (n_samples,) 실제 정답 데이터
        y_pred (np.ndarray): (n_samples,) 모델이 예측한 데이터
        
    Returns:
        float: 계산된 R^2 Score
    """

    sse = np.sum((y_true - y_pred) ** 2) # 오차 제곱합
    sst = np.sum((y_true - np.mean(y_true)) ** 2) # 총 제곱합

    # zero-division 방지
    if sst == 0.0:
        return 0.0
    
    return float(1 - (sse / sst))