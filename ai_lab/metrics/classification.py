import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    정확도 (Accuracy)를 계산함.
    전체 예측 중에서 정답을 맞춘 비율을 의미.
    
    Args:
        y_true (np.ndarray): (n_samples,) 실제 정답 데이터
        y_pred (np.ndarray): (n_samples,) 모델이 예측한 클래스 데이터
        
    Returns:
        float: 정확도 (0.0 ~ 1.0)
    """
    return float(np.mean(y_true == y_pred))


def cross_entropy_loss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    크로스 엔트로피 손실 (Cross Entropy Loss)을 계산함.
    로지스틱 회귀 및 신경망 분류 모델의 오차를 계산할 때 사용됨.
    
    Args:
        y_true (np.ndarray): (n_samples,) 실제 정답 데이터. multinomial 분류일 경우 One-hot Encoding 형태여야 함.
        y_pred_proba (np.ndarray): (n_samples,) 모델이 예측한 확률값 (0.0 ~ 1.0)
        
    Returns:
        float: 계산된 손실값(Loss)
    """
    # log(0) 계산 시 inf 에러가 발생하는 것을 방지하기 위해 아주 작은 값(epsilon)을 설정
    eps = 1e-15
    # 확률값이 0이나 1이 되지 않도록 0.00...1 ~ 0.99...9 사이로 잘라줌
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    # y_pred_proba가 eps보다 작으면 eps로, 1 - eps 보다 크면 1 - eps로 값 설정
    
    # 1D 배열인 경우 (binary CE)
    # (n,1) 형태의 2차원 배열이 들어오든 (n,) 형태의 1차원 배열이 들어오든 모두 1차원으로 맞춰서 계산
    if y_true.ndim == 1 or y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 1:
        p = np.squeeze(y_pred_proba)
        y = np.squeeze(y_true)
        loss = -np.mean(y * np.log(p) + (1-y) * np.log(1-p))
    # 2D 배열인 경우 (multinomial CE)
    else:
        loss = -np.mean(np.sum(y_true * np.log(y_pred_proba), axis=1))
    
    return float(loss)