# 모델이 학습되지 않은 상태에서 예측을 시도할 때 발생시킬 에러

class NotFittedError(ValueError, AttributeError):
    """
    모델이 아직 학습(fit)되지 않았을 때 발생시키는 에러
    """
    pass