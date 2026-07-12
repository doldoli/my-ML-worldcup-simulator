# 모델이 예측한 결과와 실제 정답을 비교해서 점수를 매기는 도구

# Classification: accuracy_score, f1_score
# Regression : MSE, r2_score

from .regression import mean_squared_error, r2_score
from .classification import accuracy_score, cross_entropy_loss

__all__ = [
    "mean_squared_error",
    "r2_score",
    "accuracy_score",
    "cross_entropy_loss"
]