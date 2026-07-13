import numpy as np
from ai_lab.utils.validation import check_array
from ai_lab.core.exceptions import NotFittedError

class StandardScaler:
    """
    데이터의 특성(Feature)들이 평균 0, 표준편차 1을 갖도록 표준화하는 전처리기 클래스.
    GD 사용시 Feature간의 scale 보정을 하여 학습의 안정성을 위하여 학습 전 필수로 사용.
    """
    def __init__(self):
        """
        StandardScaler의 파라미터를 초기화.
        """

        # train data로 fit() 되지 않은 상태에서 표준화하는 것을 방지하기 위해 평균, 표준편차 변수를 미리 선언 후 None으로 초기화
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'StandardScaler':
        """
        각 특성(Feature)의 평균과 표준편차를 계산하여 저장함.
        Train data만을 가지고 평균과 표준편차를 계산하며, Test data는 평균과 표준편차를 구하는데 사용하지 않음.

        Args:
            X (np.ndarray): (n_samples, n_features) 학습할 입력 데이터
            y (np.ndarray, optional): 전처리기에서는 사용하지 않지만 파이프라인 호환성을 위해 유지

        Returns:
            self: 평균과 표준편차 계산이 완료되어 저장된 StandardScaler 객체 자신
        """
        # 입력 데이터 X 검증 및 변환
        X = check_array(X)
        
        # Feature 별 평균 및 표준편차 계산 (n_features,)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # ZeroDivisionError 방지: 분산이 0인 특성의 표준편차를 1.0으로 보정
        self.scale_[self.scale_ == 0.0] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        fit()에서 저장해둔 평균과 표준편차를 이용해 입력 데이터(X)를 표준화
        
        Args:
            X (np.ndarray): (n_samples, n_features) 변환할 입력 데이터

        Returns:
            np.ndarray: (n_samples, n_features) 표준화가 완료된 새로운 데이터 배열
        """
        # 학습되지 않은 객체(mean_, scale_ 부재)에 대한 예외 처리
        if self.mean_ is None or self.scale_ is None:
            raise NotFittedError("해당 StandardScaler 객체가 아직 fit()이 호출되지 않았습니다.")
        
        # X 검증 및 변환
        X = check_array(X)

        # (원본 데이터 - 평균) / 표준편차 (Z-score 정규화 공식으로 정규화)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """
        전처리기의 fit과 transform을 연속으로 실행함.
        주로 train 데이터를 처음 전처리할 때 편의를 위해 사용.

        Args:
            X (np.ndarray): (n_samples, n_features) 변환할 입력 데이터
            y (np.ndarray, optional): 타겟 데이터

        Returns:
            np.ndarray: 표준화가 완료된 데이터 배열
        """
        return self.fit(X, y).transform(X)       
