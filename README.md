# ai_lab: NumPy based ML library & world cup winner sumulation

외부 ML 라이브러리를 사용하지 않고 NumPy만을 사용하여 수학적 원리부터 직접 구현한 머신러닝 라이브러리 및 해당 모델을 사용한 월드컵 승자 예측 시뮬레이션 프로그램입니다.
Repository에는 자체 제작한 ML 라이브러리('ai_lab')의 소스 코드와, 이를 활용한 2026 월드컵 몬테카를로 시뮬레이션 노트북 ('worldcup_winner.ipynb')이 포함되어 있습니다. (data 파일 포함)

## Key Features

### 1. Custom ML Library ('ai_lab')

Scikit-learn의 구조를 벤치마킹하여 직관적이고 일관된 API를 제공합니다. (fit, predict 등)

### Nerual Network ('MLPClassifier')

- MLP(Multi Layer Perceptron)의 Forward, Backpropagation 연산 구현
- 은닉층 노드 동적 생성 및 활성화 함수(ReLU, Tanh, sigmoid, Softmax) 지원
- Mini-batch Gradient Descent 및 L1/L2 Regularization 지원

### Linear Models

'LogisticRegression': Target 클래스 수에 따른 이진/다중 분류 자동 수행 (Mini-batch GD)
'LinearRegression': Linear Regression with Mini-batch GD

### Data Processing & Utilities

'StandaradScaler': fit, transform 분리를 통해서 Feature 간 표준화 수행
'train_test_split': RandomState 객체를 활용한 data shuffling 수행
'check_X_y', 'check_X': 데이터 검증 및 차원 보정

### 2. World Cup winner simulation ('worldcup_winner.ipynb')

자체 제작한 모델을 kaggle에 있는 'International football results from 1872 to 2026', FiFA World Ranking 1992-2024' 두 데이터셋으로 학습시켜 결과를 예측하고 시뮬레이션 합니다.
결과를 예측할 때는 2026 북중미 월드컵 조별리그 3경기와 32강 경기 데이터를 학습된 모델에 넣어 예측값을 얻고 Monte Carlo Simulation을 통해 16강 진출 16개팀 중 월드컵 우승 확률을 얻어냅니다.

## Repository Structure

AI_LAB
 | ai_lab
 | | core           # BaseEstimator, Mixin 등 최상위 객체 및 예외 처리 클래스
 | | metrics        # 평가지표 (Accuracy, Cross-Entropy Loss, MSE, R2 Score)
 | | utils          # 데이터 검증 유틸리티 (Numpy 변환, 결측치 및 차원 체크)
 | | model_selection# 데이터 셔플링 및 Train/Test 분할 모듈
 | | preprocessing  # 데이터 전처리 및 스케일링 (StandardScaler)
 | | linear_model   # 선형/로지스틱 회귀 모델
 | | neural_network # 다층 퍼셉트론(MLP) 모델
 |
 | data             # 모델 학습 및 시뮬레이션용 CSV 데이터 파일
 | worldcup_winner.ipynb   # 데이터 가공, 모델 학습 및 시뮬레이션 메인 코드
 | requirements.txt # 패키지 목록

## 실행 방법

### Prerequisites

- python 3.10+
- NumPy
- Pandas
- ipykernel (?)

### Installation

git clone https

### 실행 결과

```python



## 만든 이유
