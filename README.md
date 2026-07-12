# ai_lab: NumPy based ML library & world cup winner simulation

외부 ML 라이브러리를 사용하지 않고 NumPy만을 사용하여 수학적 원리부터 직접 구현한 머신러닝 라이브러리 및 해당 모델을 사용한 월드컵 승자 예측 시뮬레이션 프로그램입니다.
Repository에는 자체 제작한 ML 라이브러리('ai_lab')의 소스 코드와, 이를 활용한 2026 월드컵 몬테카를로 시뮬레이션 노트북 ('worldcup_winner.ipynb')이 포함되어 있습니다. (data 파일 포함)

### Key Features

#### 1. Custom ML Library ('ai_lab')

Scikit-learn의 구조를 벤치마킹하여 직관적이고 일관된 API를 제공합니다. (fit, predict 등)

#### Neural Network ('MLPClassifier')

- MLP의 Forward, Backpropagation 연산 구현
- 은닉층 노드 동적 생성 및 활성화 함수(ReLU, Tanh, sigmoid, Softmax) 지원
- Mini-batch GD 및 L1/L2 Regularization 지원

#### Linear Models

'LogisticRegression': Target 클래스 수에 따른 이진/다중 분류 자동 수행 (Mini-batch GD)
'LinearRegression': Linear Regression with Mini-batch GD

#### Data Processing & Utilities

'StandardScaler': fit, transform을 통해서 Feature 간 표준화 수행
'train_test_split': RandomState 객체를 활용한 data shuffling 수행
'check_X_y', 'check_array': 데이터 검증 및 차원 보정

#### 2. World Cup winner simulation ('worldcup_winner.ipynb')

자체 제작한 모델을 kaggle에 있는 'International football results from 1872 to 2026', 'FiFA World Ranking 1992-2024' 두 데이터셋을 가공하여 모델 학습을 합니다.
학습시 사용한 Feature는 최근 3경기 득실차, 랭킹차이에 따른 승패 여부입니다.
그 후, 2026 북중미 월드컵 조별리그 3경기와 32강 경기 데이터(data폴더의 ro16_teams.csv)를 학습된 모델에 넣어 예측값을 얻고 Monte Carlo Simulation을 통해 16강 진출 16개팀 중 월드컵 우승 확률을 얻어냅니다.

### Repository Structure

```text
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
```

### 실행 방법

#### Prerequisites

- python 3.10+
- NumPy
- Pandas
- Jupyter Notebook (ipykernel)

#### Installation

```bash
git clone https://github.com/doldoli/my-ML-worldcup-simulator
cd my-ML-worldcup-simulator
pip install -r requirements.txt
```

#### ML Library Usage Example

```python
import numpy as np
from ai_lab.neural_network import MLPClassifier
from ai_lab.preprocessing import StandardScaler
from ai_lab.metrics import accuracy_score

# 1. 샘플 데이터
X = np.array([[0.1, 0.5], [1.2, 0.8], [0.5, 0.2], [1.5, 1.1]])
y = np.array([0, 1, 0, 1])

# 2. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 모델 초기화 및 학습 (Mini-batch GD)
model = MLPClassifier(
    hidden_layer_sizes=(16, 8), 
    activation='tanh', 
    lr=0.01, 
    epochs=1000,
    random_state=42
)
model.fit(X_scaled, y)

# 4. 예측 및 평가
predictions = model.predict(X_scaled)
print(f"Accuracy: {model.score(X_scaled, y)}")
```

#### Simulation Results

ai_lab 라이브러리를 활용한 월드컵 예측 시뮬레이션 결과입니다. 상세한 과정은 'worldcup_winner.ipynb'에서 확인할 수 있습니다.

##### 1. 가공된 데이터셋 (모델 학습용)
<img width="548" height="372" alt="image" src="https://github.com/user-attachments/assets/e0b7787b-de94-4815-8691-ba63a759d6a4" />

##### 2. 모델 성능 비교 
<img width="436" height="224" alt="image" src="https://github.com/user-attachments/assets/9859f494-d33d-4605-bf46-0b139c28fb11" />

##### 3. 2026 월드컵 최종 우승 확률 (10000회 Monte Carlo Simulation)
<img width="436" height="680" alt="image" src="https://github.com/user-attachments/assets/db3cf69e-e09e-4ab7-9ec4-9a22cd18a9d8" />

