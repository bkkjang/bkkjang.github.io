---
layout: post
title: optimization
date: 2022-01-26 19:20:23 +0900
category: sample
---


************
### Optimization

딥러닝에서 **최적화(Optimization)**란 **손실 함수(Loss function)**의 값을 최소화 하는 파라미터 값을 찾는 것

### Generalization
**일반화(Generalization)**는 모델을 돌렸을 때, 
학습데이터로 돌렸을 때와 테스트 데이터로 돌렸을 때 출력값과 성능에서 차이가 많이 나지 않도록 하는 것

### Generalization performance
 **training error**와 **test error**사이의 차이를 말하고, generalization 성능이 좋다는 것은 이 네트워크의 성능이 학습데이터와 비슷하게 테스트 데이터에서도 나올것이라고 보장해주는 것
 
************
### Goal of Optimization and Deep Learning


최적화는 딥러닝을 위한 loss function을 최소화하는 방법을 제공하지만, 본질적으로 최적화와 딥러닝의 목표는 근본적으로 다름

> #### 최적화의 목표
Objective(Loss)를 최소화하는 것 (최적의 파라미터 찾기)
training error를 줄이는 것

> #### 딥러닝의 목표
한정된 양의 데이터를 이용해 적합한 모델을 찾는 것
generalization error를 줄이는것


> #### Empirical Risk
훈련 데이터 셋의 평균 loss 
(그냥 risk는 전체 데이터에서 예상되는 loss)

> #### Empirical Risk Minimiztion
다양한 예를 검토하여 손실을 최소화하는 모델을 찾아내는 과정
주어진 한정된 데이터의 분포를 따르는 loss function의 기대값을 최소화 시키는 과정

![](https://images.velog.io/images/bk4650/post/e3cb68ed-e14a-4538-9a59-f0ea663939af/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-01-20%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.54.32.png)

주어진 한정된 데이터로 모델을 학습시킨다면, 초기 모델의 신뢰성(confidence) 는 매우 낮을 것이다. 그만큼 모델의 risk는 높다고 할 수 있다. 이둘의 간극을 모델 출력이 주어진 데이터를 따라가지 못한다고하여 **underfitting**이라고 부른다.

모델이 학습을 진행하다보면 cross over되는 순간이 온다. 이때는 모델이 주어진 데이터를 적절히 따라가면서 모델의 위험도도 적절히 낮다. **훈련을 멈춰야하는 순간**

만약 이를 지나 좀 더 훈련을 진행한다면 모델은 주어진 데이터를 완벽하게 따라가고, 모델의 위험도도 줄어들 것이다. 하지만 이 상태에서는 새로운 데이터에 대한 범용성을 확보할 수 없다. 주어진 데이터에만 과도하게 **편향(biased)** 되었기 때문이다. 이 상태를 **overfitting** 이라고 부른다.


```python
def f(x):  #risk function
    return x * torch.cos(np.pi * x)
def g(x):  #emperical-risk function
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)
```

```python
def annotate(text, xy, xytext):  #save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='>'))
x = torch.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

아래 그래프는 훈련 데이터 세트에 대한 Empirical Risk의 최소값이 Risk의 최소값(일반화 오류)과 다른 위치에 있을 수 있음을 보여준다.

![](https://images.velog.io/images/bk4650/post/f2946fea-691a-4ad5-b55e-6072af99c28e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-01-20%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.13.38.png)
************
### Optimization Challenges in Deep Learning

모델의 generalization error보다는 objective function(loss)를 최소화하는 최적화 알고리즘의 성능에 초점

### Analytical solutions 과 Numerical solutions

> **Analytical solution** 은 문제를 잘 이해할 수 있는 형태로 구성하고 정확한 솔루션을 계산하는 것을 의미

수학적인 모델을 가지고 있고, 우리는 그 behavior(행위, 혹은 행동)을 알고 싶다고 가정하였을 때, 즉 우리는 어떤 일련의 식(set of equation)으로 표현되는 solution을 알고 싶은 것

가장 좋은 것은 우리가 알고 있는 미적, 삼각함수 등등 여러 수학적인 테크닉을 통해 그것을 정의

이를 통해 특정 circumstance(환경 혹은 상태)에서의 모델을 정확하게 정의
이것이 analytic solution

그러나 이것은 단지 단순한 모델에서만 가능 좀 더 복잡한 모델에서는 수학이 너무 복잡해 질 수 있음

>
**Numerical solutions**은 해를 추측하고 문제 해결을 멈출 만큼 충분히 잘 해결되었는지 테스트하는 것을 의미

위의 문제에서 식을 풀기위해 numerical method를 사용해서 문제를 풀게 된다.

시간에 따라 변화하는 미분 방정식을 풀 때, numerical method는 변수의 초기값에서 시작하고 식을 시간단위에 따라 변화는 이 변수들의 변화량를 찾는데 사용

딥러닝에서 대부분의 objective function는 복잡하고 분석적인 해결책이 없음. 
대신, 우리는 Numerical solutions 알고리즘을 사용해야 함 
최적화 알고리즘은 대부분 이 범주에 속함


****
**딥러닝 최적화에는 많은 과제가 있음
모델 학습 과정에서 오류를 최소화하는 파라미터 값을 찾기 위해 경사하강법(Gradient Decent)을 사용하는데, 
Local minima, Saddle Points, Vanishing Gradients 등의 문제가 생길 수 있음...**

### Local minima
****
목적 함수 f(x)에 대해, x에서 f(x)의 값이 x 근처의 다른 점에서 f(x)의 값보다 작다면, f(x)는 Local minimum이 될 수 있다. 

만약 x에서 f(x)의 값이 전체 영역의 목적 함수의 최소값이라면, f(x)는 Global minimum이다.

예를 들어, 주어진 함수는 다음과 같다.

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$

우리는 이 함수의 로컬 최소값과 전역 최소값을 근사할 수 있다.

```python
x = torch.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```
![](https://images.velog.io/images/bk4650/post/676dd550-7c36-4c85-a9da-31deb2ace521/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-01-20%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.18.40.png)

![](https://images.velog.io/images/bk4650/post/4f6a5a3a-ec38-4158-907f-faa2a4068dfd/image.png)

딥러닝 모델의 objective function은 일반적으로 많은 Local minima을 가지고 있다.
최적화 문제의 Numerical solution이 Local minima에 가까울 때, 기울기가 0에 근접하거나 0이 될 때 전역적이 아닌 국소적으로만 objective function을 최소화할 수 있음.

어느 정도의 noise를 주는 것으로 Local minimum에서 벗어날 수 있다 (Simulated Annealing)

실제로, 이것은 미니 배치에 대한 gradient의 자연스러운 변화가 Local minima에서 parameter를 제거할 수 있는 미니 배치 stochastic gradient descent의 유익한 특성 중 하나이다.


### Saddle Points


𝑓(𝑥)=𝑥<sup>3</sup> 에서 보았을 때
  
