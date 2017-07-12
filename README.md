# GP-UCB
Implement some experiments on GP-UCB[1].

## Experiments
### Naive sin + cos
1. x  range(-3, 3, 0.25)
2. y  range(-3, 3, 0.25)

![image](https://github.com/yxcT-T/GP_UCB/blob/master/result/sin_cos.gif 'sinx + cosx result')

### Random Forest
1. n_estimators  range(50, 300, 10)
2. max_depth     range(1, 10, 1)

![image](https://github.com/yxcT-T/GP_UCB/blob/master/result/RF_Beta_100.gif 'random forest result')

### GBDT
1. n_estimators  range(50, 300, 10)
2. max_depth     range(1, 10, 1)

![image](https://github.com/yxcT-T/GP_UCB/blob/master/result/GBDT_Beta_100.gif 'gbdt result')

## Some global variables
1. THREAD_NUM: The number of threads
2. DATA_PATH: The path of [dataset](http://archive.ics.uci.edu/ml/datasets/Covertype).

## Reference
[1. Srinivas, Niranjan, et al. "Gaussian process optimization in the bandit setting: No regret and experimental design." arXiv preprint arXiv:0912.3995 (2009).](https://arxiv.org/pdf/0912.3995.pdf)

