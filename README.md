# Some experiments on GP_UCB:

## Naive sin + cos
1. x  range(-3, 3, 0.25)
2. y  range(-3, 3, 0.25)

![image](https://github.com/yxcT-T/GP_UCB/blob/master/output.gif "result figure")

## Random Forest
1. n_estimators  range(50, 300, 10)
2. max_depth     range(1, 10, 1)

## GBDT
1. n_estimators  range(50, 300, 10)
2. max_depth     range(1, 10, 1)

# Some global variables

## THREAD_NUM
The number of threads
## DATA_PATH
The path of [dataset](http://archive.ics.uci.edu/ml/datasets/Covertype).
