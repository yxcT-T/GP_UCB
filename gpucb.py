# coding: utf-8
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys
from random import randint
import threading


DATA_PATH = '../data/covtype.data'
THREAD_NUM = 4


class GPUCB(object):

    def __init__(self, meshgrid, environment, beta=100.):
        self.meshgrid = np.array(meshgrid)
        self.environment = environment
        self.beta = beta

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        self.X = []
        self.T = []

    def argmax_ucb(self):
        return np.argmax(self.mu + self.sigma * np.sqrt(self.beta))

    def learn(self):
        grid_idx = self.argmax_ucb()
        self.sample(self.X_grid[grid_idx])
        gp = GaussianProcessRegressor()
        gp.fit(self.X, self.T)
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)

    def sample(self, x):
        t = self.environment.sample(x)
        self.X.append(x)
        self.T.append(t)

    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                self.mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                self.environment.sample(self.meshgrid), alpha=0.5, color='b')
        ax.scatter([x[0] for x in self.X], [x[1] for x in self.X], self.T, c='r',
                marker='o', alpha=1.0)
        plt.savefig('fig_%02d.png' % len(self.X))



class DummyEnvironment(object):

    def __init__(self, experiment = 'naive', (x, y) = ([], [])):
        self.experiment = experiment
        self.x = x
        self.y = y
        if experiment == 'naive':
            pass
        elif experiment == 'random forest':
            self.init_table_rf()
        elif experiment == 'gbdt':
            self.init_table_gbdt()
        else:
            print 'No such experiment!'
            sys.exit(0)
        print 'Init finished!'

    def sample(self, x):
        if self.experiment == 'naive':
            return np.sin(x[0]) + np.cos(x[1])
        elif self.experiment == 'random forest':
            return self.table[(x[0], x[1])]
        elif self.experiment == 'gbdt':
            return self.table[(x[0], x[1])]
        else:
            print 'No such experiment!'
            sys.exit(0)

    def accuracy(self, x, y):
        ret = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                ret += 1
        return float(ret) / float(len(x))

    def multithread_rf(self, thread_id):
        self.lock.acquire()
        print 'Thread %d begin!' % thread_id
        self.lock.release()
        for i in self.x:
            for j in self.y:
                if i / 10 % THREAD_NUM != thread_id:
                    continue
                rf = RandomForestClassifier(n_estimators=i, max_depth=j)
                rf.fit(self.data['train']['X'], self.data['train']['Y'])
                pY = rf.predict(self.data['test']['X'])
                f = self.accuracy(pY, self.data['test']['Y'])
                self.lock.acquire()
                print 'thread_id=%d, n_estimators=%d, max_depth=%d, accuracy=%f' % (thread_id, i, j, f)
                self.table[(i, j)] = f
                self.lock.release()

    def init_table_rf(self):
        '''
        初始化Random Forest每个点的准确率
        '''
        print 'init random forest\' table ...'
        self.load_data()
        self.table = {}
        self.lock = threading.Lock()
        threads = []
        for i in range(THREAD_NUM):
            td = threading.Thread(target=self.multithread_rf, args=(i,))
            td.setDaemon(True)
            td.start()
            threads.append(td)
        for td in threads:
            td.join()

    def multithread_gbdt(self, thread_id):
        for i in self.x:
            for j in self.y:
                if i / 10 % THREAD_NUM != thread_id:
                    continue
                gbdt = GradientBoostingClassifier(n_estimators=i, max_depth=j)
                gbdt.fit(self.data['train']['X'], self.data['train']['Y'])
                pY = gbdt.predict(self.data['test']['X'])
                f = self.accuracy(pY, self.data['test']['Y'])
                self.lock.acquire()
                print 'thread_id=%d, n_estimators=%d, max_depth=%d, accuracy=%f' % (thread_id, i, j, f)
                self.table[(i, j)] = f
                self.lock.release()

    def init_table_gbdt(self):
        '''
        初始化GBDT每个点的准确率
        '''
        print 'init GBDT\'s table ...'
        self.load_data()
        self.table = {}
        self.lock = threading.Lock()
        threads = []
        for i in range(THREAD_NUM):
            td = threading.Thread(target=self.multithread_gbdt, args=(i,))
            td.setDaemon(True)
            td.start()
            threads.append(td)
        for td in threads:
            td.join()

    def load_data(self):
        '''
        Load the dataset
        '''
        self.data = {'train' : {'X' : [], 'Y' : []}, 
                     'test' : {'X' : [], 'Y' : []}}
        with open(DATA_PATH, 'r') as lines:
            for line in lines:
                line = line.split(',')
                single_X = line[0 : -1]
                single_Y = line[-1]
                if randint(1, 1000) <= 100:
                    continue
                if randint(1, 1000) <= 100:
                    self.data['test']['X'].append(single_X)
                    self.data['test']['Y'].append(single_Y)
                else:
                    self.data['train']['X'].append(single_X)
                    self.data['train']['Y'].append(single_Y)


if __name__ == '__main__':
    x = np.arange(-3, 3, 0.25)
    y = np.arange(-3, 3, 0.25)
    env = DummyEnvironment('naive', (x, y))
    agent = GPUCB(np.meshgrid(x, y), env)
    for i in range(20):
        agent.learn()
        agent.plot()
