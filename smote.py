# -*- codeing = utf-8 -*-
# @Time : 29/03/2022
# @Author : rain
# @Email : stellar052323@163.com
# @File : smote.py
# @Software : PyCharm

import numpy as np
import random
from sklearn.neighbors import NearestNeighbors


class Smote:
    def __init__(self, N=50, k=5, r=2):
        self.N = N  # 合成样本占少数样本总数的N%
        self.k = k  # k代表近邻数
        self.r = r  # 距离决定因子，2代表欧氏距离
        self.newNumber = 0  # 新合成样本个数
        pass

    def combine(self, samples):
        self.samples = samples
        self.T, self.numattrs = self.samples.shape  # 少数样本个数和特征数

        if self.N < 100:
            np.random.shuffle(self.samples)  # 打乱样本顺序
            self.T = int(self.N * self.T / 100)  # 抽取前面N*T/100个样本，作为新的少数类样本
            self.samples = self.samples[0:self.T, :]  # 数组切片：逗号前面是行索引，逗号后面是列索引。
            self.N = 100
            pass

        if self.T <= self.k:
            self.k = self.T - 1  # 减去本身
            pass

        N = int(self.N / 100)  # int向下取整
        self.synthetic = np.zeros((self.T * N, self.numattrs))  # 样本合成数组初始化
        neighbors = NearestNeighbors(n_neighbors=self.k + 1, algorithm='ball_tree', p=self.r).fit(self.samples)
        """
        NearestNeighbors：调用neighbors库里的近邻函数
        n_neighbors=self.k + 1：NearestNeighbors会把原样本自己也算进近邻
        ball_tree：树搜索算法
        p=self.r：计算欧氏距离
        fit(self.samples)：输入samples样本
        """
        pass

        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1, -1)), return_distance=False)[0][1:]
            """
            neighbors.kneighbors：调用neighbors库里的kneighbors方法搜索k近邻
            reshape((1,-1))：将一维列表变成1行x列的二维数组；-1代表依据输入决定当前维度大小
            return_distance=False：不输出距离
            [0]：输出样式[],[],…,[]；[1:]：不要第一个（该样本自己）
            """
            pass

            self.__populate(N, i, nnarray)

        return self.synthetic
    pass

    def __populate(self, N, i, nnarray):  # x合成 = x + rand(0,1) ∗ (x近邻−x)
        for j in range(N):
            nrandom = random.randint(0, self.k - 1)
            diff = self.samples[nnarray[nrandom]] - self.samples[i]
            gap = random.uniform(0, 1)
            self.synthetic[self.newNumber] = self.samples[i] + gap * diff  # 合成新样本
            self.newNumber += 1
            pass
        pass
