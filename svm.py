# -*- codeing = utf-8 -*-
# @Time : 27/03/2022
# @Author : rain
# @Email : stellar052323@163.com
# @File : svm.py
# @Software : PyCharm

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import rcParams


class SVM:
    def __init__(self, synthetic_points, minor_samples, major_samples):
        self.synthetic_points = synthetic_points
        self.minor_samples = minor_samples
        self.major_samples = major_samples
        pass
        # 去掉空集
        if self.synthetic_points.all() == 0:
            self.x = np.vstack((self.minor_samples, self.major_samples))
            self.y = [0] * self.minor_samples.shape[0] + [1] * self.major_samples.shape[0]  # 分成两类
            pass
        else:
            self.x = np.vstack((self.synthetic_points, self.minor_samples, self.major_samples))  # 数组合并
            self.y = [0] * (self.synthetic_points.shape[0] + self.minor_samples.shape[0]) \
                     + [1] * self.major_samples.shape[0]  # 分成两类
            pass
        pass
        clf = svm.SVC(kernel='linear')  # 线性分类器
        clf.fit(self.x, self.y)  # 训练分类器

        self.w = clf.coef_[0]  # 线性回归系数，获取向量机的法向量w
        self.a = -self.w[0] / self.w[1]  # 斜率
        # 画图划线
        self.xx = np.linspace(4, 7)
        self.yy = self.a * self.xx - (clf.intercept_[0]) / self.w[1]  # clf.intercept_[0]用来获得截距(这里共有两个值，分别为到x和到y的)

        # print("w:", self.w)
        # print("a:", self.a)
        # print("support_vectors_:\n", clf.support_vectors_)  # 支持向量
        # print("clf.coef_:", clf.coef_)

        # SVM绘图
        config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun']}  # 设置绘图格式
        rcParams.update(config)

        plt.figure()
        plt.title("SVM分类结果")
        plt.plot(self.xx, self.yy)
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=75, )  # 支持向量
        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y, cmap=plt.cm.Paired)  # cm-colormap,Paired两个相近色彩输出
        plt.xlim((4, 7))
        plt.ylim((2, 4.5))
        plt.xlabel("Feature1")
        plt.ylabel("Feather2")
        plt.show()

        # 计算准确度
        svm_score = clf.score(self.x, self.y)
        print("Accuracy score =", svm_score)

    pass
