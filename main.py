# -*- codeing = utf-8 -*-
# @Time : 25/02/2022 21:58
# @Author : rain
# @File : main.py
# @Software: PyCharm

import numpy as np  # 数组计算库
import matplotlib.pyplot as plt
from smote import Smote  # 导入smote文件定义的Smote类
from svm import SVM
from imblearn.over_sampling import BorderlineSMOTE
from matplotlib import rcParams  # 定义绘图格式


if __name__ == "__main__":

    # 导入数据
    with open('Iris-versicolor.txt', 'r') as f:
        """
        导入并处理少数样本
        """
        tmpmin = []
        res = []
        datamin = f.readline()  # 初始化，读入第一行
        while datamin:
            a = datamin.replace('\n', ' ').replace(',Iris-versicolor', '').split(",")
            tmpmin.append(a)
            datamin = f.readline()
            pass
        for mm in tmpmin:  # 整合列表
            for nn in mm:
                res.append(float(nn))
                pass
            pass
        pass

    minor_samples = np.array(res).reshape((len(res)//2), 2)  # 定义数组维数
    print("Minor_samples：\n", minor_samples)

    with open('Iris-setosa.txt', 'r') as ff:  # 导入多数样本
        """
        导入并处理多数样本
        """
        tapmax = []
        ress = []
        datamax = ff.readline()
        while datamax:
            b = datamax.replace('\n', ' ').replace(',Iris-setosa', '').replace(',Iris-virginica', '').split(',')
            tapmax.append(b)
            datamax = ff.readline()
            pass
        for m in tapmax:
            for n in m:
                ress.append(float(n))
                pass
            pass
        pass

    major_samples = np.array(ress).reshape((len(ress)//2), 2)
    print("Major_samples：\n", major_samples)

    # Borderline_SMOTE
    x = np.vstack((minor_samples, major_samples))
    y = [0] * minor_samples.shape[0] + [1] * major_samples.shape[0]

    sm = BorderlineSMOTE(random_state=42, kind="borderline-1")
    """
    random_state=42随机数种子
    BorderlineSMOTE分为borderline-1和borderline-2，前者在K近邻中选择少数类样本来进行插值，后者选择任意类别样本来进行插值
    """
    x_res, y_res = sm.fit_resample(x, y)  # Borderline_SMOTE抽样
    # print("x_res=", x_res)
    # print("y_res=", y_res)

    # 调用smote函数
    smote = Smote(N=325)  # 这里可以设置合成样本比例
    synthetic_points = smote.combine(minor_samples)
    print("Synthetic_points：\n", synthetic_points)

    # smote绘图
    config = {
        "font.family": 'serif',
        "font.size": 15,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun']}  # 设置绘图格式
    rcParams.update(config)

    plt.figure()  # 图一（未合成）
    plt.rc('axes', unicode_minus=False)
    plt.title("未合成样本")
    plt.scatter(minor_samples[:, 0], minor_samples[:, 1])
    plt.scatter(major_samples[:, 0], major_samples[:, 1])
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.legend(["Minor samples", "Major samples"], loc='best')
    plt.show()
    plt.figure()  # 图二（smote合成后）
    plt.rc('axes', unicode_minus=False)
    plt.title("SMOTE合成新样本后")
    plt.scatter(minor_samples[:, 0], minor_samples[:, 1])  # 散点图
    plt.scatter(major_samples[:, 0], major_samples[:, 1])
    plt.scatter(synthetic_points[:, 0], synthetic_points[:, 1])  # 取合成数组中每组元素的的第一个，第二个
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.legend(["Minor samples", "Major samples", "Synthetic samples"], loc='best')
    plt.show()
    plt.figure()  # 图三（Borderline——SMOTE合成后）
    plt.rc('axes', unicode_minus=False)
    plt.title("Borderline-SMOTE合成新样本后")
    plt.scatter(major_samples[:, 0], major_samples[:, 1], c="darkorange")
    plt.scatter(x_res[(len(x)-1):-1, 0], x_res[len(x)-1:-1, 1], c="forestgreen")  # Borderline——SMOTE合成的新样本
    plt.legend(["Synthetic samples", "Major samples"], loc='best')
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.show()

    # SVM(SMOTE)
    void = np.empty((1, 2))  # 空数组
    svm1 = SVM(void, minor_samples, major_samples)  # 无smote
    svm2 = SVM(synthetic_points, minor_samples, major_samples)  # 有smote
    # SVM(Borderline_SMOTE)
    Border_synthetic_points = x_res[(len(x)-1):-1, ]
    svm3 = SVM(void, Border_synthetic_points, major_samples)
