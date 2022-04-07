# -*- codeing = utf-8 -*-
# @Time : 31/03/2022
# @Author : rain
# @Email : stellar052323@163.com
# @File : borderline_smote.py
# @Software : PyCharm

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import BorderlineSMOTE

X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=2, n_redundant=0, flip_y=0,
                           n_features=2, n_clusters_per_class=1, n_samples=100, random_state=9)
sm = BorderlineSMOTE(random_state=42, kind="borderline-1")
"""
BorderlineSMOTE又分为borderline-1和borderline-2，前者在K近邻中选择少数类样本来进行插值，后者选择任意类别样本来进行插值
"""
X_res, y_res = sm.fit_resample(X, y)
print(X)
print(y)
print(X_res)
print(y_res)
