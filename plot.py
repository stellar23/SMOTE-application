# -*- codeing = utf-8 -*-
# @Time : 03/03/2022 21:34
# @Author : rain
# @File : plot.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np

"""
x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2
# 图一
plt.figure()
plt.plot(x, y1)
# 设置范围
plt.xlim(-1, 2)
plt.ylim(-2, 3)
# 标注
plt.xlabel('I am x')
plt.ylabel('I am y')
plt.yticks([-2, -1.8, -1, 1.22, 3], ['really bad', 'bad', 'normal', 'good', 'very good'])
plt.show()
# 图二
plt.figure(num=3, figsize=(8, 5))
l2, = plt.plot(x, y2, label='parabola')
l1, = plt.plot(x, y1, color='red', linewidth=10.0, linestyle='--', label='line')  # 设置显示格式
plt.legend(handles=[l1, l2], loc='best')  # 显示图例;注意：传入handles需要加逗号
plt.show()
"""

print("------------------------------ 散点图 --------------------------------")

X = np.random.normal(0, 1, 1024)  # 正态分布
Y = np.random.normal(0, 1, 1024)
T = np.arctan2(X, Y)
plt.scatter(X, Y, s=10, c=T, alpha=0.5)  # 数据源，大小，颜色，透明度
plt.xticks(())  # 删去所有ticks
plt.yticks(())
plt.show()
