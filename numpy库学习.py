# -*- codeing = utf-8 -*-
# @Time : 27/02/2022 19:32
# @Author : rain
# @File : numpy库学习.py
# @Software: PyCharm

import numpy as np  # 导入数组矩阵运算库

a = np.array([1, 2, 3, 4, 5])  # 创建一个数组，初始化数据
b = np.zeros((2, 3))
c = np.ones((3, 2))
d = np.linspace(0, 3, 5)  # 返回区间内等间距分布的数组(输出范围，样本总个数)
e = np.arange(3, 7)  # 创建一个递增或递减数列
print(a)
print(b)
print(c)
print(d)
print(a.shape)  # 输出数组尺寸（行，列）
print(e)
print("------------------ 生成随机数组与数据类型相关 -------------------")
r = np.random.rand(4, 4)
print(r)
print(r.dtype)  # 默认64位float
r1 = np.ones((4, 4), dtype=np.int32)  # 可定义其他数据类型
r2 = r1.astype(float)  # 转换数据类型
print(r1)
print(r2)
print("------------------ 基础计算（begin） -------------------")
p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
print(p1 + p2)
print(p1 - p2)
print(p1 * p2)
print(p1 / p2)
print(np.dot(p1, p2))  # 点乘运算
p3 = np.array([(1, 2), (3, 4)])
p4 = np.array([(5, 6), (7, 8)])
print(p3 @ p4)  # 矩阵运算
print(p4 @ p3)
p5 = p3[(p3 > 1) & (p3 % 2 == 0)]  # 条件筛选
print("p5=", p5)
print("------------------- 基础计算（end） -------------------")
