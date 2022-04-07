# -*- codeing = utf-8 -*-
# @Time : 07/03/2022 19:51
# @Author : rain
# @File : data insert.py
# @Software: PyCharm


import numpy as np

"""
file = open(filename,mode)
r 只读方式打开，文件不存在报错
w 写入文件，文件不存在，创建文件
a 增加方式写入，文件不存在，创建新文件
"""
"""
file = open('iris_smote.txt', "r")
datas = file.readlines()
print(datas)
file.close()
"""
# 推荐使用
with open('Iris-versicolor.txt', 'r')as f:  # 导入少数样本
    res = []
    datamin = f.readline()
    while datamin:
        a = datamin.replace('\n', ' ').split(",")
        res.append(a)
        datamin = f.readline()
        pass
    ress = []
    for i in res:
        for j in i:
            print(i, j)
            ress.append(float(str(j)))
    print(ress)
    pass
    # datasmin = f.readlines()
    # datasmin1 = [i.replace('\n', ' ').replace(',', ' ') for i in datasmin]  # 删去换行符
    # print(datasmin1.split(' '))
    # for j in range(len(datasmin1)):
    #     datasmin2=[]
    #     if j < len(datasmin1):
    #         datasmin1[j].split(' ')
    #         datasmin2.append(datasmin[j])
    #         pass
    #     pass
    print(datasmin2)
    samples = np.array(datasmin2).reshape(5, 1)  # 定义数组维数
    pass
with open('Iris-setosa.txt', 'r')as ff:  # 导入多数样本
    datasmax = ff.readlines()
    datasmax1 = [i.replace('\n', '').replace(',', ' ') for i in datasmax]  # 删去换行符
    datasmax2 = [float(y) for y in datasmax1]
    major_samples = np.array(datasmax2).reshape(30, 2)
    pass
