# -*- coding: utf-8 -*-
# 导入相应的包
from sklearn import  preprocessing
import fucntion as f
import scipy.io as sio
import math
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import matplotlib.pylab as plt
from sklearn import linear_model
#训练数据
trainingAll = r'/Users/computer/Desktop/data/all.txt'
training2_4g = r'/Users/computer/Desktop/data/all2_4.txt'
training5g = r'C:/Users/computer/Desktop/data/all5.txt'

#测试数据
testingAll = r'/Users/computer/Desktop/data/test_all.txt'
testing2_4g = r'/Users/computer/Desktop/data/test_2.4g.txt'
testing5g = r'/Users/computer/Desktop/data/test_5g.txt'

#坐标数据
cordinaryAll = r'/Users/computer/Desktop/data/position.txt'
cordinaryTest = r'/Users/computer/Desktop/data/position_test.txt'



# 数据准备
trainingSet = np.loadtxt(training2_4g)
testingSet = np.loadtxt(testing2_4g)
cordinaryAllSet = np.loadtxt(cordinaryAll)
cordinaryTestSet = np.loadtxt(cordinaryTest)
# k-means聚类
# 将原始数据做归一化处理
"""
scaler = preprocessing.StandardScaler().fit(trainingSet)
trainingSet = scaler.transform(trainingSet)
testingSet = scaler.transform(testingSet)
"""

#针对Wi-Fi 1 坐标为（4.8 6.6)   先2.4g
x = trainingSet[126:217,0]

x1 = []
for i in range(7):
    x1.append(trainingSet[132+13*i,0])
y1 = []
for j in range(7):
    if j == 0:
        y1.append(0.7)
    elif j == 1:
        y1.append(1.8)
    else:
        y1.append(1.8+(j-1)*0.8)
y = np.tile(np.array([4.8,6.6]),(len(x),1))


d = cordinaryAllSet[126:217,:] - y
d = d ** 2
d = d.sum(axis=1)
d = d**0.5

x1 = np.array(x1)
y1 = np.array(y1)
regr = linear_model.LinearRegression()
regr.fit(y1.reshape(-1,1),x1)
plt.scatter(y1,x1,color='blue')
plt.plot(y1,regr.predict(y1.reshape(-1,1)),color='red')
plt.show()
print(regr.coef_,regr.intercept_)


