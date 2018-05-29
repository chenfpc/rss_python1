# -*- coding: utf-8 -*- 
# 导入相应的包
from sklearn import  preprocessing
import fucntion as f
import scipy.io as sio
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import matplotlib.pylab as plt
#训练数据
trainingAll = r'C:\Users\computer\Desktop\data\all.txt'
training2_4g = r'C:\Users\computer\Desktop\data\all2_4.txt'
training5g = r'C:\Users\computer\Desktop\data\all5.txt'

#测试数据
testingAll = r'C:\Users\computer\Desktop\data\test_all.txt'
testing2_4g = r'C:\Users\computer\Desktop\data\test_2.4g.txt'
testing5g = r'C:\Users\computer\Desktop\data\test_5g.txt'

#坐标数据
cordinaryAll = r'C:\Users\computer\Desktop\data\position.txt'
cordinaryTest = r'C:\Users\computer\Desktop\data\position_test.txt'



# 数据准备
trainingSet = np.loadtxt(training2_4g)
testingSet = np.loadtxt(testing2_4g)
originalTestingSet = np.loadtxt(testing2_4g)
cordinaryAllSet = np.loadtxt(cordinaryAll)
cordinaryTestSet = np.loadtxt(cordinaryTest)
# k-means聚类
# 将原始数据做归一化处理
scaler = preprocessing.StandardScaler().fit(trainingSet)
trainingSet = scaler.transform(trainingSet)
testingSet = scaler.transform(testingSet)
x1, x2, y1, y2 = train_test_split(testingSet, cordinaryTestSet, test_size=0.3)

# classfication = f.runClassfication(trainingSet, cordinaryAllSet)

#clusterKnn算法，默认cluster个数为4
#f.runClusterKnn(trainingSet, testingSet, originalTestingSet, cordinaryAllSet, cordinaryTestSet,classfication)
# f.runClusterKnn(trainingSet,testingSet[26:35],cordinaryAllSet,cordinaryTestSet[26:35],3) #为什么一直5.7160042606
#f.runGression()




#以下测试的是仿真数据

fileName = r'C:\Users\computer\Desktop\data\workspace.mat'
data = sio.loadmat(fileName)
alldata = data["tempx"]
alldistance = data["distance"]

fileName_test = r'C:\Users\computer\Desktop\data\1.mat'
testdata = sio.loadmat(fileName_test)
alldata1 = testdata["tempx"]
alldistance1 = testdata["distance"]

fileName_test2 = r'C:\Users\computer\Desktop\data\2.mat'
testdata2 = sio.loadmat(fileName_test)
alldata2 = testdata2["tempx"]
alldistance2 = testdata2["distance"]

fileName_2 = r'C:\Users\computer\Desktop\data\workspace_2.4.mat'
data_2 = sio.loadmat(fileName)
alldata_2 = data_2["tempx"]
alldistance_2 = data_2["distance"]

fileName_test_2 = r'C:\Users\computer\Desktop\data\1_2.4.mat'
testdata_2 = sio.loadmat(fileName_test)
alldata1_2 = testdata_2["tempx"]
alldistance1_2 = testdata_2["distance"]

fileName_test2_2 = r'C:\Users\computer\Desktop\data\2_2.4.mat'
testdata2_2 = sio.loadmat(fileName_test)
alldata2_2 = testdata2_2["tempx"]
alldistance2_2 = testdata2_2["distance"]

fileName_test_24G = r'C:\Users\computer\Desktop\data\3_2.4.mat'
test_24Gdata = sio.loadmat(fileName_test_24G)
test_24G = test_24Gdata["tempx"]

fileName_test_5G = r'C:\Users\computer\Desktop\data\3.mat'
test_5Gdata = sio.loadmat(fileName_test_5G)
test_5G = test_5Gdata["tempx"]
# 数据准备

# k-means聚类
# 将原始数据做归一化处理
# scaler = preprocessing.StandardScaler().fit(alldata)
#alldata = scaler.transform(alldata)
# classfication = f.runClassfication(alldata, alldistance)
# f.runClusterKnnTest(alldata,alldistance,alldata1,scaler,classfication)
#f.runKnnSimulate(alldata,alldistance,alldata1,7)
f.runGression(alldata_2, alldata1_2, alldata2_2, alldistance, alldata, alldata1, alldata2, test_24G, test_5G)
