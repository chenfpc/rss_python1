# -*- coding: utf-8 -*-
# 导入相应的包
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch
# import Matlab as m
import sklearn as sk
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import matplotlib.pylab as plt

"""
    testData: 表示测试点的数据（RSS和位置）
    position_test: 只是包含测试点的位置
    dataTag: kmeans分类的类别
    centroid: 每个类别的中心
"""

# 训练数据
trainingAll = r'/Users/computer/Desktop/data/all.txt'
training2_4g = r'/Users/computer/Desktop/data/all2_4.txt'
training5g = r'C:/Users/computer/Desktop/data/all5.txt'

# 测试数据
testingAll = r'/Users/computer/Desktop/data/test_all.txt'
testing2_4g = r'/Users/computer/Desktop/data/test_2.4g.txt'
testing5g = r'/Users/computer/Desktop/data/test_5g.txt'

# 坐标数据
cordinaryAll = r'/Users/computer/Desktop/data/position.txt'
cordinaryTest = r'/Users/computer/Desktop/data/position_test.txt'


def runGression():
    predict_5g = runClusterKnn(training5g, testing5g, cordinaryAll, cordinaryTest, 4)
    predict_2g = runClusterKnn(training2_4g, testing2_4g, cordinaryAll, cordinaryTest, 4)
    # print(len(predict_2g),predict_2g[0],len(predict_5g),predict_5g[0])
    reg = sk.linear_model.LinearRegression()
    reg.fit()


def runKnnSimulate(alldata, alldistance, alldata1, k):
    # 测试集划分
    traindata, testdata, train_distance, test_distance = train_test_split(alldata1, alldistance, train_size=0.9)
    # training set 就是包含3600个点所有数据。注意，training set 的distance 和 testing set 的distance是同一个文件
    trainingSet_cordinary = np.column_stack((traindata, train_distance))  # 这里直接随机选区五分之一的数据点作为测试点，然后取knn = 7进行仿真。
    testingSet_cordinary = np.column_stack((testdata, test_distance))
    cordinaryTestSet = test_distance

    result = 0

    predict_cordinary = [None] * len(testdata)
    for i in range(len(testdata)):
        knnResult = calculateCordinary(k, trainingSet_cordinary, testingSet_cordinary[i], i, cordinaryTestSet)
        result += knnResult[0]
        predict_cordinary[i] = knnResult[1]
    print("平均误差为")
    print(result / len(testdata))
    return predict_cordinary


def runKnnReality(trainingSet, testingSet, cordinaryAllSet, cordinaryTestSet, k):
    trainingSet_cordinary = np.column_stack((trainingSet, cordinaryAllSet))

    testingSet_cordinary = np.column_stack((testingSet, cordinaryTestSet))

    # 针对nlos 两个教室的情况
    """
    testingSet = testingSet[25:46, :]
    testingSet_cordinary = testingSet_cordinary[25:46, :]
    cordinaryTestSet = cordinaryTestSet[25:46, :]
    """

    # print(trainingSet[0],"is",len(trainingSet))

    result = 0

    predict_cordinary = [None] * len(testingSet)
    for i in range(len(testingSet)):
        knnResult = calculateCordinary(k, trainingSet_cordinary, testingSet_cordinary[i], i, cordinaryTestSet)
        result += knnResult[0]
        predict_cordinary[i] = knnResult[1]
    print("平均误差为")
    print(result / len(testingSet))
    return predict_cordinary


def runClassfication(trainingSet, cordinarySet):
    trainingSet1 = np.row_stack((trainingSet[0:63, :], trainingSet[126:217, :]))
    cordinarySet1 = np.row_stack((cordinarySet[0:63, :], cordinarySet[126:217, :]))
    trainingSet2 = trainingSet[63:126, :]
    cordinarySet2 = cordinarySet[63:126, :]
    trainingSet3 = trainingSet[217:298]
    cordinarySet3 = cordinarySet[217:298]
    trainingSet4 = trainingSet[298:355]
    cordinarySet4 = cordinarySet[298:355]
    trainingSet5 = trainingSet[355:467]
    cordinarySet5 = cordinarySet[355:467]

    datax1 = np.concatenate((trainingSet1, cordinarySet1), axis=1)
    datax2 = np.concatenate((trainingSet2, cordinarySet2), axis=1)
    datax3 = np.concatenate((trainingSet3, cordinarySet3), axis=1)
    datax4 = np.concatenate((trainingSet4, cordinarySet4), axis=1)
    datax5 = np.concatenate((trainingSet5, cordinarySet5), axis=1)
    # data1 = runCluster(trainingSet1, cordinarySet1, 7)
    # data2 = runCluster(trainingSet2, cordinarySet2, 7)
    # data3 = runCluster(trainingSet3, cordinarySet3, 7)
    # data4 = runCluster(trainingSet4, cordinarySet4, 7)
    # data5 = runCluster(trainingSet5, cordinarySet5, 7)
    # return data1, data2, data3, data4, data5

    data1 = subArea(trainingSet1)
    data2 = subArea(trainingSet2)
    data3 = subArea(trainingSet3)
    data4 = subArea(trainingSet4)
    data5 = subArea(trainingSet5)

    return (datax1, data1), (datax2, data2), (datax3, data3), (datax4, data4), (datax5, data5)
    # trainingSet = np.reshape(trainingSet,(60,60,12))
    # cordinarySet = np.reshape(cordinarySet,(60,60,2))
    # trainingSet1 = np.zeros((900,12))
    # cordinarySet1= np.zeros((900,2))
    # trainingSet2 = np.zeros((900,12))
    # cordinarySet2= np.zeros((900,2))
    # cordinarySet3 =np.zeros((900,2))
    # trainingSet3 = np.zeros((900,12))
    # trainingSet4 = np.zeros((900,12))
    # cordinarySet4= np.zeros((900,2))
    #
    # index = 0
    # for i in range(30):
    #     for j in range(30):
    #         trainingSet1[index] = trainingSet[i,j,:]
    #         cordinarySet1[index] = cordinarySet[i,j,:]
    #         trainingSet2[index] = trainingSet[i,j+30,:]
    #         cordinarySet2[index] = cordinarySet[i,j+30,:]
    #         index = index + 1
    # index = 0
    # for i in range(30):
    #     for j in range(30):
    #         trainingSet4[index] = trainingSet[i + 30, j, :]
    #         cordinarySet4[index] = cordinarySet[i + 30, j, :]
    #         trainingSet3[index] = trainingSet[i + 30, j+30, :]
    #         cordinarySet3[index] = cordinarySet[i + 30, j+30, :]
    #         index = index + 1
    #
    # print(len(trainingSet1),len(cordinarySet1))
    #
    #
    #
    # data1 = runCluster(trainingSet1, cordinarySet1,80)
    # data2 = runCluster(trainingSet2, cordinarySet2,80)
    # data3 = runCluster(trainingSet3, cordinarySet3,80)
    # data4 = runCluster(trainingSet4, cordinarySet4,80)
    #
    # return data1, data2, data3, data4


def subArea(featureSet):
    length = len(featureSet)
    centroid = featureSet.sum(axis=0)
    centroid = centroid / length

    return centroid


def runCluster(trainingSet, cordinarySet, clusters):
    """
    #trainingSet = np.row_stack((trainingSet[0:63,:],trainingSet[126:217,:]))
    print(len(trainingSet))
    meandistortions = []
    for k in range(100):
        kmeans = KMeans(n_clusters=(k + 1))
        kmeans.fit(trainingSet)
        meandistortions.append(
            sum(np.min(cdist(trainingSet, kmeans.cluster_centers_, 'euclidean'), axis=1)) / trainingSet.shape[0])

    plt.plot(np.linspace(0,100,100), meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('平均畸变程度')
    plt.title('用肘部法则来确定最佳的K值');
    plt.show()
    """

    trainingSet_cordinary = np.column_stack((trainingSet, cordinarySet))
    cluster = clusters
    centroid = kmeans(trainingSet, cluster)[0]
    dataTag = [list() for i in range(cluster)]
    # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
    label = vq(trainingSet, centroid)[0]
    for i in range(len(trainingSet)):
        dataTag[label[i]].append(trainingSet_cordinary[i])

    return dataTag, centroid


def runClusterKnn(trainingSet, testingSet, originalTestingSet, cordinaryAllSet, cordinaryTestSet, classfication):
    trainingSet_cordinary = np.column_stack((trainingSet, cordinaryAllSet))

    testingSet_cordinary = np.column_stack((testingSet, cordinaryTestSet))

    """
        #针对nlos
        testingSet = testingSet[25:46, :]
        testingSet_cordinary = testingSet_cordinary[25:46, :]
        cordinaryTestSet = cordinaryTestSet[25:46, :]
    """

    # https://blog.csdn.net/u013719780/article/details/51755124

    """
    meandistortions = []
    for k in range(30):
        kmeans = KMeans(n_clusters= (k+1))
        kmeans.fit(trainingSet)
        meandistortions.append(sum(np.min(cdist(trainingSet, kmeans.cluster_centers_, 'euclidean'), axis=1)) / trainingSet.shape[0])


    plt.plot(range(30), meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('平均畸变程度')
    plt.title('用肘部法则来确定最佳的K值');
    plt.show() 
    """

    # 使用层次聚类
    # dismat = pdist(traindata,"euclidean") #点与点之间的距离矩阵，用欧式距离
    # z = sch.linkage(dismat,method="average")
    # #将层级聚类结果以树状图表示出来
    # p = sch.dendrogram(z)
    # plt.savefig("plot_dendrogram.png")
    # cluster = sch.fcluster(z,t=1,criterion='inconsistent')

    # 使用kmeans函数进行聚类,输入第一维为数据,第二维为聚类个数k.
    # 有些时候我们可能不知道最终究竟聚成多少类,一个办法是用层次聚类的结果进行初始化.当然也可以直接输入某个数值.
    # k-means最后输出的结果其实是两维的,第一维是聚类中心,第二维是损失distortion,我们在这里只取第一维,所以最后有个[0]

    result = clusterKNN(testingSet_cordinary, originalTestingSet, cordinaryTestSet, classfication)
    print("平均误差为")
    print(result[0])
    return result[1]


"""
  注意：这是runClusterKnn的原代码
  def runClusterKnn(trainingSetString, testingSetString, cordinaryAllString, cordinaryTestString, clusters):
    # 分为4类
    data0 = []
    data1 = []
    data2 = []
    data3 = []

    # 数据准备
    trainingSet = np.loadtxt(trainingSetString)
    testingSet = np.loadtxt(testingSetString)
    cordinaryAllSet = np.loadtxt(cordinaryAllString)
    cordinaryTestSet = np.loadtxt(cordinaryTestString)
    # k-means聚类
    # 将原始数据做归一化处理
    scaler = preprocessing.StandardScaler().fit(trainingSet)
    trainingSet = scaler.transform(trainingSet)
    trainingSet_cordinary = np.column_stack((trainingSet, cordinaryAllSet))
    testingSet = scaler.transform(testingSet)
    testingSet_cordinary = np.column_stack((testingSet, cordinaryTestSet))

  
        #针对nlos
        testingSet = testingSet[25:46, :]
        testingSet_cordinary = testingSet_cordinary[25:46, :]
        cordinaryTestSet = cordinaryTestSet[25:46, :]
   

    # 使用kmeans函数进行聚类,输入第一维为数据,第二维为聚类个数k.
    # 有些时候我们可能不知道最终究竟聚成多少类,一个办法是用层次聚类的结果进行初始化.当然也可以直接输入某个数值.
    # k-means最后输出的结果其实是两维的,第一维是聚类中心,第二维是损失distortion,我们在这里只取第一维,所以最后有个[0]
    # centroid = kmeans(data,max(cluster))[0]
    centroid = kmeans(trainingSet, clusters)[0]
    # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
    label = vq(trainingSet, centroid)[0]
    for i in range(len(trainingSet)):
        if label[i] == 0:
            data0.append(trainingSet_cordinary[i])
        elif label[i] == 1:
            data1.append(trainingSet_cordinary[i])
        elif label[i] == 2:
            data2.append(trainingSet_cordinary[i])
        elif label[i] == 3:
            data3.append(trainingSet_cordinary[i])
    # print(len(data0),len(data1),len(data2),len(data3))

    dataTag = [data0, data1, data2, data3]
    result = clusterKNN(testingSet_cordinary, cordinaryTestSet, dataTag, centroid)
    print("平均误差为")
    print(result[0])
    return result[1]
"""


def judge(testPoint):
    result = ""
    wifi1 = testPoint[0]
    wifi2 = testPoint[4]
    wifi3 = testPoint[8]
    # if -19 >= wifi1 >= -57.65:
    #     result = '0'
    # elif -23 >= wifi2 >= -61.65:
    #     result = '1'
    # elif -17 >= wifi3 >= -55.65:
    #     result = '2'
    # else:
    #     result = '3'

    if -61 <= wifi1 <= -22:
        if wifi3 <= -60:
            result = "AC"
        else:
            result = "D"
    elif wifi1 <= -62:
        if wifi3 <= -57:
            result = "B"
        elif wifi2 <= -56:
            result = "F"
        else:
            result = "E"

    return result


def clusterKNN(testData, originalTestSet, positions_test, classfication):
    # testdata是含有position的
    index = 0
    error = 0
    len1 = len(positions_test)
    # 新建一个predict_cordinary保存 每个点使用clusterknn算法生成的坐标
    predict_cordinary = [None] * len1
    for i in range(len1):
        label = judgeCluster(testData[i][:-2], originalTestSet[i][:], classfication)

        # data = label[0]
        # clusterIndex = label[1]
        # result = calculateCordinary(3, data[clusterIndex], testData[i], i, positions_test)
        result = calculateCordinary(3, label[0], testData[i], i, positions_test)
        error = error + result[0]
        predict_cordinary[i] = result[1]
    return error / len(positions_test), predict_cordinary


def judgeCluster(source, source1, classfication):
    clusterResult = judge(source1)
    data = np.array([])

    if clusterResult == "AC":
        data = classfication[0]
    if clusterResult == "B":
        data = classfication[1]
    if clusterResult == "D":
        data = classfication[2]
    if clusterResult == "E":
        data = classfication[3]
    if clusterResult == "F":
        data = classfication[4]

    # if clusterResult == "0":
    #     data = classfication[0]
    # if clusterResult == "1":
    #     data = classfication[1]
    # if clusterResult == "2":
    #     data = classfication[2]
    # if clusterResult == "3":
    #     data = classfication[3]

    # index = 0
    # temp = 100000
    # centroid = data[1]
    # # print(source1, centroid[0])
    # for i in range(len(centroid)):
    #     sub = source - centroid[i]
    #     # print("the sub is ", sub)
    #     differ = sub ** 2
    #     # print("DIFFER",differ)
    #     result = differ.sum(axis=0)
    #     if result < temp:
    #         temp = result
    #         index = i
    # return data[0], index
    return data


"""
  K:选取几个k近邻点
  data: 所处聚类的数据
  point: 测试点
  index: 第几个点
  position_test: 这些测试点的位置
  该函数目的是为了求出给定点point在data聚类的k个近邻点，返回的是定位的误差
"""


def calculateCordinary(k, data, point, index, positions_test):
    result = knn(point, data, k)
    print("knn得到的平均位置", result)  # 定位结果返回
    predic_position = result
    print("实际位置：", positions_test[index])
    result = (result - positions_test[index]) ** 2
    result = result.sum(axis=0) ** 0.5
    print("误差为", result)
    return result, predic_position


"""
    inX: 测试点
    dataset:所属聚类的数据
    
"""


def knn(inX, dataset, k):
    # print(dataset[0][19])
    set = np.array(dataset)
    set = set[:, :-2]
    datasetSize = len(set)
    # print(datasetSize)
    x = inX[:][:-2]
    # print(x)
    diffMat = np.tile(x, (datasetSize, 1)) - set
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    # print("sqdistance",sqDistance)
    # 对平方和进行开根号
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()
    sumx = 0
    sumy = 0
    length = len(dataset[0])

    """
    这个函数是用于weight取平均
      k_sum = 0
    weight = []

    for i in range(k):
        temp = x - dataset[sortedDistIndicies[i]][:-2]
        temp = temp**2
        temp = temp.sum(axis=0)
        temp = temp**0.5
        temp = 1 / temp
        k_sum = k_sum + temp
        weight.append(temp)
    for i in range(k):
        sumx = sumx + (weight[i] / k_sum) * dataset[sortedDistIndicies[i]][length - 2]
        sumy = sumy + (weight[i] / k_sum) * dataset[sortedDistIndicies[i]][length - 1]
    return sumx, sumy
        
    """
    if k > datasetSize:
        for i in range(datasetSize):
            sumx = sumx + dataset[sortedDistIndicies[i]][length - 2]
            sumy = sumy + dataset[sortedDistIndicies[i]][length - 1]
            return sumx / datasetSize, sumy / datasetSize
    else:
        for i in range(k):
            sumx = sumx + dataset[sortedDistIndicies[i]][length - 2]
            sumy = sumy + dataset[sortedDistIndicies[i]][length - 1]
            # print(dataset[sortedDistIndicies[i]][18], dataset[sortedDistIndicies[i]][19])

        return sumx / k, sumy / k


def runClusterKnnTest(alldata, alldistance, alldata1, scaler, classfication):
    # 测试集划分
    traindata, testdata, train_distance, test_distance = train_test_split(alldata1, alldistance, train_size=0.8)
    originalData = testdata
    testdata = scaler.transform(testdata)
    traindata = scaler.transform(traindata)
    trainingSet_cordinary = np.column_stack((traindata, train_distance))
    testingSet_cordinary = np.column_stack((testdata, test_distance))
    cordinaryTestSet = test_distance

    # x = alldata1;
    # x = np.reshape(x,(60,60,12))
    # y = alldistance;
    # y = np.reshape(y,(60,60,2))
    #
    # print("jishu")
    # test1 = x[25][47][:]
    # test2 = scaler.transform(test1.reshape(1,-1))
    # print(test2)
    # test_distance1 = y[25,47,:]
    # print(test_distance1.shape)
    # test2 = np.reshape(test2,(1,12))
    # test_distance1 = np.reshape(test_distance1,(1,2))
    # test1 = np.reshape(test1,(1,12))
    # print(test2.shape,test_distance1.shape)

    runClusterKnn(alldata, testdata, originalData, alldistance, test_distance, classfication)
    """
    #https://blog.csdn.net/u013719780/article/details/51755124
    meandistortions = []
    for k in range(100):
        kmeans = KMeans(n_clusters= (k+1))
        kmeans.fit(traindata)
        meandistortions.append(sum(np.min(cdist(traindata, kmeans.cluster_centers_, 'euclidean'), axis=1)) / traindata.shape[0])


    plt.plot(range(100), meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('平均畸变程度')
    plt.title('用肘部法则来确定最佳的K值');
    plt.show()
    """

    # 使用层次聚类
    # dismat = pdist(traindata,"euclidean") #点与点之间的距离矩阵，用欧式距离
    # z = sch.linkage(dismat,method="average")
    # #将层级聚类结果以树状图表示出来
    # p = sch.dendrogram(z)
    # plt.savefig("plot_dendrogram.png")
    # cluster = sch.fcluster(z,t=1,criterion='inconsistent')

    # 使用kmeans函数进行聚类,输入第一维为数据,第二维为聚类个数k.
    # 有些时候我们可能不知道最终究竟聚成多少类,一个办法是用层次聚类的结果进行初始化.当然也可以直接输入某个数值.
    # k-means最后输出的结果其实是两维的,第一维是聚类中心,第二维是损失distortion,我们在这里只取第一维,所以最后有个[0]
    """
    cluster = 70
    centroid = kmeans(traindata, cluster)[0]
    dataTag = [list() for i in range(cluster)]
    # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
    label = vq(traindata, centroid)[0]
    for i in range(len(traindata)):
        dataTag[label[i]].append(trainingSet_cordinary[i])
    result = clusterKNN(testingSet_cordinary, cordinaryTestSet, dataTag, centroid)
    print("平均误差为")
    print(result[0])
    return result[1]
    """
