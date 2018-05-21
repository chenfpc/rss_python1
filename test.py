import numpy as np
data2_4 = np.loadtxt(r"/Users/computer/Desktop/data/all2_4.txt")
data5 = np.loadtxt(r"/Users/computer/Desktop/data/all5.txt")
import matplotlib.pyplot as plt
lenth = len(data2_4)
dictionary = {}
data1 = data2_4[:,1]
data2 = data5[:,1]

data1.sort()
data2.sort()

for i in range(lenth):
    if(data2[i] in dictionary.keys()):
        dictionary[data2[i]] =  dictionary[data2[i]] + 1
    else:
        dictionary[data2[i]] = 1

x = []
for i in dictionary.values():
    x.append(i)

y = []
for j in dictionary.keys():
    y.append(j)
print(len(x),len(y))
plt.bar(x,y,color="rgb")
#plt.show()

x = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
x1 = np.reshape(x,(2,2,4))
print(x1[0])

# 60,60
x = np.arange(64)
b = np.reshape(x,(4,4,4),order='A')
#print(b)

def x():
    return (4, 3), (2, 3)


s = x();
print(s[0][1])

a = np.array([[1, 2, 1], [3, 4, 9]])
b = np.array([[5, 6], [7, 8]])
print(a.shape, b.shape)
c = np.concatenate((a, b), axis=0)
