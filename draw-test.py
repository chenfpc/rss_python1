import matplotlib.pyplot as plt
styles = ["-", "-"]
data = [[1,2,3,4,5],[2.3,1.2,3.5,5.6]]

import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

sample = np.random.uniform(0, 1, 50)
ecdf = sm.distributions.ECDF(sample)

x = np.linspace(min(sample), max(sample))
y = ecdf(x)
plt.step(x, y)
plt.show()

def drawCdf(drawData):
    index = 0
    for data in drawData:
        data.sort()
        plotdata = [[],[]]
        plotdata[0] = data
        count = len(plotdata[0])
        for i in range(count):
            plotdata[1].append((i+1)/count)
        plt.plot(plotdata[0],plotdata[1],styles[index],lineWidth=2)
        index = index + 1
    plt.show()

#drawCdf(data)