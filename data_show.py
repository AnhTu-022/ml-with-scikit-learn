
import pandas
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepalum-laenge', 'sepalum-breiten', 'petalum-länge', 'petalum-breiten', 'klassen']
data = pandas.read_csv(url, names=names)

#print(dataset.shape)

#print(dataset.head(10))

#print(dataset.describe())

#print(dataset.groupby('klassen').size())

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# histograms
#dataset.hist()

## scatter plot matrix
scatter_matrix(data)

plt.show()