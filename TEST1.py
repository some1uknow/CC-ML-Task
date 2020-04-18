import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
iris = load_iris()
#data = iris.data.T
#print(data)
XTrain, XTest, YTrain, YTest = train_test_split(iris['data'], iris['target'], random_state = 0)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(XTrain, YTrain)
acc = 100 * (knn.score(XTest, YTest))
print(f"{acc}% accuracy")
 

    
