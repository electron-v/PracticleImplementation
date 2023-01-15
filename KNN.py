
import numpy as np
from collections import Counter

#Created Data
data = np.array([[1,1],[1,2],[2,1],[3,2],[1,3],[8,6],[8,7],[9,7],[10,8]])
target = [0,0,0,0,0,1,1,1,1]

#To calculate ecludian distance
def ecludian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


def knn(data,target,observation,k):
    distances = []

    for i,x in enumerate(data):
        distances.append([ecludian_distance(x,observation),i])
        distances = sorted(distances)

    vote = []
    for i in range(k):
        res = target[distances[i][1]]
        vote.append(res)

    #caclulate the class with maximum votes
    return Counter(vote).most_common()[0][0]

print(knn(data,target,np.array([2,3]),3))
