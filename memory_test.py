import numpy as np
from time import time

np.random.seed(3)
x = np.random.random((5000, 3))
n = x.shape[0]

class Graph(object):
    def __init__(self, numNodes):
        self.adjacencyMatrix = []
        for i in range(numNodes):
            self.adjacencyMatrix.append([0 for i in range(numNodes)])
        self.numNodes = numNodes

start = time()
A = Graph(n).adjacencyMatrix
print('Elapsed: ', time() - start)