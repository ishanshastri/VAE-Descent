import numpy as np
from math import exp
import random

class Neuron:
    Weights: list()
    #Biases: list()

    def __init__(self, inp_dim, rng):
        self._initializeWeights(inp_dim, rng)

    def _initializeWeights(self, inp_dim, rng):
        Weights = [rng() for i in range(inp_dim + 1)]

    def Evaluate(self, input):
        #w^x
        np.dot(self.Weights, input)


class Net:
    Activations: callable
    Length: callable
    Network: list()#dict

    def __init__(self, dimensions, activations, norm):
        self.Activations = activations
        self.Length = norm

        self._initializeWeights(dimensions)

    def _initializeWeights(self, dimensions, seed=[], genFunc=lambda: random.random()):
        """
        initialize network (randomly if no seed given)
        """
        #self.Network = ['STUB']#{'STUB': 0}
        self.Network = []
        #setWeights = np.vectorize(lambda x: x.append(genFunc()))#septVigts
        for l in dimensions:
            layer = [genFunc() for i in range(l)]
            self.Network.append(layer)

    def _evaluate(self, input):
        """
        Send input through network and return output (forward prop)
        """
        return 0

    def _error(self, a, b):
        """
        Get error between two datapoints
        """
        return self.Length(np.subtract(a-b))#consider outsourcing subtract

    def _backProp(self, result, target):
        """
        backpropogate errors through network recursively, update weights
        """
        self.Network.append('backpropped (!)')

    #Publics
    def GetLayer(self, depth):
        """
        Get layer at given depth
        """
        return self.Network[depth]

#Test
n = Net([3, 2, 3], 
[lambda x:1/(1+exp(-1*x)) for i in range(3)], 
lambda a:np.linalg.norm(a))#lambda a,b:abs(a-b))
n._backProp(3, 3)
n._backProp(4, 3)

#print(n.Network)
#print(n.Activations)

neuron = Neuron()

#print(numpy.subtract([2, 4, 2], [3, 2, 4]))
#print(numpy.subtract(3, 4))
#print(np.linalg.norm([3, 4]))
