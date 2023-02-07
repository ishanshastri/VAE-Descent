import numpy
from math import exp

class Net:
    Activation: callable
    Distance: callable
    Network: list()#dict

    def __init__(self, dimensions, activation, norm):
        self.Activation = activation
        self.Distance = norm

        self._initializeWeights(dimensions)

    def _initializeWeights(self, dimensions, seed=[]):
        """
        initialize network (randomly if no seed given)
        """
        self.Network = ['STUB']#{'STUB': 0}

    def _evaluate(self, input):
        """
        Send input through network and return output (forward prop)
        """
        return 0

    def _backProp(self, result, target):
        """
        backpropogate errors through network recursively, update weights
        """
        self.Network.append('backpropped')

    #Publics
    def GetLayer(self, depth):
        """
        Get layer at given depth
        """
        return self.Network[depth]

#Test
n = Net([], lambda x:1/(1+exp(-1*x)), lambda a,b:abs(a-b))
n._backProp(3, 3)
print(n.Network)