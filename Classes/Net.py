import numpy as np
from math import exp
import random

class Neuron:
    Weights: list()
    #Biases: list()


    def __init__(self, inp_dim, rng):
        self._initializeWeights(inp_dim, rng)

    def _initializeWeights(self, inp_dim, rng):
        self.Weights = [rng() for i in range(inp_dim + 1)]

    def Evaluate(self, input):
        return np.dot(self.Weights, input)
    
    def printWeights(self):
        print(self.Weights)

    def __str__(self):
        return str(self.Weights)




class Net:
    Activations: callable
    Length: callable
    Network: list()#dict

    def __init__(self, dimensions, activations, norm):
        self.Activations = activations
        self.Length = norm
        self._initializeNet(dimensions)


    def _initializeNet(self, network_dimensions, seed=[], genFunc=lambda: random.random()):
        """
        initialize network (randomly if no seed given)
        """
        #self.Network = ['STUB']#{'STUB': 0}
        self.Network = []
        #setWeights = np.vectorize(lambda x: x.append(genFunc()))#septVigts

        for i in range (1, len(network_dimensions)):
            layer = []
            for neur in range (network_dimensions[i]):
                layer.append(Neuron(network_dimensions[i-1],  genFunc))
            self.Network.append(layer)

    def __str__(self):
        return str(self.Network)
  
    def _evaluate(self, input, curr):
        if (curr == len(self.Network)):
            return input
        output = [1]
        for neur in self.Network[curr]:
            output.append(neur.Evaluate(input))
            print(output)
        return  self._evaluate(output, curr+1)
       # self.kill
       # for i in range (self.Network[curr+1]):
       #     layer = self.Network[i]
       #     for neur in range (len(layer)):
       #         output.append(evaluate)


   # def back_forth_prop

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
n = Net([2, 3, 2, 3], [lambda x:1/(1+exp(-1*x)) for i in range(3)], lambda a:np.linalg.norm(a))#lambda a,b:abs(a-b))
#n._backProp(3, 3)
#n._backProp(4, 3)

print(n._evaluate([1, 2, 2], 0))
'''
for l in n.Network:
    for n in l:
        print(n)
    print("new layer")
'''
#
# print(n.Activations)

#neuron = Neuron(3, lambda: random.random())
#neuron.printWeights()

#print(numpy.subtract([2, 4, 2], [3, 2, 4]))
#print(numpy.subtract(3, 4))
#print(np.linalg.norm([3, 4]))
