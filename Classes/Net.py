import numpy as np
from math import exp
import random

def print_net(n):
    print("================")
    for l in n.Network:
        print("layer")
        for n in l:
            print(n)
    print("================")

#s = Sigmoid()
#print(s.Derivative(0))
#============================================================================
class BackProp:
    def __init__(self) -> None:
        pass

    def GetGrad(self, net, input, target, output):
        depth = len(net.Network)
        for i in range(depth-1, -1, -1):
            pass

class Neuron:
    Weights: list()
    Activation: callable

    def __init__(self, inp_dim, rng, act):
        self.Activation = act
        self._initializeWeights(inp_dim, rng)

    def _initializeWeights(self, inp_dim, rng):
        self.Weights = [rng() for i in range(inp_dim + 1)]

    def Evaluate(self, input):
        dotprod = np.dot(self.Weights, input)
        w_grads = []
        i_grads = []
        dim = len(self.Weights)
        for i in range(dim):
            w_grads.append(input[i]*self.Activation[1](dotprod))
            i_grads.append(self.Weights[i]*self.Activation[1](dotprod))
        return (self.Activation[0](dotprod), w_grads, i_grads)
    

    def printWeights(self):
        print(self.Weights)

    def __str__(self):
        return str(self.Weights)

class Net:
    Activations: callable
    Length: callable
    Network: list()

    def __init__(self, dimensions, activations, norm):
        self.Activations = activations
        self.Length = norm
        self._initializeNet(dimensions)

    def _initializeNet(self, network_dimensions, seed=[], genFunc=lambda: random.random()):
        """
        initialize network (randomly if no seed given)
        """
        self.Network = []
        #setWeights = np.vectorize(lambda x: x.append(genFunc()))#septVigts

        for i in range (1, len(network_dimensions)):
            layer = []
            for neur in range (network_dimensions[i]):
                layer.append(Neuron(network_dimensions[i-1],  genFunc, self.Activations[i-1]))
            self.Network.append(layer)

    def __str__(self):
        return str(self.Network)
  
    def _evaluate(self, input, curr):
        if (curr == len(self.Network)):
            return input[1:]
        output = [1]
        for neur in self.Network[curr]:
            output.append(neur.Evaluate(input)[0])
            #print(output)
        return self._evaluate(output, curr+1)

    #def _getNumWeights(self):
    #    return np.sum(self.Dimensions)

   #def back_forth_prop:
        #propoagate back and forth between EIT->E2->DC->four-corners-of-MC->M3->MC->repeat

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

#Test(s)
sigmoid = lambda x:1/(1+exp(-1*x))
d_dx_sigmoid = lambda x:sigmoid(x)*(1-sigmoid(x))
n = Net([2, 3, 2, 3, 5], [[sigmoid, d_dx_sigmoid] for i in range(4)], lambda a:np.linalg.norm(a))

#print("output:", n._evaluate([1, 0, 0], 0))
#print_net(n)

bp = BackProp()
bp.GetGrad(n, [], [], [])

neur = Neuron(2, lambda : 1, [sigmoid, d_dx_sigmoid])
print(neur)
print(neur.Evaluate([1, 1, 1]))
