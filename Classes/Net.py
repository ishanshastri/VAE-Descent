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

    def Evaluate(self, input, input_roc=[]):#input_roc might be redundant (monkey moment(s))
        dotprod = np.dot(self.Weights, input)
        w_grads = []
        i_grads = []
        dim = len(self.Weights)
        if input_roc==[]:
            input_roc = [1 for i in range(dim)]
        for i in range(dim):
            w_grads.append(input[i]*self.Activation[1](dotprod))
            i_grads.append(input_roc[i]*self.Weights[i]*self.Activation[1](dotprod))
        return (self.Activation[0](dotprod), w_grads, i_grads)

    def printWeights(self):
        print(self.Weights)

    def __str__(self):
        return str(self.Weights)

class Net:
    Activations: callable
    Length: callable
    Network: list()
    GradientVector: list()

    _numWeights: int

    def __init__(self, dimensions, activations, norm):
        self.Activations = activations
        self.Length = norm
        self._numWeights=0
        for d in dimensions[1:]:
            self._numWeights+=(d+1)
        #print(self._numWeights)
        self._initializeNet(dimensions)

    def _initializeNet(self, network_dimensions, seed=[], genFunc=lambda: random.random()):
        """
        initialize network (randomly if no seed given)
        """
        self.Network = []
        self.GradientVector = []#[0 for i in range(self._numWeights)]
        #setWeights = np.vectorize(lambda x: x.append(genFunc()))#septVigts

        for i in range (1, len(network_dimensions)):
            layer = []
            for neur in range (network_dimensions[i]):
                layer.append(Neuron(network_dimensions[i-1],  genFunc, self.Activations[i-1]))
            self.Network.append(layer)

    def __str__(self):
        return str(self.Network)
  
    def _evaluate(self, input, curr=0, derivs=[]):
        if (curr == len(self.Network)):
            return (input[1:], derivs)
        output = [1]
        partials = []

        for neur in self.Network[curr]:
            res = neur.Evaluate(input)
            #self.GradientVector.extend(res[1])
            output.append(res[0])
            partials.append((res[1], res[2]))

        derivs.append(partials)
        return self._evaluate(output, curr+1, derivs)

    #def _numWeights(self):
    #    return np.sum(self.Dimensions)

   #def back_forth_prop:
        #propoagate back and forth between EIT->E2->DC->four-corners-of-MC->M3->MC->repeat

    def _error(self, a, b):
        """
        Get error between two datapoints
        """
        return self.Length[0](np.subtract(a-b))#consider outsourcing subtract

    def _backProp(self, input, target, prev=[]):#target includes the 1
        """
        backpropogate errors through network recursively, update weights
        """
        #self.Network.append('backpropped (!)') A+ implementation
        #neurror

        self.GradientVector = []
        result = self._evaluate(input)
        err = self._error(result, target)
        print(err)
        layers = len(result[1])
        if prev==[]:
            prev = [1 for i in range(len(target))]
        for i in range(layers-1, -1, -1):
            for neur in result[1][i]:
                for weight in range(len(result[1][i][0])):
                    self.GradientVector.append()

    #def _calcGrad()

    #Publics
    def GetLayer(self, depth):
        """
        Get layer at given depth
        """
        return self.Network[depth]

#Test(s)
sigmoid = lambda x:1/(1+exp(-1*x))
d_dx_sigmoid = lambda x:sigmoid(x)*(1-sigmoid(x))
#n = Net([2, 3, 2, 3, 5], [[sigmoid, d_dx_sigmoid] for i in range(4)], [lambda a:np.linalg.norm(a), lambda a: 2*a])
n = Net([2, 1, 1, 2], [[sigmoid, d_dx_sigmoid] for i in range(3)], [lambda a:np.linalg.norm(a), lambda a: 2*a])

res = n._evaluate([1, 1, 1])
print("output:", res[0])
print("derivs: ", res[1])
#print_net(n)
#print(n.GradientVector)
bp = BackProp()
bp.GetGrad(n, [], [], [])

for d in res[1]:
    print("L")
    for p in d:
        print(p)

neur = Neuron(2, lambda : 1, [sigmoid, d_dx_sigmoid])
#print(neur)
#print(neur.Evaluate([1, 1, 1])[0])

#print(neur.Evaluate([1, 1, 1])[1])