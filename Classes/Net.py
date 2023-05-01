import numpy as np
from math import exp
import random
import matplotlib.pyplot as plt

def print_net(n):
    print("================")
    for l in n.Network:
        print("layer")
        for n in l:
            print(n)
    print("================")

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
        #Append cost layer
        #c_neur = Neuron(2, None, None, True)

    def __str__(self):
        return str(self.Network)
  
    def _evaluate(self, input, curr=0, derivs=[]):
        #print(curr)
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

   #def back_forth_prop:
        #propoagate back and forth between EIT->E2->DC->four-corners-of-MC->M3->MC->repeat

    def _error(self, a, b):
        """
        Get error between two datapoints
        """
        return self.Length[0](np.subtract(a-b))#consider outsourcing subtract

    def _backProp_(self, input, target, prev=[]):#target includes the 1
        """
        backpropogate errors through network recursively, update weights
        """

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

    def _backProp_n(self, output, target, blayer = 0, prev = []):#prev is partial derivs of cost function w.r.t. last layer output
        local_partials_d_dw = output[1][0]
        local_partials_d_do = output[1][1]
        cur_grads = [local_partials_d_dw[i]*prev[i] for i in range(len(prev))]

    def _getCost(self, hypo, target):
        diff = np.subtract(hypo, target)
        grad = [self.Length[1](diff[i]) for i in range(len(diff))]
        return [self.Length[0](diff), grad]

    def BackProp(self, output, layer, out_d_d, grad, l_rate = 0.03, lionless = False):
        """
        Recursive implementation of the Gradient Descent Algorithm
        
        By default uses LION optimization (faster, more effective in practise), 
        otherwise keeps all directional information of the gradient
        """
        #Return gradient vector once all weights have been updated
        if layer<0:
            return grad

        #Initialize gradient vec
        cur_grads = []
        l = output[1][layer]
        prev_outs = [0 for i in range(len(l[0][0]))]
        for i in range(len(l)): #loop thru each neuron in layer 
            neuron = l[i]
            d_dw = neuron[0] #derivs wrt weights
            d_dx = neuron[1] #derivs wrt inputs

            temp_outs = []
            for j in range(len(d_dw)):
                #Add to gradient vector, get partial derivs w.r.t outputs
                cur_grads.append(out_d_d[i]*d_dw[j])
                temp_outs.append(out_d_d[i]*d_dx[j])

                #Calculate vector component using lion
                lion = 0
                if out_d_d[i]*d_dw[j] < 0:
                    lion = -1
                elif out_d_d[i]*d_dw[j] > 0:
                    lion = 1

                #Bypass for lion function
                if lionless:
                    lion = out_d_d[i]*d_dw[j]
                self.Network[layer][i].Weights[j] -= lion*l_rate

            #Applying sum-rule of partial derivs
            prev_outs = np.add(prev_outs, temp_outs) 

        #Pass on relevant information to update weights in next layer (moving backwards)
        return self.BackProp(output, layer-1, prev_outs, grad + cur_grads)

    def GetLayer(self, depth):
        """
        Get layer at given depth
        """
        return self.Network[depth]

t_net = Net([1, 1, 2, 1], [[lambda x:x, lambda x:1] for i in range(3)], [lambda a:np.linalg.norm(a), lambda a: 2*a])
r1 = t_net._evaluate([1, 2]) #input of 2 (1 is bias), -> we want output 4 (trying to recreate *2 function)
print("Layers:")
for l in t_net.Network:
    print("L")
    for n in l:
        print(n)
print("Derivs:")
for d in r1[1]:
    print("L")
    for p in d:
        print(p)

costs = []
c1s = []
c2s = []
results = []
mini = None
for i in range(100):
    input = np.random.randint(1, 100)
    r = t_net._evaluate([1, input])#input])
    #print("#", i, r[0])
    cost = t_net._getCost(r[0], 2*input) #tuple of cost and deriv wrt input
    costs.append(abs(cost[0]))
    if mini == None or mini > cost[0]:
        mini = cost[0]
    if i > 500 and abs(cost[0]) < min(0.05, mini):
        break
    #grad = t_net._susser(r, 2, cost[1], [], (1.0/(i+2)), i) #result, starting layer #, cost deriv
    l = False
    if i > 100:
        l = True
    grad = t_net.BackProp(r, 2, cost[1], [], (1.0/(i+2))) #result, starting layer #, cost deriv
    results.append(r)
#for l in t_net.Network:
#print(costs)
print(t_net._evaluate([1, 8])[0])
print(t_net._evaluate([1, 3])[0])
#print(results[len(results)-1][0])
print("cost: ", costs[len(costs)-1])
#c1s = costs
plt.plot(costs, 'g')
#plt.show()
costs = []
net_2 = t_net = Net([1, 4, 4, 4, 7, 3, 1], [[lambda x:x, lambda x:1] for i in range(6)], [lambda a:np.linalg.norm(a), lambda a: 2*a])
for i in range(10000):
    input = np.random.randint(1, 1000) #2
    r = net_2._evaluate([1, input])#input])
    cost = net_2._getCost(r[0], 2*input) #tuple of cost and deriv wrt input
    if i > 500 and abs(cost[0]) < 0.0015:
        break
    costs.append(abs(cost[0]))
    grad = net_2.BackProp(r, 2, cost[1], [], (1.0/(i+2))) #result, starting layer #, cost deriv
print(net_2._evaluate([1, 8])[0])
print(net_2._evaluate([1, 3])[0])
print("cost: ", costs[len(costs)-1])
plt.plot(costs, 'b')
#plt.show()

#deep activatiosn
sigmoid = lambda x:1/(1+exp(-1*x))
d_dx_sigmoid = lambda x:sigmoid(x)*(1-sigmoid(x))

net_3 = t_net = Net([1, 4, 4, 4, 7, 3, 1], [[sigmoid, d_dx_sigmoid] for i in range(3)] + [[lambda x:x, lambda x:1] for i in range(3)], [lambda a:np.linalg.norm(a), lambda a: 2*a])
for i in range(1000):
    input = np.random.randint(1, 1000) #2
    r = net_2._evaluate([1, input])#input])
    cost = net_3._getCost(r[0], 2*input) #tuple of cost and deriv wrt input
    if i > 500 and abs(cost[0]) < 0.0015:
        break
    costs.append(abs(cost[0]))
    grad = net_3.BackProp(r, 2, cost[1], [], (1.0/(i+2))) #result, starting layer #, cost deriv
print(net_3._evaluate([1, 8])[0])
print(net_3._evaluate([1, 3])[0])
print("cost: ", costs[len(costs)-1])
plt.plot(costs, 'r')
plt.show()