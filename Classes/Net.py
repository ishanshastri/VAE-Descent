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

class CostNeuron(Neuron):
    def Evaluate(self, input, target, input_roc=[]):
        tot = 0
        for i in range(len(input)):
            tot += target[i]-input[i]
        deriv = 2*(tot)
        return [np.linalg.norm(np.subtract(input, target)), tot] 
        #Generalize (abstract it)

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
        print(curr)
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

    def _backProp_n(self, output, target, blayer = 0, prev = []):#prev is partial derivs of cost function w.r.t. last layer output
        local_partials_d_dw = output[1][0]
        local_partials_d_do = output[1][1]
        cur_grads = [local_partials_d_dw[i]*prev[i] for i in range(len(prev))]
        #next_grads = [not me]
    #def _calcGrad()

    def _getCost(self, hypo, target):
        diff = np.subtract(hypo, target)
        grad = [self.Length[1](diff[i]) for i in range(len(diff))]
        return [self.Length[0](diff), grad]

    def _susser(self, output, layer, out_d_d, grad):
        if layer<0:
            return grad
        cur_grads = []
        cur_grads = []#np.empty(len(output[1][layer])*len(output[1][layer][0][0]))
        l = output[1][layer]
        prev_outs = [0 for i in range(len(l))]
        for i in range(len(l)): #loop thru each neuron in layer 
            neuron = l[i]
            #print(neuron)
            d_dw = neuron[0] #derivs wrt weights
            d_dx = neuron[1] #derivs wrt inputs
            #d_outs = out_d_d[1]
            #print("doubts:", d_outs)
            #for j in range(len(d_dw)):
            #cur_grads = cur_grads + [out_d_d[i]*d_dw[j] for j in range(d_dw)]
            temp_outs = []
            for j in range(len(d_dw)):
                cur_grads.append(out_d_d[i]*d_dw[j])
                temp_outs.append(out_d_d[i]*d_dx[j])
            #temp_outs += [out_d_d[i]*d_dx[j] for j in range(d_dx)]
            prev_outs = np.add(prev_outs, temp_outs)
            #cur_grads = np.add(cur_grads, )
            #out_ds += 
        return self._susser(output, layer-1, prev_outs, grad + cur_grads)
        #out_ds = 

           # d_dw = l[0]
           # d_dx = l[1]



        #cur_grads = [d_dw[i]*out_d_d[i] for i in range(len(d_dw))]

        #cur_grads = []

    #Publics
    def GetLayer(self, depth):
        """
        Get layer at given depth
        """
        return self.Network[depth]
'''
#Test(s)
sigmoid = lambda x:1/(1+exp(-1*x))
d_dx_sigmoid = lambda x:sigmoid(x)*(1-sigmoid(x))
#n = Net([2, 3, 2, 3, 5], [[sigmoid, d_dx_sigmoid] for i in range(4)], [lambda a:np.linalg.norm(a), lambda a: 2*a])
n = Net([2, 1, 1, 2], [[sigmoid, d_dx_sigmoid] for i in range(2)] + [[lambda x:x, lambda x:1]], [lambda a:np.linalg.norm(a), lambda a: 2*a])

res = n._evaluate([1, 1, 1])

#print("derivs: ", res[1])
#print_net(n)
#print(n.GradientVector)
bp = BackProp()
bp.GetGrad(n, [], [], [])

for d in res[1]:
    print("L")
    for p in d:
        print(p)

def b_prop(res, layers):
    ind = layers-1
    layer = res[1][ind]
    

b_prop(res, 3)

neur = Neuron(2, lambda : 1, [sigmoid, d_dx_sigmoid])
c_neur = CostNeuron(3,  lambda : 1, [sigmoid, d_dx_sigmoid])#fix params
#print("Eval_Result: ", c_neur.Evaluate([1, 2, 2], [1, 2, 3]))
#print(neur)
#print(neur.Evaluate([1, 1, 1])[0])

#print(neur.Evaluate([1, 1, 1])[1])
'''
'''
L
([0.21646371070662088, 0.21646371070662088, 0.21646371070662088], [0.023298310088874834, 0.06017903508249897, 0.08280784850208056])
L
([0.18530551108987986, 0.12658759816004056], [0.16858080135412962, 0.05756391367055788])
L
([0.2232145669668299, 0.16838215712831228], [0.09444668972928201, 0.07591037060305607])
([0.2028502258481585, 0.15302029373089238], [0.07658835468975284, 0.1486404021046774])
'''
'''

#print(n._getCost([0, 1, 0], [1, 0, 0]))#output, target
#cost = n._getCost()
print("output:", res[0])
print("target:", [1, 1])
print("gradient: ", n._susser(res, 2, n._getCost(res[0], [1, 1])[1], []))
'''
t_net = Net([1, 1, 1, 1], [[lambda x:x, lambda x:1] for i in range(3)], [lambda a:np.linalg.norm(a), lambda a: 2*a])
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
#print("layers: ", len(r1[1]))
#print("layers?: ", len(t_net.Network))
cost = t_net._getCost(r1[0], 4) #tuple of cost and deriv wrt input
grad = t_net._susser(r1, 2, cost[1], [])

print("out: ", r1[0])
print("target: ", 4)
print("cost: ", cost[0])
print("grad: (not me)", grad)


