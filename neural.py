import math
import random

class InputNeuron:
    pass

class HiddenNeuron:
    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = [random.uniform(-0.1, 0.1) for _ in [None] + inputs]
        self.activity = []
        self.error = 0
        self.delta = 0
        self.weight_change = 0

    def update_activity(self):
        activities = [1] + [neuron.a for neuron in self.inputs]
        s = sum([w * a for w, a in zip(self.weights, activities)])
        self.a = 1 / (1 + math.exp(-s))
        return self.a

class OutputNeuron:
    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = [random.uniform(-0.1, 0.1) for _ in [None] + inputs]
        self.activity = []
        self.delta = 0
        self.weight_change = 0

    def update_activity(self):
        activities = [1] + [neuron.a for neuron in self.inputs]
        s = sum([w * a for w, a in zip(self.weights, activities)])
        self.a = 1 / (1 + math.exp(-s))
        return self.a 

class Network:

    def __init__(self, input, hidden, output):
        self.input = [InputNeuron() for _ in range(input)]
        self.hidden = [HiddenNeuron(self.input) for _ in range(hidden)]
        self.output = [OutputNeuron(self.hidden) for _ in range(output)]

    def run(self, inputs):
        for i in range(len(inputs)):
            self.input[i].a = inputs[i]
        for neuron in self.hidden:
#             print('updating hidden')
#             print(neuron.weights)
            neuron.activity.append(neuron.update_activity())
#
        for neuron in self.output:
#             print('updating output')
            neuron.activity.append(neuron.update_activity())
        return [neuron.a for neuron in self.output]
     
    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)
    
    def sigmoidPrimeo(self, s):
        #derivative of sigmoid
        return [a * (1 - b) for a,b in zip(s,s)]
      
        
    def print(self):
        print("hidden")
        for neuron in self.hidden:
            print(neuron.weights)
        print("output")
        for neuron in self.output:
            print(neuron.weights)
            
            
    def backward(self,X, y, output):
    
        output_error = [a - b for a, b in zip(y,output)] # error in output
        output_delta = [ a*b for a,b in zip(output_error,self.sigmoidPrimeo(output))] # applying derivative of sigmoid to error
        
        for neuron in self.hidden:
            neuron.error = [a*b for a,b in zip(output_delta, neuron.weights)]
#             print(neuron.error)
        for neuron in self.hidden: 
            neuron.delta = [a*b for a, b in zip(neuron.error, neuron.activity )]
#             print(neuron.delta)
            
        input1 = [] 
        input2 = []
        
        for pair in X:
            input1.append(pair[0])
            input2.append(pair[1])
            
#         print(input1)
#         print(input2)
            
            
#          for neuron in self.hidden: 
#             weight_change1 = [ab for a,b in zip(input1, neuron.delta)]# these are the changes
#             print(weight_change1)
#             weight_change2 = [a*b for a,b in zip(input2, neuron.delta)]
#             weight_change  = [a+b for a,b in zip(weight_change1, weight_change2)]
#             neuron.weights = [a+b for a,b in zip(weight_change1, neuron.weights)] # apply changes

        weight_change1 = [a*b for a,b in zip(input1, self.hidden[0].delta)]
        weight_change2 = [a*b for a,b in zip(input2, self.hidden[1].delta)]
        
        self.hidden[0].weights = [a+b for a,b in zip(weight_change1, self.hidden[0].weights)]
        self.hidden[1].weights = [a+b for a,b in zip(weight_change2, self.hidden[1].weights)] 
        
        for neuron in self.output:                                            
            weight_changed = [a*b for a,b in zip(neuron.activity,output_delta)] # hidden --> output changes
            neuron.weights = [a+b for a,b in zip(weight_changed, neuron.weights)]   
            
            
            
        
        
    def train (self, X, y):
        predicted = []
        for i in X:
            predicted += self.run(i)
        #print(predicted)
        self.backward(X, y, predicted)
        for neuron in self.hidden:
            del neuron.activity[:]
        for neuron in self.output:
            del neuron.activity[:]

                
        
n = Network(2, 2, 1)

for i in range(100):
    n.train([[0,0], [0,1], [1,0], [1,1]], [0, 1, 1, 0])

print(n.run([0, 0]))  # Should be [0]
print(n.run([0, 1]))  # Should be [1]
print(n.run([1, 0]))  # Should be [1]
print(n.run([1, 1]))  # Should be [0]
