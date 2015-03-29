import network
import numpy as np

net = network.Network([8,3,8])
inputs=[(0.,0.,0.,0.,0.,0.,0.,1.),(0.,0.,0.,0.,0.,0.,1.,0.),(0.,0.,0.,0.,0.,1.,0.,0.),(0.,0.,0.,0.,1.,0.,0.,0.),(0.,0.,0.,1.,0.,0.,0.,0.),(0.,0.,1.,0.,0.,0.,0.,0.),(0.,1.,0.,0.,0.,0.,0.,0.),(1.,0.,0.,0.,0.,0.,0.,0.)]

new_inputs = []
inp = []
for i in inputs:
	new_inputs.append(np.asarray((i,)))

for i in new_inputs:
	term = np.transpose(i)
	inp.append(term)

arr_inputs = np.asarray(inp)
arr_outputs = np.asarray(inp)
training_data = zip(arr_inputs,arr_outputs)

print "__________________________________________________________________\n\n"

net.SGD(training_data, 1000, 1, 1.0)

op = []
net2 = network.Network([8,3])
net2.weights[0] = net.weights[0]
net2.biases[0] = net.biases[0]
for test,y in training_data:
	op.append(net2.feedforward(test))

print "__________________________________________________________________\n\n"

for v in op:
	new = []
	for val in v:
		if val >=0.5:
			new.append(1)
		else:
			new.append(0)
	print new

print "\n__________________________________________________________________\n\n"