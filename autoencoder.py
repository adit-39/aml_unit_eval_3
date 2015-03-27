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
#arr_outputs = np.asarray([0,0,0,0,0,0,0,1])
#print type(arr_outputs)
training_data = zip(arr_inputs,arr_outputs)


print type(training_data) #list
print type(training_data[0]) #tuple
print type(training_data[0][0]) #ndarray
print type(training_data[0][0][0])	#ndarray
print type(training_data[0][0][0][0]) #float32

print
print type(training_data) #list
print type(training_data[1]) #tuple
print type(training_data[1][0]) #ndarray
print type(training_data[1][0][0])	#ndarray
print type(training_data[1][0][0][0]) #float32


print
print

print len(training_data)
print len(training_data[0])
print training_data[0][0].shape
print training_data[0][0][0].shape
print training_data[0][0][0][0]


print net.weights
print "_______________________________________________________________________________\n\n"

net.SGD(training_data, 30, 3, 3.0)

print net.weights
print "________________________________________________________________________________\n\n"

net.feedforward(training_data[0])

# net2 = network.Network([8,3])
# net2.weights = net.weights[0]
# net2.biases = net.biases

# test = np.asarray([0.,0.,0.,0.,0.,0.,0.,1.])
# test = test.reshape((8,1))

# net2.weights = np.transpose(net2.weights)
# print test.shape
# print net2.weights.shape
# print
# print
# for row in net2.weights:
# 	a = net2.feedforward([0.,0.,0.,0.,0.,0.,0.,1.])
# a = net2.feedforward([0.,0.,0.,0.,0.,0.,0.,1.])
# print a