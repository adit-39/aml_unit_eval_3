import network
import numpy as np


def  train_model(filename):
	arr_inputs = get_inputs(filename)
	arr_outputs = arr_inputs
	training_data = zip(arr_inputs,arr_outputs)
	net.SGD(training_data, 10, 100, 5.0)
	print "Model trained for : "+filename

def get_inputs(filename):
	inputs =[]
	new_inputs = []
	with open(filename) as f:
		for line in f.readlines():
			l = line.strip("\n").split(",")
			inputs.append(tuple([float(l[i]) for i in range(len(l))]))


	inp = []
	for i in inputs:
		new_inputs.append(np.asarray((i,)))

	for i in new_inputs:
		term = np.transpose(i)
		inp.append(term)
	print "Generated inputs for : "+filename
	return np.asarray(inp)


def generate_vqfiles(filename):
	arr_inputs = get_inputs(filename)
	output = []
	binary_vq = []
	for x in arr_inputs:
		output.append(net2.feedforward(x))
	for val in output:
		num = []
		for v in val:
			if v >=0.5:
				num.append(1)
			else:
				num.append(0)
		binary_vq.append(num)
	#writing to a vq file
	with open("vq"+filename[4:-8]+"vq.txt","w") as g:
		for v in binary_vq:
			sum=0
			for val in range(len(v)):
				sum+=2**(4-val) * v[val]
			output.append(sum)
			g.write(str(sum)+"\n")
	print "Generated vq_file for : "+filename+"\n"
	return output



net = network.Network([13,5,13])
train_model("mfcc/autoencoder_training_mfccs.txt")

print "_________________________________________________________________\n\n"


net2 = network.Network([13,5])
net2.weights[0] = net.weights[0]
net2.biases[0] = net.biases[0]


#Generating all the necessary vq files
for i in range(1,11):
	filename = "mfcc/c"+str(i)+"_test_mfcc.txt"
	with open(filename, "r") as k:
		vq_output = generate_vqfiles(filename)
		#print vq_output

filename = "mfcc/male1_trg_mfcc.txt"
with open(filename, "r") as k:
	vq_output = generate_vqfiles(filename)

filename = "mfcc/male2_test_mfcc.txt"
with open(filename, "r") as k:
	vq_output = generate_vqfiles(filename)

filename = "mfcc/female1_trg_mfcc.txt"
with open(filename, "r") as k:
	vq_output = generate_vqfiles(filename)

filename = "mfcc/female2_test_mfcc.txt"
with open(filename, "r") as k:
	vq_output = generate_vqfiles(filename)

filename = "mfcc/multi_1_trg_mfcc.txt"
with open(filename, "r") as k:
	vq_output = generate_vqfiles(filename)

filename = "mfcc/multi_2_trg_mfcc.txt"
with open(filename, "r") as k:
	vq_output = generate_vqfiles(filename)

filename = "mfcc/silent_1_trg_mfcc.txt"
with open(filename, "r") as k:
	vq_output = generate_vqfiles(filename)

filename = "mfcc/single_1_trg_mfcc.txt"
with open(filename, "r") as k:
	vq_output = generate_vqfiles(filename)

filename = "mfcc/single_2_trg_mfcc.txt"
with open(filename, "r") as k:
	vq_output = generate_vqfiles(filename)