from pipes import *
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

def save_to_csv(input_filename = 'waveform_samples.txt', output_filename = 'csv/foo.csv'):
	file = open(input_filename,'rb')
	data = pickle.load(file)
	file.close()
	a = np.asarray([ data[0], data[1], data[2] ])
	#print('numpy_array:' + str(a))
	np.savetxt(output_filename,a, delimiter = ',', fmt = "%s")

#The OFDM Module they will provide will have 32 carriers in a single channel.
#Meaning that 

def run_file_to_sink(ascii_message, sink = "/dev/stdout"):
	#Assume OFDMModule is a sink
	encoded_msg = Encoder.encode(ascii_message)
	waveform_samples = encoded_msg >> Encoder.modulate(encoded_msg)

def run_source_to_sink(source = "/dev/stdin", sink = "/dev/stdout"):
	ascii_message = cat(source)
	encoded_msg = Encoder.encode(ascii_message)
	waveform_samples = encoded_msg >> Encoder.modulate(encoded_msg) #TODO: Make this a sink


def run_source_to_file(source = "/dev/stdin", sink = "/dev/stdout"):
	ascii_message = cat(source)
	encoded_msg = Encoder.encode(ascii_message)
	waveform_samples = Encoder.modulate(encoded_msg) #TODO: Make this a sink
	Encoder.save_to_file()


if __name__ == 'main':
	try:
		run_pipes(args)
	except KeyboardInterrupt:
		exit()
