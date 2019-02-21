import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle


from encoder.BPSK import *
from encoder.QPSK import *
from encoder.QAM import *
from encoder.conversions import *

#For good results, make the baud rate at least 2 times the frequency
'''
class NoisyChannel:
	def __init__(self, **kwargs):
		self.size = kwargs.get('size', 10000)
	
	def getNoise(self, **kwargs)
'''
class Encoder:
	def __init__(self, **kwargs):
		self.re_init(**kwargs)

	def re_init(self, **kwargs):
		try:
			self.freq = kwargs.get('freq', 60)
			self.encoding = kwargs.get('encoding', None)
			self.baud = kwargs.get('baud', 2*self.freq)
			self.samples = kwargs.get('samples', 100*self.baud) # The amount of samples to use over the period of the carrier wave. (2pi)
			self.default_mod = kwargs.get('default_mod', 'BPSK_Modulator')
			self.BPSK_Mod = BPSK_Modulator(self.freq, self.samples, self.baud, 1)
			self.QPSK_Mod = BPSK_Modulator(self.freq, self.samples, self.baud, 1)
			self.QAM_Mod =  QAM_Modulator(self.freq, self.samples, self.baud, 1)
		except:
			print("[ENCODER] Error in initialization of Modulators: Baudrate exceeds time-axis limit.")

	def get_time_axis(input):
		return self.BPSK_Mod.t_array[0:len(input)]

	def get_raw_waveform_qam(self, input):
		return self.QAM_Mod.modulate(input)

	def get_raw_waveform_bpsk(self, input):
		return self.BPSK_Mod.modulate(input)
	
	def get_raw_waveform_qpsk(self, input):
		return self.QPSK_Mod.modulate(input)

	def get_raw_waveform_default(self, input):
		function = getattr(self, 'default_mod')
		if function == 'BPSK_Modulator':
			return self.get_raw_waveform_bpsk(input)
		elif function == 'QPSK_Modulator':
			return self.get_raw_waveform_qpsk(input)
		elif function == 'QAM_Modulator':
			return self.get_raw_waveform_qam(input)
		return []

	def get_modulator_default(self):
		function = getattr(self, 'default_mod')
		if function == 'BPSK_Modulator':
			return self.BPSK_Mod
		elif function == 'QPSK_Modulator':
			return self.QPSK_Mod
		elif function == 'QAM_Modulator':
			return self.QAM_Mod

	def encode_default(self, input):
		#Operations here to encode. #FIX ME
		return self.get_raw_waveform_default(input)

	def encode_from_str(self, input):
		return self.encode_default(str_to_binary_str(input))

	def encode_from_binary_str(self, input):
		return self.encode_default(input) ##FIX ME

	def save_to_file(self, input, filename='training_waveform.txt'): #Save an encoded message in a waveform to a file
		with open(filename, "wb") as file: #Serialized
			pickle.dump(input, file)

	def load_from_file(self, filename='training_waveform.txt'): #Load an already encoded message in a waveform
		with open (filename, 'rb') as file: #Unserialize
			return pickle.load(file)

	def load_str_dir(self, input_dir = 'data/*.txt'):
		list_txt_files = glob.glob(input_dir)
		data = ''
		for file in list_txt_files:
			with open(file, 'r') as file_resource:
				data+=file_resource.read().replace('\n', '')
		return data
'''
	def numpy_array_to_tfdata(self, input): #Input: Numpy array in the format [(timeaxis), (waveformsamples)]^T with timeaxis on first row, waveform samples on second row with N samples for N columns
		features = input[0]

		assert features.shape[0] == labels.shape[0]

		features_placeholder = tf.placeholder(features.dtype, features.shape)
		labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

		dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
		# [Other transformations on `dataset`...]
		dataset = ...
		iterator = dataset.make_initializable_iterator()

	sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
                                          '''
def main():
	encoder = Encoder()
	encoder.re_init(default_mod = 'BPSK_Modulator')
	raw_data_string = encoder.load_str_dir()
	binary_str = str_to_binary_str(raw_data_string).replace(' ', '')
	data = encoder.encode_from_str(raw_data_string) # Waveform samples
	time_axis = encoder.get_modulator_default().t_array[0:len(data)]
	binary_pulse = encoder.get_modulator_default().binary_pulse(binary_str)
	#print(binary_pulse)
	print('[LENS]: Time: ' + str(len(time_axis)) + ' Data: ' + str(len(data)) + ' Pulse: ' + str(len(binary_pulse)) + ' String: ' + str(len(binary_str)))
	matrix = np.row_stack((time_axis, data, binary_pulse))
	encoder.save_to_file(matrix, 'waveform_samples.txt')
	#print(data)
	print(encoder.load_from_file('waveform_samples.txt'))

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("KeyboardInterrupt")
		exit()