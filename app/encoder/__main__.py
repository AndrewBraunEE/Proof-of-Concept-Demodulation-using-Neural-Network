import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import textwrap
import pyldpc
from pyldpc.ldpcalgebra import*
__all__ = ['BinaryProduct', 'InCode', 'BinaryRank','Coding_random','Coding']
import itertools

from encoder.BPSK import *
from encoder.QPSK import *
from encoder.QAM import *
from encoder.conversions import *

#For good results, make the baud rate at least 2 times the frequency

class LDPC:
	#For more information on how this works, see: https://nbviewer.jupyter.org/github/hichamjanati/pyldpc-tutos/blob/master/pyLDPC-Tutorial-Basics.ipynb?flush_cache=true
	def __init__(self, **kwargs):
		self.current_tg_matrix = None
		self.current_h_matrix = None
		self.k = kwargs.get('k', 7)
		self.invrate = kwargs.get('invrate', None)
		if self.invrate != None:
			self.construct_use_invrate(self.invrate, self.k)
##Deprecated. Use construct_use_invrate to generate h and g matrices for better and more consistent results.
	def construct_h_matrix(self, col_num, ones_per_col, ones_per_row):
		#assert any(guess % ones_per_row == 0) == True
		assert ones_per_col < ones_per_row
		self.current_h_matrix = pyldpc.RegularH(col_num, ones_per_col, ones_per_row)
		return self.current_h_matrix

	def construct_tg_matrix(self, h_matrix = None):
		if h_matrix.any() != None:
			tG = pyldpc.CodingMatrix(h_matrix)
			#G's shape is actually the tuple (message's length, codewod's length)
			n,k = tG.shape
			self.current_tg_matrix = tG
			return tG
		elif self.current_tg_matrix != None:
			return self.current_tg_matrix 
##End of deprecated functions
	def construct_use_invrate(self, invrate, k, systematic=True):
		self.k = k
		self.current_h_matrix, self.current_tg_matrix = pyldpc.HtG(invrate, k, systematic)
		self.n, self.k = self.current_tg_matrix.shape
		return self.current_h_matrix, self.current_tg_matrix

	def binstr_to_messagesize(self, bin_input):
		bin_input = wrap(bin_input, self.k)
		while len(bin_input[-1]) < self.k:
			bin_input[-1] += '0' #Zeropad last element
		return bin_input

	def binstr_to_messagesize_inverse(self, bin_input):
		bin_input = wrap(bin_input, self.n)
		return bin_input

	def encode(self, bin_input, tg_matrix = None):
		#If string input: Form numpy [1, length] binary vector and return matrix-multiplied message.
		#If bitarray: Do the same
		#if tg_matrix.any == None:
		tg_matrix = self.current_tg_matrix
		#print(bin_input)
		n,k = tg_matrix.shape
		#print("n, k" + str(n) +"  " + str(k))
		EncodedArray = None
		return_str = ''
		if isinstance(bin_input, list):
			#raw_str = ''.join(bin_input)
			for element in bin_input: #bin_input is an array of smaller frames that need to be encoded
				index = 0
				array = np.zeros((1, len(element)))
				for char in element:
					array[(0, index)] = int(char, 2)
					index += 1
				array = array.transpose()
				EncodedArray = BinaryProduct(tg_matrix, array)
				#print("EncodedArray: " + str(EncodedArray))
				return_str += str(EncodedArray).replace('[', '').replace(']', '').replace('.','').replace(' ', '').replace('\n', '')
			#print('return_str_encode:' + return_str)
			return return_str
		if isinstance(bin_input, str): #bin_input is a single frame that needs to be encoded
			bin_input = bin_input.replace(' ', '')
			array = np.zeros((1, len(bin_input)))
			index = 0
			for element in bin_input:
				array[(0, index)] = int(element, 2)
				index += 1
			array = array.transpose()
			#x_size, y_size = array.shape
			EncodedArray = BinaryProduct(tg_matrix, array)
		elif isinstance(bin_input, bytes): #todo: check if this works
			array = np.zeros((1, len(bin_input)))
			index = 0
			for element in bin_input:
				array[(0, index)] = int(element, 2)
				index += 1
			array = array.transpose()
			#x_size, y_size = array.shape
			EncodedArray = BinaryProduct(tg_matrix, array)
		for element in EncodedArray:
			return_str += str(element)
		return_str = return_str.replace('[', '').replace(']', '').replace('.','')
		return return_str

	def decode(self, input, tg_matrix = None): #Fixme, let any tg_matrix be an argument
		#Decodes the encoded input using the tg_matrix that was previously constructed.
		#pyldpc handles any error correction by calling DecodedMessage from the pyldpc library.
		tg_matrix = self.current_tg_matrix
		DecodedArray = None
		if isinstance(input, list):
			#raw_str = ''.join(bin_input)
			return_str = ''
			DecodedArray = []
			#print("input:" + str(input))
			for element in input: #bin_input is an array of smaller frames that need to be encoded
				#print("decode_input:" + str(element))
				#print('element: ' + str(element))
				array = np.zeros((1, len(element)))
				index = 0
				for char in element:
					array[(0, index)] = int(char, 2)
					index += 1
				array = array.transpose()
				#print(array)
				return_str += str(pyldpc.DecodedMessage(tg_matrix, array)).replace('[', '').replace(']', '').replace('.','').replace(' ', '').replace('\n', '')
			#print("return_str_decode:" + return_str)
			return return_str
		elif isinstance(input, str):
			input = input.replace(' ', '')
			array = np.zeros((1, len(input)))
			index = 0
			for element in input:
				array[(0, index)] = int(element, 2)
				index += 1
			array = array.transpose()
			DecodedArray =  pyldpc.DecodedMessage(tg_matrix, array)
		elif isinstance(input, bytes):
			array = np.zeros((1, len(input)))
			index = 0
			for element in input:
				array[(0, index)] = int(element, 2)
				index += 1
			array = array.transpose()
			DecodedArray = pyldpc.DecodedMessage(tg_matrix, array)
		else: #Numpy array
			array = np.zeros((1, len(input)))
			index = 0
			for element in input:
				array[(0, index)] = int(element)
				index += 1
			array = array.transpose()
			DecodedArray = pyldpc.DecodedMessage(tg_matrix, array)
		return_str = ''
		for element in DecodedArray:
			return_str += str(element)
		return return_str

class Hamming: #Implementation of Hamming 7, 4 Encoding scheme
	def __flip(self, bit):
		if bit == '1':
			bit = '0'
		elif bit == '0':
			bit = '1'
		elif bit == 1:
			bit = 0
		elif bit == 0:
			bit = 1
		return bit

	def __replace_str_index(self, text,index=0,replacement=''):
		return '%s%s%s'%(text[:index],replacement,text[index+1:])

	def encode(self, bin_input):
		##MAPPING OF BIN_INPUT (4 bits) to 7-BIT
		#See page 14 @ https://courses.cs.washington.edu/courses/cse466/12au/calendar/09a-ErrorControlCodes.pdf?fbclid=IwAR0fVzwwdDOHNhzv0BJzjHGMhui3w9RMK-YeI6V6QFK2toiGQmDKpREuof0
		#print(bin_input)
		if isinstance(bin_input, str):
			bin_input = wrap(bin_input, 4)

		while len(bin_input[-1]) < 4: #Get the last element in the list
			bin_input[-1] = bin_input[-1] + '0' #Zeropad if necessary
		
		ham_map = {
			'0000': '0000000',
			'0001': '0001011',
			'0010': '0010111',
			'0011': '0011100',
			'0100': '0100110',
			'0101': '0101101',
			'0110': '0110001',
			'0111': '0111010',
			'1000': '1000101',
			'1001': '1001110',
			'1010': '1010010',
			'1011': '1011001',
			'1100': '1100011',
			'1101': '1101000',
			'1110': '1110100',
			'1111': '1111111'
		}
		output_str = ''
		for element in bin_input:
			output_str += ham_map[element] #Append the coded output to the output string

		return output_str

	def decode(self, input):
		#Credit for this specific function goes to xnor
		#print(input)
		if isinstance(input, str):
			input = wrap(input, 7)

		corrected_elements = []

		#Check for an error
		for element in input:
			p1 = (int(element[0]) + int(element[1]) + int(element[2]))%2
			p2 = (int(element[1]) + int(element[2]) + int(element[3]))%2
			p3 = (int(element[0]) + int(element[2]) + int(element[3]))%2
			parity_bits = str(p1) + str(p2) + str(p3)
			#Calculate the parity bits [b4:b6] and compare. 
			if p1 != int(element[4]) and p2 == int(element[5]) and p3 == int(element[6]):
				corrected_bit = self.__flip(str(p1))
				element = self.__replace_str_index(element, index = 4, replacement = corrected_bit)
			elif p1 == int(element[4]) and p2 != int(element[5]) and p3 == int(element[6]):
				corrected_bit = self.__flip(str(p2))
				element = self.__replace_str_index(element, index = 5, replacement = corrected_bit)
			elif p1 != int(element[4]) and p2 != int(element[5]) and p3 == int(element[6]): #D1
				corrected_bit = self.__flip(str(element[0]))
				element = self.__replace_str_index(element, index = 0, replacement = corrected_bit)
			elif p1 == int(element[4]) and p2 == int(element[5]) and p3 != int(element[6]):
				corrected_bit = self.__flip(str(p3))
				element = self.__replace_str_index(element, index = 6, replacement = corrected_bit)
			elif int(p1) != int(element[4]) and int(p2) == int(element[5]) and int(p3) != int(element[6]):#Error here
				corrected_bit = self.__flip(str(element[1]))
				element = self.__replace_str_index(element, index = 1, replacement = corrected_bit)
			elif p1 == int(element[4]) and p2 != int(element[5]) and p3 != int(element[6]):
				corrected_bit = self.__flip(str(element[2]))
				element = self.__replace_str_index(element, index = 2, replacement = corrected_bit)
			elif p1 != int(element[4]) and p2 != int(element[5]) and p3 != int(element[6]):
				corrected_bit = self.__flip(str(element[3]))
				element = self.__replace_str_index(element, index = 3, replacement = corrected_bit)
			corrected_elements.append(element)

		ham_map_inverse = {
			'0000000': '0000',
			'0001011': '0001',
			'0010111': '0010',
			'0011100': '0011',
			'0100110': '0100',
			'0101101': '0101',
			'0110001': '0110',
			'0111010': '0111',
			'1000101': '1000',
			'1001110': '1001',
			'1010010': '1010',
			'1011001': '1011',
			'1100011': '1100',
			'1101000': '1101',
			'1110100': '1110',
			'1111111': '1111'
		}
		output_str = ''
		for element in corrected_elements:
			output_str += ham_map_inverse[element]

		return output_str

class RandomCodes:
	def __init__(self, input_size = 0, output_size = 0):
		self.generate_new_block(input_size, output_size)

	def hamming_dmin(self, msg): #Calculate the minimum hamming distance for the given input
		freq = len(msg) #Start at the max value. The frequency of differences is the Hamming Distance
		index_min = None
		#print(str(msg))
		for index, element in enumerate(self.values):
			#print('index: ' + str(index) + ' element: ' + str(element))
			temp_freq = 0
			cnt = 0
			for character in element: #Find how many characters are different in each codeword compared to our input. Save the minimum
				if msg[cnt] != element[cnt]:
					temp_freq += 1
					#print(temp_freq)
				cnt += 1
			if temp_freq < freq:
				freq = temp_freq
				index_min = index
		return (index_min, freq)

	def generate_frequency_table(self, input_list):
		# Creating an empty dictionary  
		freq = {} 
		for item in input_list: 
			if (item in freq): 
				freq[item] += 1
			else: 
				freq[item] = 1
		return freq

	def generate_new_block(self, input_size, output_size):
		self.input_size = input_size
		self.output_size = output_size
		num_combinations = 2^input_size
		list_keys = list(itertools.product([0, 1], repeat = input_size))
		for index, element in enumerate(list_keys):
			string_to_use = ''
			for number in element:
				string_to_use += str(number)
			list_keys[index] = string_to_use

		list_values = []
		for index, element in enumerate(list_keys):
			myinteger_str = int_to_binary_str(np.random.randint(0, 2**output_size - 1))
			while len(myinteger_str) < output_size:
					myinteger_str += '0' #Zeropad
					#print(myinteger_str)
			while myinteger_str in list_values: #Keep repeating randomization if the value is already in the list
				myinteger_str = int_to_binary_str(np.random.randint(0, 2**output_size - 1))
				#print(myinteger_str)
				while len(myinteger_str) < output_size:
					myinteger_str += '0' #Zeropad
					#print(myinteger_str)
			list_values.append(myinteger_str)
		self.block = dict(zip(list_keys, list_values))
		self.keys = list_keys
		self.values = list_values

	def encode(self, bin_input):
		if isinstance(bin_input, str):
			bin_input = wrap(bin_input, self.input_size)

		while len(bin_input[-1]) < self.input_size: #Get the last element in the list
			bin_input[-1] = bin_input[-1] + '0' #Zeropad if necessary
		
		output_str = ''
		for element in bin_input:
			output_str += self.block[element] #Append the coded output to the output string

		return output_str

	def decode(self, input):
		if isinstance(input, str):
			input = wrap(input, self.output_size)

		while len(input[-1]) < self.output_size: #Get the last element in the list; This shouldn't be necessary and if it's run; that means something is *likely* wrong
			input[-1] = input[-1] + '0' #Zeropad if necessary
		


		output_str = ''
		for element in input:
			index, hamming_distance = self.hamming_dmin(element) #Find the most probable message
			#print('index, h_distance: ' + str(index) + '  ' + str(hamming_distance))
			#index = self.values.index(element)
			output_str += self.keys[index]

		return output_str

class Encoder:
	def __init__(self, **kwargs):
		self.re_init(**kwargs)

	def re_init(self, **kwargs):
		#try:
		self.freq = kwargs.get('freq', 20)
		self.encoding = kwargs.get('encoding', 'all')
		self.baud = kwargs.get('baud', 10*self.freq)
		self.samples = kwargs.get('samples', 10*self.baud) # The amount of samples to use over the period of the carrier wave. (2pi)
		self.default_mod = kwargs.get('default_mod', 'BPSK_Modulator')
		self.BPSK_Mod = BPSK_Modulator(self.freq, self.samples, self.baud, 1)
		self.QPSK_Mod = BPSK_Modulator(self.freq, self.samples, self.baud, 1)
		self.QAM_Mod =  QAM_Modulator(self.freq, self.samples, self.baud, 1)
		self.invrate = kwargs.get('invrate', 3)
		self.HammingEncoder = Hamming() #7, 4 Encoder
		self.RandomEncoder = RandomCodes(2,4)
		self.LDPC_msg_size = kwargs.get('LDPC_msg_size', 7)
		self.LDPCEncoder = LDPC(invrate = self.invrate, k = self.LDPC_msg_size)

		#except:
		#print("[ENCODER] Error in initialization of Modulators: Baudrate exceeds time-axis limit.")

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
		if self.encoding == 'Hamming':
			input = self.HammingEncoder.encode(input)
		elif self.encoding == 'Random':
			input = self.RandomEncoder.encode(input)
		elif self.encoding == 'LDPC':
			input = self.LDPCEncoder.binstr_to_messagesize(input)
			input = self.LDPCEncoder.encode(input)
		elif self.encoding == 'all':
			input = self.HammingEncoder.encode(self.LDPCEncoder.encode(self.LDPCEncoder.binstr_to_messagesize((self.RandomEncoder.encode(input)))))
		self.last_input = input
		return self.get_raw_waveform_default(input)

	def decode(self, waveform):
		return self.decode_default(self.get_modulator_default().demodulate(waveform))

	def decode_default(self, input): #input is a binary string
		if self.encoding == 'Hamming':
			input = self.HammingEncoder.decode(input)
		elif self.encoding == 'Random':
			input = self.RandomEncoder.decode(input)
		elif self.encoding == 'LDPC':
			#print(input)
			input = self.LDPCEncoder.binstr_to_messagesize_inverse(input)
			#print(input)
			input = self.LDPCEncoder.decode(input)
		elif self.encoding == 'all':
			input = self.RandomEncoder.decode(self.LDPCEncoder.decode(self.LDPCEncoder.binstr_to_messagesiz0_inverse(self.HammingEncoder.decode(input))))
		return input

	def encode_from_str(self, input):
		return self.encode_default(str_to_binary_str(input))

	def encode_from_binary_str(self, input):
		return self.encode_default(input) ##FIX ME

	def save_to_file(self, input, filename='training_waveform.txt'): #Save an encoded message in a waveform to a file
		with open(filename, "wb") as file: #Serialized
			pickle.dump(input, file)
			#np.save(file, input, allow_pickle = True)
			#np.savetxt(filename, input, delimiter=",")

	def load_from_file(self, filename='training_waveform.txt'): #Load an already encoded message in a waveform
		with open (filename, 'rb') as file: #Unserialize
			return pickle.load(file)
			#return np.load(file)
			#return np.loadtxt(file)

	def load_str_dir(self, input_dir = 'data/*.txt'):
		list_txt_files = glob.glob(input_dir)
		data = ''
		for file in list_txt_files:
			with open(file, 'r') as file_resource:
				data+=file_resource.read().replace('\n', '')
		return data

	def numpy_array_to_tfdata(self, input): #Input: Numpy array in the format [(timeaxis), (waveformsamples)]^T with timeaxis on first row, waveform samples on second row with N samples for N columns
		data_tf = tf.convert_to_tensor(input, np.float32)
		saver = tf.train.Saver(input)
		sess = tf.InteractiveSession()  
		saver.save(session, 'my-checkpoints', global_step = step)

	def decoded_binary_pulse(self, input):
		return np.fromiter(input, dtype=np.float32, count=len(input))
	
										  
def save_for_training_input(time_axis, data, binary_pulse, encoder, file_dir = None):
	matrix = []
	matrix.append(time_axis)
	matrix.append(data)
	matrix.append(binary_pulse)
	encoder.save_to_file(matrix, 'waveform_samples.txt')

def save_to_file(file_dir = None, encoder_object = None):
	if encoder_object == None:
		encoder = Encoder()
		encoder.re_init(default_mod = 'BPSK_Modulator', encoding = 'LDPC')
	else:
		encoder = encoder_object
	if file_dir != None:
		raw_data_string = encoder.load_str_dir(file_dir)
	else:
		raw_data_string = encoder.load_str_dir()
	binary_str = str_to_binary_str(raw_data_string).replace(' ', '')
	data = encoder.encode_from_str(binary_str) # Waveform samples
	time_axis = encoder.get_modulator_default().t_array[0:len(data)]
	binary_pulse = encoder.get_modulator_default().binary_pulse(binary_str)
	print('[LENS]: Time: ' + str(len(time_axis)) + ' Data: ' + str(len(data)) + ' Pulse: ' + str(len(binary_pulse)) + ' String: ' + str(len(binary_str)))
	#matrix = np.row_stack((time_axis, data))
	matrix = numpy.array([time_axis, data, binary_pulse])
	encoder.save_to_file(matrix, 'waveform_samples.txt')
	print('DONE')

def load_to_tf_tests():
	encoder = Encoder()
	encoder.re_init(default_mod = 'BPSK_Modulator')
	matrix = encoder.load_from_file('waveform_samples.txt')
	encoder.numpy_array_to_tfdata(matrix)
	#encoder.save_to_file(dataset)

def encoding_tests():
	#For codeword length N = 16 and code rate r = 0.5, we have k = 8.
	#N * r = k
	return 0

if __name__ == '__main__':
	try:
		save_to_file()
	except KeyboardInterrupt:
		print("KeyboardInterrupt")
		exit()