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
import tensorflow as tf

#For good results, make the baud rate at least 2 times the frequency

class LDPC:
	#For more information on how this works, see: https://nbviewer.jupyter.org/github/hichamjanati/pyldpc-tutos/blob/master/pyLDPC-Tutorial-Basics.ipynb?flush_cache=true
	def __init__(self, **kwargs):
		self.current_tg_matrix = None
		self.current_h_matrix = None
		self.k = kwargs.get('k', None)
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
		return self.current_h_matrix, self.current_tg_matrix

	def binstr_to_messagesize(self, bin_input):
		bin_input = wrap(bin_input, k)
		while len(bin_input[-1]) < k:
			bin_input[-1] += '0' #Zeropad last element
		return bin_input

	def encode(self, bin_input, tg_matrix = None):
		#If string input: Form numpy [1, length] binary vector and return matrix-multiplied message.
		#If bitarray: Do the same
		#if tg_matrix.any == None:
		tg_matrix = self.current_tg_matrix
		
		n,k = tg_matrix.shape
		EncodedArray = None

		if isinstance(bin_input, str):
			bin_input = bin_input.replace(' ', '')
			array = np.zeros((1, len(bin_input)))
			index = 0
			for element in bin_input:
				array[(0, index)] = int(element, 2)
				index += 1
			array = array.transpose()
			x_size, y_size = array.shape
			'''if y_size != k:
				print('[n,k]: ' + str(tg_matrix.shape) + ' [array_len]: '+ str(array.shape))
				raise ValueError(" Size of message v must be equal to number of Coding Matrix G's rows " )  '''
			EncodedArray = BinaryProduct(tg_matrix, array)
		elif isinstance(bin_input, bytes):
			array = np.zeros((1, len(bin_input)))
			index = 0
			for element in bin_input:
				array[(0, index)] = int(element, 2)
				index += 1
			array = array.transpose()
			x_size, y_size = array.shape
			'''if y_size != k:
				print('[n,k]: ' + str(tg_matrix.shape) + ' [array_len]: '+ str(array.shape)) 
				raise ValueError(" Size of message v must be equal to number of Coding Matrix G's rows " ) '''
			EncodedArray = BinaryProduct(tg_matrix, array)
		return_str = ''
		EncodedArray
		for element in EncodedArray:
			return_str += str(element)
		return_str = return_str.replace('[', '').replace(']', '').replace('.','')
		return return_str

	def decode(self, input, tg_matrix = None):
		#if tg_matrix == None:
		tg_matrix = self.current_tg_matrix

		#assert(tg_matrix != None)
		#print(input)
		DecodedArray = None
		if isinstance(input, str):
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
	def encode(self, bin_input):
		##MAPPING OF BIN_INPUT (4 bits) to 7-BIT
		#See page 14 @ https://courses.cs.washington.edu/courses/cse466/12au/calendar/09a-ErrorControlCodes.pdf?fbclid=IwAR0fVzwwdDOHNhzv0BJzjHGMhui3w9RMK-YeI6V6QFK2toiGQmDKpREuof0
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
			'1000': '1000010',
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
		#print(input)
		if isinstance(input, str):
			input = wrap(input, 7)

		#while len(input[-1]) < 7: #Get the last element in the list; This shouldn't be necessary and if it's run; that means something is *likely* wrong
		#	input[-1] = input[-1] + '0' #Zeropad if necessary
		
		ham_map_inverse = {
			'0000000': '0000',
			'0001011': '0001',
			'0010111': '0010',
			'0011100': '0011',
			'0100110': '0100',
			'0101101': '0101',
			'0110001': '0110',
			'0111010': '0111',
			'1000010': '1000',
			'1001110': '1001',
			'1010010': '1010',
			'1011001': '1011',
			'1100011': '1100',
			'1101000': '1101',
			'1110100': '1110',
			'1111111': '1111'
		}
		output_str = ''
		for element in input:
			output_str += ham_map_inverse[element]

		return output_str

class RandomCodes:
	def __init__(self, input_size = 0, output_size = 0):
		self.generate_new_block(input_size, output_size)

	def generate_frequency_table(self, input_list):
		# Creating an empty dictionary  
		freq = {} 
		for item in input_list: 
			if (item in freq): 
				freq[item] += 1
			else: 
				freq[item] = 1
		return freq

	def replace_collisions(self):
		return 0
		'''
		collisions_freq = self.generate_frequency_table(self.block)
		collision_list = []
		for key, value in collisions_freq:
			if value >= 2:
				collision_list.append(key) #Generate a list of colliding keys
		myinteger_str = ''
		for i in range(self.output_size):
			myinteger_str = int_to_binary_str(np.random.randint(0, 2**output_size - 1))
			while len(myinteger_str) < output_size:
				myinteger_str += '0' #Zeropad
			list_values.append(myinteger_str)
		'''

	def generate_new_block(self, input_size, output_size):
		self.input_size = input_size
		self.output_size = output_size
		num_combinations = 2^input_size
		#print(num_combinations)
		list_keys = list(itertools.product([0, 1], repeat = input_size))
		#print(list_keys)
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
			list_values.append(myinteger_str) ##FIXME: What happens when I get the same random number twice?
		self.block = dict(zip(list_keys, list_values))
		#print(self.block)
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
			index = self.values.index(element)
			output_str += self.keys[index]

		return output_str

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

	def numpy_array_to_tfdata(self, input): #Input: Numpy array in the format [(timeaxis), (waveformsamples)]^T with timeaxis on first row, waveform samples on second row with N samples for N columns
		data_tf = tf.convert_to_tensor(input, np.float64)
		saver = tf.train.Saver(input)
		sess = tf.InteractiveSession()  
		saver.save(session, 'my-checkpoints', global_step = step)
	
										  
def string_to_save_tests():
	encoder = Encoder()
	encoder.re_init(default_mod = 'BPSK_Modulator')
	raw_data_string = encoder.load_str_dir()
	binary_str = str_to_binary_str(raw_data_string).replace(' ', '')
	data = encoder.encode_from_str(raw_data_string) # Waveform samples
	time_axis = encoder.get_modulator_default().t_array[0:len(data)]
	binary_pulse = encoder.get_modulator_default().binary_pulse(binary_str)
	#print(binary_pulse)
	print('[LENS]: Time: ' + str(len(time_axis)) + ' Data: ' + str(len(data)) + ' Pulse: ' + str(len(binary_pulse)) + ' String: ' + str(len(binary_str)))
	#matrix = np.row_stack((time_axis, data))
	matrix = numpy.array([time_axis, data, binary_pulse])
	encoder.save_to_file(matrix, 'waveform_samples.txt')
	print(matrix)
	#print(data)
	#print(encoder.load_from_file('waveform_samples.txt'))

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

def hamming_verification():
	HammingEncoder = Hamming()
	string_msg = 'hey_im_hamming'
	binstr = str_to_binary_str(string_msg).replace(' ', '')
	encoded_msg = HammingEncoder.encode(binstr)
	decoded_msg = HammingEncoder.decode(encoded_msg)
	decoded_str = binary_str_to_str(decoded_msg)
	print('[UNENCODED]: ' + binstr +  ' [ENCODED MSG]: ' + encoded_msg + ' [DECODED MSG]: ' + decoded_msg)
	print('[ENCODED MSG]: ' + string_msg + ' [DECODED MSG]: ' + decoded_str)

def ldpc_verification(): 
	Encoder = LDPC()
	string_msg = 'hey_im_LDPC'
	binstr = str_to_binary_str(string_msg).replace(' ', '')
	print(len(binstr))
	original_len = len(binstr)
	d_v = 8
	d_c = 16
	rate = d_c
	codeword_len = len(binstr)
	print(codeword_len)
	H = Encoder.construct_h_matrix(codeword_len, d_v, d_c)#7 chosen because char = 7 bits
	tg = Encoder.construct_tg_matrix(H)
	encoded_msg = Encoder.encode(binstr)
	print(len(encoded_msg))
	decoded_msg = Encoder.decode(encoded_msg, tg_matrix = tg)
	decoded_str = binary_str_to_str(decoded_msg)
	print('[UNENCODED]: ' + binstr +  ' [ENCODED MSG]: ' + encoded_msg + ' [DECODED MSG]: ' + decoded_msg)
	print('[ENCODED MSG]: ' + string_msg + ' [DECODED MSG]: ' + decoded_str)

def ldpc_verification_new():
	Encoder = LDPC()
	string_msg = 'hey_im_LDPC'
	binstr = str_to_binary_str(string_msg).replace(' ', '')
	##LDPC Parameters
	invrate = 3
	k = len(binstr)
	h, tg = Encoder.construct_use_invrate(invrate, k)
	encoded_msg = Encoder.encode(binstr)
	#print(h, tg)
	print(str(len(encoded_msg)) +'    ' + str(len(binstr)))
	#print(len(encoded_msg))
	decoded_msg = Encoder.decode(encoded_msg, tg_matrix = tg)
	decoded_str = binary_str_to_str(decoded_msg)
	print('[UNENCODED]: ' + str(binstr) +  ' [ENCODED MSG]: ' + str(encoded_msg) + ' [DECODED MSG]: ' + str(decoded_msg))
	print('[ENCODED MSG]: ' + str(string_msg) + ' [DECODED MSG]: ' + str(decoded_str))

def random_codes_verification():
	string_msg = 'hey_im_random'
	binstr = str_to_binary_str(string_msg).replace(' ', '')
	RandomEncoder = RandomCodes(2, 3)
	encoded_msg = RandomEncoder.encode(binstr)
	decoded_msg = RandomEncoder.decode(encoded_msg)
	decoded_str = binary_str_to_str(decoded_msg)
	print('[UNENCODED]: ' + str(binstr) +  ' [ENCODED MSG]: ' + str(encoded_msg) + ' [DECODED MSG]: ' + str(decoded_msg))
	print('[ENCODED MSG]: ' + str(string_msg) + ' [DECODED MSG]: ' + str(decoded_str))

if __name__ == '__main__':
	try:
		random_codes_verification()
		#string_to_save_tests()
		#load_to_tf_tests()
	except KeyboardInterrupt:
		print("KeyboardInterrupt")
		exit()