import numpy as np
import matplotlib.pyplot as plt
import glob

from encoder.BPSK import *
from encoder.QPSK import *
from encoder.QAM import *
from encoder.conversions import *

#For good results, make the baud rate at least 2 times the frequency

class Encoder:
	def __init__(self, **kwargs):
		self.re_init(kwargs)

	def re_init(self, **kwargs)
		try:
			self.freq = kwargs.get('freq', 60)
			self.encoding = kwargs.get('encoding', None)
			self.baud = kwargs.get('baud', 2*self.frequency)
			self.samples = kwargs.get('samples', 100*self.baud) # The amount of samples to use over the period of the carrier wave. (2pi)
			self.default_mod = kwargs.get('default_mod', 'BPSK_Modulator')
			self.BPSK_Mod = BPSK_Modulator(self.freq, self.samples, self.baud, 1)
			self.QPSK_Mod = BPSK_Modulator(self.freq, self.samples, self.baud, 1)
			self.QAM_Mod =  QAM_Modulator(self.freq, self.samples, self.baud, 1)
		except:
			print("[ENCODER] Error in initialization of Modulators: Baudrate exceeds time-axis limit.")

	def get_raw_waveform_qam(self, input):
		return self.QAM_Mod.modulate(input)

	def get_raw_waveform_bpsk(self, input):
		return self.BPSK_Mod.modulate(input)
	
	def get_raw_waveform_qpsk(self, input):
		return self.QPSK_Mod.modulate(input)

	def get_raw_waveform_default(self, input):
		function = getattr(self, self.default_mod)
		return function(input)

	def encode_default(self, input):
		#Operations here to encode. #FIX ME
		return self.get_raw_waveform_default(input)

	def encode_from_str(self, input):
		return self.encode_default(str_to_binary_str(input))

	def encode_from_binary_str(self, input)
		return self.encode_default(input) ##FIX ME

	def save_to_file(self, input, filename): #Save an encoded message in a waveform to a file
		return 0

	def load_from_file(self, input): #Load an already encoded message in a waveform
		return 0

	def load_str_dir(self, input_dir = 'data/*.txt'):
		list_txt_files = glob.glob(input_dir)
		data = ''
		for file in list_txt_files
			with open(file, 'r') as file_resource:
    			data=file_resource.read().replace('\n', '')
    	return data

def main():
	encoder = Encoder()
	encoder.re_init(default_mod = 'BPSK_Modulator')
	data = encoder.encode_default(encoder.load_str_dir())
	encoder.save_to_file(data, 'waveform_samples')

if __name__ == '__main__'
	try:
		main()
	except KeyboardInterrupt:
		print("KeyboardInterrupt")
		exit()