import sys
import numpy as np
class ErrorMetrics:
	def __init__(self, demodulator):
		self.demodulator = demodulator

	def BER(self, data, output_data):
		wrong_bits = 0
		#print('nn_output:' + str(data) + '\ndecoded_output:' + str(output_data))
		for index, element in np.ndenumerate(data):
			#print('element: ' + str(element))
			#print('element2: ' + str(output_data[index]))
			if round(element) != round(output_data[index]):
				wrong_bits = wrong_bits + 1
		ber = wrong_bits / len(data)
		return ber

	def NVE(self, data_BER_array, output_data_BER_array, waveform): #This is the normalized validation error
		NVE = 0
		#print(data_BER_array, output_data_BER_array)
		if len(data_BER_array) != len(output_data_BER_array):
			sys.stderr.write("Wrong length")
		ber = self.BER(data_BER_array, output_data_BER_array)
		observed_demod = self.demodulator.demodulate(waveform)
		ber_map = 0.0
		index = 0
		for element in observed_demod:
			element1 = round(output_data_BER_array[index]) #neural_netoutput
			if round(element1) != round(int(element)):
				ber_map += 1
			index += 1
		if ber_map != 0:
			NVE = NVE + (ber/ber_map)
		else:
			NVE = NVE / len(data_BER_array)
		return NVE