import sys
class ErrorMetrics:
	def __init__(self, demodulator):
		self.demodulator = demodulator

	def BER(data, output_data):
		wrong_bits = 0
		for index, element in enumerate(data):
			if element != output_data[index]:
				wrong_bits = wrong_bits + 1
		ber = wrong_bits / len(data)
		return ber

	def NVE(data_BER_array, output_data_BER_array): #This is the normalized validation error
		NVE = 0.0
		if len(data_BER_array) != len(output_data_BER_array):
			sys.stderr.write("Wrong length")
		for index, element in enumerate(data_BER_array):
			ber = BER(data, output_data)
			ber_map = self.demodulator.demodulate()
			NVE = NVE + (ber/ber_map)
		NVE = NVE / len(data_BER_array)
		return NVE