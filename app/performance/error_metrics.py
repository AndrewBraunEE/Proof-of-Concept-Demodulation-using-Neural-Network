class ErrorMetrics:
	def __init__(self):

	def BER(data, output_data):
		wrong_bits = 0
		for index, element in enumerate(data):
			if element != output_data[index]
				wrong_bits = wrong_bits + 1
		ber = wrong_bits / len(data)
		return ber

	def NVE(data, output_data): #This is the normalized validation error
		ber = BER(data, output_data)
		
