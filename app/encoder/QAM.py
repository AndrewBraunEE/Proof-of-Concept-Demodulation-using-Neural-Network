import numpy as np

j = np.complex(0, 1)

class QAM_Modulator:

	def __init__(self, frequency_center, t_samples, t_baud, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.t_array = np.repeat(np.linspace(0, 2*np.pi, self.t_samples), 10)
		self.s1 = np.cos(2*np.pi*self.frequency_center * self.t_array) + j * np.sin(2*np.pi*self.frequency_center * self.t_array)
		self.s2 = np.sin(2*np.pi*self.frequency_center * self.t_array) - j * np.cos(2*np.pi*self.frequency_center * self.t_array)
		self.tb = t_baud
	
	def binary_pulse(self, data):
		pulse = []
		cnt = 0
		for i in range(0, self.tb*(len(data)), self.tb):
			pulse_chunk = np.ones(self.tb) * int(data[cnt])
			for element in pulse_chunk:
				pulse.append(element)
			cnt = cnt + 1
		return pulse

	def modulate(self, data):
		while len(self.t_array) < len(data) * self.tb: #Extend carrier signal to match message size
			t_array_new = np.repeat(self.t_array, 2)
			self.t_array = t_array_new
		self.s1 = np.cos(2*np.pi*self.frequency_center * self.t_array) + j * np.sin(2*np.pi*self.frequency_center * self.t_array)
		self.s2 = np.sin(2*np.pi*self.frequency_center * self.t_array) - j * np.cos(2*np.pi*self.frequency_center * self.t_array)
		#Reindex the carrier based off of the resized vector to compensate for long message size
		demux1 = []
		demux2 = []
		#Demux: Separate the binary stream into the orthogonal carriers. We multiply our samples with our orthogonal basis set and add these together
		#To form our waveform (which by definition is in quadrature)
		returned_waveform = []
		switch = True
		cnt = 0
		for element in data:
			if element != '1' and element != '0':
				continue #Multiplex only for binary stream; not for any accidental spaces.
			if switch == True:
				if element == '1':
					demux1.append(1)
				elif element == '0':
					demux1.append(0)
				switch = False
			elif switch == False:
				if element == '1':
					demux2.append(1)
				elif element == '0':
					demux2.append(0)
				switch = True
				cnt = cnt + 1
		if len(demux1) > len(demux2): #Zero Pad the Waveforms when adding, if an odd bit stream
			demux2.append(0)
		elif len(demux1) > len(demux2):
			demux1.append(0)
		cnt = 0
		for i in range(0, self.tb*(len(demux1)), self.tb): #Modulate the binary data for self.tb samples.
			signal_chunk_i = demux1[cnt] * self.s1[i:(i + self.tb)] * self.amplitude
			signal_chunk_q = demux2[cnt] * self.s2[i:(i + self.tb)] * self.amplitude
			signal_chunk = signal_chunk_i + signal_chunk_q
			#Add both signals from both bases
			for element in signal_chunk:
				returned_waveform.append(element)
			cnt = cnt + 1
		return returned_waveform * self.amplitude

	def demodulate(self, waveform):
		normalized_waveform = []
		index = 0
		for element in waveform:
			normalized_waveform.append(element / self.amplitude)
		#Normalize our waveforms in order to normalize our decision regions.
		data = []
		data_str = ''
		for i in range(0, self.tb*(len(normalized_waveform)), self.tb):
			signal_chunk = normalized_waveform[i:(i + self.tb)] #Select and correlate samples corresponding to a Binary 0 or 1
			if len(signal_chunk) == 0:
				break;
			signal_sum = 0
			basis_sum_i = 0
			basis_sum_q = 0
			for k in range(len(signal_chunk)):
				signal_sum = signal_sum + signal_chunk[k] #Sum up the points
				basis_sum_i = basis_sum_i + signal_chunk[k] * np.conjugate(self.s1[i + k]) #Sum up the chunk in the basis
				basis_sum_q = basis_sum_q + signal_chunk[k] * np.conjugate(self.s2[i + k]) 
				#We multiply by the conjugate as a method to remove the s2 carrier from our sampled binary 0 or 1 (Division by a Complex Number = Multiplication by Conjugate)
			signal_sum = signal_sum
			x_length = self.t_array.reshape(-1) #Construct a 1d view
			x_length = x_length[self.tb] - x_length[self.tb] 
			basis_sum_i = basis_sum_i / (x_length)
			basis_sum_q = basis_sum_q / (x_length) #Averaging and correlating if a 1 or 0
			if np.abs(np.real(basis_sum_i)) >= 0.5: #Bit1 is a 1
				data_str += '1'
			else:
				data_str += '0'
			if np.abs(np.real(basis_sum_q)) >= 0.5: #Bit2 is a 1
				data_str += '1'
			else:
				data_str += '0'
		#print(data_str)
		return data_str