import numpy as np
from scipy.integrate import simps
#To prevent aliasing:
# Bandwidth = (1/2)*(SampleFrequencY)

j = np.complex(0, 1)

class QPSK_Modulator:

	def __init__(self, frequency_center, t_samples, t_baud, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.t_array = np.repeat(np.linspace(0, 2*np.pi, self.t_samples), 10)
		#print('T_Array:' + str(len(self.t_array)))
		self.s1 = np.cos(2* np.pi * self.frequency_center * self.t_array) + 1*j * np.sin(2* np.pi * self.frequency_center * self.t_array)
		self.s2 = np.cos(2*np.pi*self.frequency_center*self.t_array + np.pi/2) + 1*j*np.sin(2*np.pi*self.frequency_center*self.t_array + np.pi/2)
		self.tb = t_baud
		#print(self.s1)

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
		#Demux
		#We multiplex our input binary stream into separate bitstreams: one for each orthonormal basis (we have two orthonormal bases here)
		#We then modulate these streams by phase shift (multiplying by -1 or 1 for a rotation of 180 degrees or 0 rotation) and modulate using carrier signals in quadrature
		#The sum is our modulated QPSK.
		while len(self.t_array) < len(data) * self.tb: #Extend carrier signal to match message size
			t_array_new = np.repeat(self.t_array, 2)
			self.t_array = t_array_new
			#print('[DATA]: ' + str(len(data)) + ' t_array: ' + str(len(self.t_array)))
		self.s1 = np.cos(2* np.pi * self.frequency_center * self.t_array) + 1*j * np.sin(2* np.pi * self.frequency_center * self.t_array)
		self.s2 = np.cos(2*np.pi*self.frequency_center*self.t_array + np.pi/2) + 1*j*np.sin(2*np.pi*self.frequency_center*self.t_array + np.pi/2)
		demux1 = []
		demux2 = []
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
					demux1.append(-1)
				switch = False
			elif switch == False:
				if element == '1':
					demux2.append(1)
				elif element == '0':
					demux2.append(-1)
				switch = True
				cnt = cnt + 1
		cnt = 0
		if len(demux1) > len(demux2): #Zeropad the matrix to make addition possible.
			demux2.append(0)
		elif len(demux2) > len(demux1):
			demux1.append(0)
		for i in range(0, self.tb*(len(demux1)), self.tb):
			signal_chunk_i = demux1[cnt] * self.s1[i:(i + self.tb)] * self.amplitude
			signal_chunk_q = demux2[cnt] * self.s2[i:(i + self.tb)] * self.amplitude
			signal_chunk = signal_chunk_i + signal_chunk_q
			for element in signal_chunk:
				returned_waveform.append(element)
			cnt = cnt + 1
		#print('Waveforms: ' + str(returned_waveform))
		#print('len W1/W2/input: ' + str(len(demux1)) + ' ' + str(len(demux2)) + ' ' + str(len(data.replace(' ', ''))))
		#Add both signals from both bases
		#print('len out:' + str(len(Out_Waveform)))
		#print(returned_waveform[0:8*self.tb])
		return returned_waveform


	def demodulate(self, waveform):
		normalized_waveform = []
		for element in waveform:
			normalized_waveform.append(element / self.amplitude)
		data = []
		data_str = ''
		for i in range(0, self.tb*(len(normalized_waveform)), self.tb):
			signal_chunk = normalized_waveform[i:(i + self.tb)]
			if len(signal_chunk) == 0:
				break;
			signal_sum = 0
			basis_sum_i = 0
			basis_sum_q = 0
			for k in range(len(signal_chunk)):
				signal_sum = signal_sum + signal_chunk[k] 
			for k in range(len(signal_chunk)):
				basis_sum_i = basis_sum_i + (signal_chunk[k]) * np.conjugate(self.s1[i + k]) #
				basis_sum_q = basis_sum_i + (signal_chunk[k]) * np.conjugate(self.s2[i + k])
			signal_sum = signal_sum
			basis_sum_i = basis_sum_i
			basis_sum_q = basis_sum_q
			'''THIS IS WRONG SOMEHOW'''
			if np.real(basis_sum_i) >= 0: #Bit1 is a 1
				data_str += '1'
			else:
				data_str += '0'
			if np.imag(basis_sum_q) >= 0: #Bit2 is a 1
				data_str += '1'
			else:
				data_str += '0'
			
			#phase = np.arctan(np.imag(signal_sum * basis_sum_i)/np.real(signal_sum * basis_sum_q)) * (180 / np.pi)
			#print(phase)
		#print('QPSK Integrator Bits:' + str(bits))
		#print('data_str: ' + str(data_str))
		return data_str