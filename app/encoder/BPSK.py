import numpy as np
import numpy.matlib
class BPSK_Modulator:
	def __init__(self, frequency_center, t_samples, t_baud, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.t_array = np.repeat(np.linspace(0, 2*np.pi, self.t_samples), 10)
		#print('T_Array:' + str(len(self.t_array)))
		self.s1 = np.cos(2*np.pi*self.frequency_center*self.t_array) 				#According to wikipedia for s1
		self.tb = t_baud

	def binary_pulse(self, data):
		pulse = []
		cnt = 0
		print(len(data))
		for i in range(0, self.tb*(len(data)), self.tb):
			pulse_chunk = np.ones(self.tb) * int(data[cnt])
			for element in pulse_chunk:
				pulse.append(element)
			cnt = cnt + 1
		return pulse

	def modulate(self, data):
		#According to wikipedia for s1
		data = data.replace(' ', '')
		modulated_waveform = []
		N = len(data)
		print(str(N))
		oversample_factor = int(self.t_samples/self.frequency_center)
		while len(self.t_array) < len(data) * self.tb: #Extend carrier signal to match message size
			t_array_new = np.repeat(self.t_array, 2)
			self.t_array = t_array_new
			#print('[DATA]: ' + str(len(data)) + ' t_array: ' + str(len(self.t_array)))
		self.s1 = np.cos(2*np.pi*self.frequency_center*self.t_array) #Reindex the carrier based off of the resized vector to compensate for long message size

		for i in range(0, len(data)): #For each element in the bitstream, translate into -1 or 1 to modulate phase
			if(data[i]=='0'):
				modulated_waveform.append(-1)
			elif(data[i]=='1'):
				modulated_waveform.append(1)
		#print(modulated_waveform)
		#print('mod_waveform:' + str(len(modulated_waveform)))
		#print('Oversample:' + str(oversample_factor))
		#Return bitstream at Tb (t_samples) baud with rect pulse
		#print('AI Length:' + str(len(ai)))
		returned_waveform = []
		cnt = 0
		#print('max:' + str(self.tb*(len(data) - 1)) + 'data_len: ' + str(len(data)) + 'tb: ' + str(self.tb) + 'mod_waveform_len: ' + str(len(modulated_waveform)))
		for i in range(0, self.tb*(len(modulated_waveform)), self.tb):
			signal_chunk = modulated_waveform[cnt] * self.s1[i:(i + self.tb)] * self.amplitude #Generate waveform by multiplying chunk of carrier with phase shift from binary pulse )-1 or 1)
			#print(signal_chunk)
			for element in signal_chunk:
				returned_waveform.append(element)
			#print(returned_waveform)
			#print(cnt)
			cnt = cnt + 1
		#print('out_waveform: ' + str(len(returned_waveform)))
		return returned_waveform

	def demodulate(self, waveform):
		#Implementation of demodulator for BPSK. We first normalize the waveform for our decision regions with a known amplitude.
		#Afterwards, we use our matched filter (a moving average) to record the polarity of the sinusoid of our bit in our 'signal chunk' (the samples of our carrier that our bit lies in)
		#We then compare this as a ratio to the moving average of our carrier signal to feed into our decision region to what we originally modulated our phase by (that is, multiplying by -1 or 1)
		#This decision occurs at y = 0
		normalized_waveform = []
		for element in waveform:
			normalized_waveform.append(element / self.amplitude)
		data = []
		cnt = 0
		for i in range(0, self.tb*(len(normalized_waveform)), self.tb):
			signal_chunk = normalized_waveform[i:(i + self.tb)]
			#Moving Average:
			#print(signal_chunk)
			if len(signal_chunk) == 0:
				break;
			signal_sum = 0
			basis_sum = 0
			for k in range(len(signal_chunk)):
				signal_sum = signal_sum + signal_chunk[k] #Sum up the points and average
				basis_sum = basis_sum + self.s1[i + k] #Sum up the chunk in the basis
			signal_sum = signal_sum / len(signal_chunk)
			basis_sum = basis_sum / len(signal_chunk)
			if signal_sum / basis_sum >= 0:
				data.append('1')
			else:
				data.append('0')
			cnt = cnt + 1
		data_str = ''.join(map(str, data))
		#print('data_str_len: ' + str(len(data_str)) + 'input_len: ' + str(len(normalized_waveform)))
		#print('data_str: ' + str(data_str))
		return data_str