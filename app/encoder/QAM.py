import numpy as np

class QAM_Modulator:

	def __init__(self, frequency_center, t_samples, sample_freq, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.sample_freq = sample_freq
		self.t_array = np.linspace(0, 2*np.pi*frequency_center*self.t_samples, self.t_samples)
		self.s00 = 0 * (self.t_array)
		self.s01 = np.sin(2*np.pi*self.frequency_center * self.t_array)
		self.s11 = np.cos(2* np.pi * self.frequency_center * self.t_array + np.pi) + np.sin(2*np.pi*self.frequency_center * self.t_array)		
		self.s10 = np.cos(2* np.pi * self.frequency_center * self.t_array + np.pi)
	def modulate(self, data):
		#print(s00)
		#print(s01)
		#print(s11)
		#print(s10)
		modulated_waveform = []
		for i in range(0, len(data)):
			if(data[i]=='0'):
				modulated_waveform.append(self.s00)
			elif(data[i]=='1'):
				modulated_waveform.append(self.s01)
		return modulated_waveform * self.amplitude

	def demodulate(self, waveform):
		normalized_waveform = []
		index = 0
		for element in waveform:
			normalized_waveform.append(element / self.amplitude)
		data = []
		#print(normalized_waveform)
		for i in range(0, len(normalized_waveform)):
			bit1 = normalized_waveform[i]*self.s01[i]
			bit2 = normalized_waveform[i]*self.s11[i]
			#print(bit1 + bit2)
			if bit1 > 0.5:
				bit1 = 1
			else:
				bit1 = 0
			if bit2 > 0.5:
				bit2 = 1
			else:
				bit1 = 0
			data.append(str(bit1) + str(bit2))
		return data