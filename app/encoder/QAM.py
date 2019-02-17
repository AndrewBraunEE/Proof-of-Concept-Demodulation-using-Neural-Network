import numpy as np

class QAM_Modulator:

	def __init__(self, frequency_center, t_samples, sample_freq, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.sample_freq = sample_freq
		self.t_array = np.arange(0, self.sample_freq/2, self.sample_freq/self.t_samples)
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
			if(data[i]=='00'):
				modulated_waveform.append(self.s00)
			elif(data[i]=='01'):
				modulated_waveform.append(self.s01)
			elif(data[i]=='11'):
				modulated_waveform.append(self.s11)	
			elif(data[i]=='10'):
				modulated_waveform.append(self.s10)
		return modulated_waveform * self.amplitude

	def demodulate(self, waveform):
		normalized_waveform = []
		for element in waveform:
			normalized_waveform = element / self.amplitude 
		data = []
		for i in range(0, len(normalized_waveform)):
			bit1 = normalized_waveform[i]*np.cos(2* np.pi * self.frequency_center * self.t_array + np.pi)
			bit2 = normalized_waveform[i]*np.sin(2 * np.pi * self.frequency_center * self.t_array)
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