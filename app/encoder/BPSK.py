import numpy as np
class BPSK_Modulator:
	def __init__(self, frequency_center, t_samples, sample_freq, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.sample_freq = sample_freq
		self.t_array = np.arange(0, self.sample_freq/2, self.sample_freq/self.t_samples)
		self.s0 = np.cos(2* np.pi * self.frequency_center * self.t_array + np.pi) #According to wikipedisa for s0
		self.s1 = np.cos(2*np.pi*self.frequency_center*self.t_array) 				#According to wikipedia for s1

	def modulate(self, data):
		#According to wikipedia for s1
		modulated_waveform = []
		for i in range(0, len(data)):
			if(data[i]=='0'):
				modulated_waveform.append(self.s0[i])
			elif(data[i]=='1'):
				modulated_waveform.append(self.s1[i])
		return modulated_waveform * self.amplitude

	def demodulate(self, waveform):
		normalized_waveform = []
		for element in waveform:
			normalized_waveform = element / self.amplitude 
		data = []
		for i in range(0, len(normalized_waveform)):
			phase = self.s0*normalized_waveform + self.s1*normalized_waveform
			if self.s0 - phase > self.s1 - phase:
				data.append('0')
			else:
				data.append('1')
		return data