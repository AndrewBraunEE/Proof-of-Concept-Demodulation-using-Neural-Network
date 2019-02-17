import numpy
class BPSK_Modulator:
	def __init__(self, frequency_center, t_samples, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.t_array = np.arange(0, 2, self.t_samples)
		self.s0 = np.cos(2* np.pi * self.frequency_center * t_array + np.pi) #According to wikipedisa for s0
		self.s1 = np.cos(2*np.pi*self.frequency_center*t_array) 				#According to wikipedia for s1

	def modulate(data):
		#According to wikipedia for s1
		modulated_waveform = []
		for i in xrange(0, len(data, 1)):
			if(data[i]=='00'):
				modulated_waveform.append(s1)
			elif(data[i]=='01'):
				modulated_waveform.append(s2)
		return modulated_waveform * self.amplitude

	def demodulate(waveform):
		normalized_waveform = waveform / self.amplitude 
		data = []
		for i in xrange(0, len(data), 1):
			phase = self.s0*normalized_waveform + self.s1*normalized_waveform
			if self.s0 - phase > self.s1 - phase
				data.append('00')
			else
				data.append('01')
		return data