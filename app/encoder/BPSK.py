class BPSK_Modulator:
	def __init__(self, frequency_center, t_samples, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude

	def modulate(data):
		t_array = np.arange(0, 2, t_samples)
		s0 = np.cos(2* np.pi * frequency_center * t_array + np.pi) #According to wikipedisa for s0
		s1 = np.cos(2*np.pi*frequency_center*t_array) 				#According to wikipedia for s1
		modulated_waveform = []
		for i in xrange(0, len(data, 1)):
			if(data[i]=='00'):
				modulated_waveform.append(s1)
			elif(data[i]=='01'):
				modulated_waveform.append(s2)
		return modulated_waveform * self.amplitude
