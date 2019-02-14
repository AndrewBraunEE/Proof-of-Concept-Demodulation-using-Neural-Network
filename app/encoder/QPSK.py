class QPSK_Modulator:

	def __init__(self, frequency_center, t_samples, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
	def modulate(data):
		t_array = np.arange(0, 2, t_samples)
		s1 = np.cos(2* np.pi * frequency_center * t_array)
		s2 = np.cos(2*np.pi*frequency_center*t_array + np.pi/2)
		s3 = np.cos(2*np.pi*frequency_center*t_array + np.pi)
		s4 = np.cos(2*np.pi*frequency_center*t_array + (np.pi)*1.5)
		modulated_waveform = []
		for i in xrange(0, len(data, 1)):
			if(data[i]=='00'):
				modulated_waveform.append(s1)
			elif(data[i]=='01'):
				modulated_waveform.append(s2)
			elif(data[i] == '10'):
				modulated_waveform.append(s2)
			elif(data[i] == '11'):
				modulated_waveform.append(s4)
		return modulated_waveform * self.amplitude
