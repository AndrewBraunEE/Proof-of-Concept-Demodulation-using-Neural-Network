import numpy

class QPSK_Modulator:

	def __init__(self, frequency_center, t_samples, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.t_array = np.arange(0, 2, t_samples)
		self.s1 = np.cos(2* np.pi * frequency_center * t_array)
		self.s2 = np.cos(2*np.pi*frequency_center*t_array + np.pi/2)
		self.s3 = np.cos(2*np.pi*frequency_center*t_array + np.pi)
		self.s4 = np.cos(2*np.pi*frequency_center*t_array + (np.pi)*1.5)
	def modulate(data):
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
	def demodulate(waveform):
		normalized_waveform = waveform / self.amplitude 
		out_data = []
		bits = conv(normalized_waveform, ones(1, len(waveform)))
		for i in xrange(0, len(waveform), 1):
			bit1 = normalized_waveform[i]*np.cos(2* np.pi * frequency_center * t_array + np.pi)
			bit2 = normalized_waveform[i]*np.sin(2 * np.pi * frequency_center * t_array)
			if bit1 >= 0:
				bit1 = 1
			else:
				bit1 = 0
			if bit2 >= 0:
				bit1 = 1
			else:
				bit1 = 0
			data.append(str(bit1) + str(bit2))
		return data