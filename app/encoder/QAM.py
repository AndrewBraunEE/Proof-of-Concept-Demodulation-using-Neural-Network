import numpy

class QAM_Modulator:

	def __init__(self, frequency_center, t_samples, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
	def modulate(data):
		t_array = np.arange(0, 2, t_samples)
		s00 = 0 * np*(t_array)
		s01 = np.sin(2*np.pi*frequency_center * t_array)
		s11 = np.cos(2* np.pi * frequency_center * t_array + np.pi) + np.sin(2*np.pi*frequency_center * t_array)		
		s10 = np.cos(2* np.pi * frequency_center * t_array + np.pi)
		modulated_waveform = []
		for i in xrange(0, len(data, 1)):
			if(data[i]=='00'):
				modulated_waveform.append(s00)
			elif(data[i]=='01'):
				modulated_waveform.append(s01)
			elif(data[i]=='11'):
				modulated_waveform.append(s11)	
			elif(data[i]=='10'):
				modulated_waveform.append(s10)
		return modulated_waveform * self.amplitude

	def demodulate(data):
		normalized_waveform = waveform / self.amplitude 
		data = []
		for i in xrange(0, len(data), 1):
			bit1 = normalized_waveform[i]*np.cos(2* np.pi * frequency_center * t_array + np.pi)
			bit2 = normalized_waveform[i]*np.sin(2 * np.pi * frequency_center * t_array)
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