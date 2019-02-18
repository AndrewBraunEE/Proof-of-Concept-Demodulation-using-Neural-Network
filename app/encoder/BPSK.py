import numpy as np
import numpy.matlib
class BPSK_Modulator:
	def __init__(self, frequency_center, t_samples, sample_freq, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.sample_freq = sample_freq
		self.t_array = np.linspace(0, 2*np.pi*frequency_center*self.t_samples, self.t_samples)
		#print('T_Array:' + str(len(self.t_array)))
		self.s0 = np.cos(2* np.pi * self.frequency_center * self.t_array + np.pi) #According to wikipedisa for s0
		self.s1 = np.cos(2*np.pi*self.frequency_center*self.t_array) 				#According to wikipedia for s1

	def modulate(self, data):
		#According to wikipedia for s1
		modulated_waveform = []
		N = len(data)
		oversample_factor = int(self.t_samples/self.frequency_center)
		for i in range(0, len(data)):
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
		for element in modulated_waveform:
			returned_waveform.append(element * self.s1[cnt] * self.amplitude)
			cnt = cnt + 1
		return returned_waveform

	def demodulate(self, waveform):
		normalized_waveform = []
		for element in waveform:
			normalized_waveform.append(element / self.amplitude)
		data = []
		for i in range(0, len(normalized_waveform)):
			if normalized_waveform[i] / self.s1[i] >= 0:
				data.append('1')
			else:
				data.append('0')
		data_str = ''.join(map(str, data))
		return data_str