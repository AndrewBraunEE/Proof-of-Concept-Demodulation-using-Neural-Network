import numpy as np
class QPSK_Modulator:

	def __init__(self, frequency_center, t_samples, sample_freq, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.sample_freq = sample_freq
		self.t_array = np.linspace(0, 2*np.pi*frequency_center*self.t_samples, self.t_samples)
		#print('T_Array:' + str(len(self.t_array)))
		self.s1 = np.cos(2* np.pi * self.frequency_center * self.t_array)
		self.s2 = np.cos(2*np.pi*self.frequency_center*self.t_array + np.pi/2)
		#print(self.s1)
	def modulate(self, data):
		#Demux
		waveform = []
		waveform2 = []
		Out_Waveform = []
		switch = True
		cnt = 0
		for element in data:
			if element != '1' or element != '0':
				cnt = cnt + 1 #Multiplex only for binary stream; not for any accidental spaces.
				pass
			if switch == True:
				if element == '1':
					waveform.append(1*self.s1[cnt])
				elif element == '0':
					waveform.append(-1*self.s1[cnt])
				switch = False
			elif switch == False:
				if element == '1':
					waveform2.append(1*self.s2[cnt])
				elif element == '0':
					waveform2.append(-1*self.s2[cnt])
				switch = True
			cnt = cnt + 1
		print('Waveforms: ' + str(waveform) + '\n' + str(waveform2))
		if len(waveform) > len(waveform2): #Zero Pad the Waveforms when adding if an odd bit stream/
			#waveform2.append(0)
			pass
		elif len(waveform2) > len(waveform):
			#waveform.append(0)
			pass
		cnt = 0
		for element in waveform:
			Out_Waveform.append(waveform[cnt] + waveform2[cnt])
			cnt = cnt + 1
		#Add both signals from both bases
		print('len W1/W2:' + str(len(waveform)) + str(len(waveform2)))
		print('len out:' + str(len(Out_Waveform)))
		return Out_Waveform * self.amplitude


	def demodulate(self, waveform):
		normalized_waveform = []
		for element in waveform:
			normalized_waveform.append(element / self.amplitude)
		data = []
		#bits = np.conv(normalized_waveform, ones(1, len(waveform)))
		for i in range(0, len(normalized_waveform)):
			bit1 = normalized_waveform[i]*self.s1[i]
			bit2 = normalized_waveform[i]*self.s2[i]
			if bit1 >= 0:
				bit1 = 1
			else:
				bit1 = 0
			if bit2 >= 0:
				bit2 = 1
			else:
				bit2 = 0
			data.append(str(bit1) + str(bit2))
		data_str = ''.join(data)
		#print(data_str)
		return data_str