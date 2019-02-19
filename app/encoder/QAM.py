import numpy as np

class QAM_Modulator:

	def __init__(self, frequency_center, t_samples, sample_freq, amplitude):
		self.frequency_center = frequency_center
		self.t_samples = t_samples #Sample Rate
		self.amplitude = amplitude
		self.sample_freq = sample_freq
		self.t_array = np.linspace(0, 2*np.pi*frequency_center*self.t_samples, self.t_samples)
		self.s1 = np.cos(2*np.pi*self.frequency_center * self.t_array)
		self.s2 = np.sin(2*np.pi*self.frequency_center * self.t_array)
	def modulate(self, data):
		#Demux
		#print('DATA:' + data)
		waveform = []
		waveform2 = []
		Out_Waveform = []
		switch = True
		cnt = 0
		for element in data:
			if element != '1' and element != '0':
				continue #Multiplex only for binary stream; not for any accidental spaces.
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
		#print('Waveforms: ' + str(waveform) + '\n' + str(waveform2))
		if len(waveform) > len(waveform2): #Zero Pad the Waveforms when adding if an odd bit stream/
			waveform2.append(0)
		elif len(waveform2) > len(waveform):
			waveform.append(0)
		cnt = 0
		#print('len W1/W2:' + str(len(waveform)) + str(len(waveform2)))
		for element in waveform:
			Out_Waveform.append(waveform[cnt] + waveform2[cnt])
			cnt = cnt + 1
		#Add both signals from both bases
		#print('len out:' + str(len(Out_Waveform)))
		return Out_Waveform * self.amplitude

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