import numpy as np
import matplotlib.pyplot as plt
class Channel:
	def __init__(self, **kwargs):
		self.create_noise(**kwargs)

	def create_noise(self, **kwargs):
		self.signal = kwargs.get('signal', [])
		x_axis = kwargs.get('x', [])
		#print(x_axis)
		L = 0
		if self.signal != []:
			L = len(self.signal)
		else:
			L = len(x_axis)
		self.SNRdB = float(kwargs.get('SNRdB', 27))
		self.SNR = kwargs.get('SNR', float(10)**(float(self.SNRdB)/float(10)))
		self.Ps = float(kwargs.get('Ps', sum(np.power(self.signal, 2))/L)) #Average Power of Signal
		self.Pn = self.Ps/self.SNR #Wanted Noise Power
		self.w = np.random.randn(1, L) #Generate random vector 0 - mean
		self.w = self.w-sum(self.w)/L 
		self.Pw = float((float(1)/float(L)))*np.sum(self.w**float(2)) #Scale to wanted power
		constant = np.sqrt(self.Pn/self.Pw)
		self.w = self.w * constant
		self.Pw_meas = 1/L*sum((self.w)**2)#Measure resulting SNR
		self.SNRdB_meas = 10*np.log10(self.Ps/self.Pw_meas)
		self.std_dev = kwargs.get('std_dev', None)
		self.cnt = kwargs.get('phase_sample_bias', 0)

		return self.w #Our generated Noise Distribution

	def output(noise_distro = None, signal = None):
		if noise_distro == None:
			noise_distro = self.w
		if signal == None:
			signal = self.signal

		return signal + noise_distro

	def work_items(input_items): #Acts as a source / sink
		output_items = []
		for element in input_items:
			output_items.append(input_items[cnt] + self.w[cnt])
			self.cnt += 1
		return output_items

def main(): #Testing to see if noise creation is being generated correctly.
	x = np.arange(0, 2*np.pi, 1/(1000*2*np.pi))
	noisychannel = Channel(x = x, SNRdB = 10, Ps = 1)
	plt.plot(x, noisychannel.w[0,:])
	plt.show()
	#print(x)
	#print(noisychannel.w[0,:])

if __name__ == '__main__':
	try:
		print(__name__)
		main()
	except KeyboardInterrupt:
		exit()