import matplotlib.pyplot as plot

from modulation_lib import *

file ='Modulations'
freq=20

baudrate = 9600

Fs = baudrate * 10; #Sampling Rate, always oversample by 10.
Ts = 1.0/Fs; # sampling interval

t = np.arange(0,2,Ts)

b = np.array(['11','01','00','10','11'])
samples_per_bit = 2 * Fs / b.size
ddb = np.repeat(b, samples_per_bit)

y_qpsk = QPSK_mod(ddb,freq,Ts)

.......

myplot[3,0].plot(t,y_qpsk)
myplot[3,0].set_xlabel('Time')
myplot[3,0].set_ylabel('Amplitude')

myplot[3,1].plot(fspec_ask,abs(spec_qpsk),'r')#plot spectrum
myplot[3,1].set_xlabel('Freq (Hz)')
myplot[3,1].set_ylabel('|Y(freq)|')
