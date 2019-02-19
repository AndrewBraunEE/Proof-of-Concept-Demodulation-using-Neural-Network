from encoder.BPSK import *
from encoder.QPSK import *
from encoder.QAM import *
from encoder.conversions import *


TestString = "Hello World! Hopefully this demodulates for everything"
b_str = str_to_binary_str(TestString)
print('b_str:' + b_str)
TestFC = 10000
TestSamples = 1000000


BPSK_mod = BPSK_Modulator(TestFC, TestSamples, 10000, 1) #30 Hz, 100 Samples, 1 Volt
QPSK_mod = QPSK_Modulator(TestFC, TestSamples, 10000, 1) # Number of Samples and Sampling Frequency should be the same. WOOPS 
QAM_mod = QAM_Modulator(TestFC, TestSamples, 10000, 1) 

bpsk_waveform = BPSK_mod.modulate(b_str)
qpsk_waveform = QPSK_mod.modulate(b_str)
qam_waveform = QAM_mod.modulate(b_str)

#print(bpsk_waveform)

#print(qpsk_waveform)

#print(qam_waveform)

bpsk_bin = BPSK_mod.demodulate(bpsk_waveform)
qpsk_bin = QPSK_mod.demodulate(qpsk_waveform)
'''
qam_bin = QAM_mod.demodulate(qam_waveform)

#print(bpsk_bin)
#print(qam_bin)

'''
#print(bpsk_bin)
print('qpsk_bin:' + str(qpsk_bin))

bpsk_str = binary_str_to_str(bpsk_bin)

qpsk_str = binary_str_to_str(qpsk_bin)

#qam_str = binary_str_to_str(None)

#print('BPSK:' + bpsk_str)
#print('QPSK:' + qpsk_str)
#print('QAM:' + qam_str)

assert qam_str == TestString
assert qpsk_str == TestString
assert bpsk_str == TestString
