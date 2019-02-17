from encoder.BPSK import *
from encoder.QPSK import *
from encoder.QAM import *
from encoder.conversions import *


TestString = "Hello World! Hopefully this demodulates for everything"
b_str = str_to_binary_str(TestString)
TestFC = 30
TestSamples = 100


BPSK_mod = BPSK_Modulator(TestFC, TestSamples, 10000, 1) #30 Hz, 100 Samples, 1 Volt
QPSK_mod = QPSK_Modulator(TestFC, TestSamples, 10000, 1) 
QAM_mod = QAM_Modulator(TestFC, TestSamples, 10000, 1) 

bpsk_waveform = BPSK_mod.modulate(b_str)
qpsk_waveform = QPSK_mod.modulate(b_str)
qam_waveform = QAM_mod.modulate(b_str)

print(bpsk_waveform)

#print(qpsk_waveform)

#print(qam_waveform)

bpsk_bin = BPSK_mod.demodulate(bpsk_waveform)
qpsk_bin = QPSK_mod.demodulate(qpsk_waveform)
qam_bin = QAM_mod.demodulate(qam_waveform)

#print(qam_bin)

bpsk_str = binary_str_to_str(bpsk_bin)
qpsk_str = binary_str_to_str(qpsk_bin)
qam_str = binary_str_to_str(qam_bin)

assert qam_str == TestString
assert qpsk_str == TestString
assert bpsk_str == TestString