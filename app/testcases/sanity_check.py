
#For good results, make the baud rate at least 2 times the frequency

TestString = "Hello World! Hopefully this demodulates for everything"
b_str = str_to_binary_str(TestString)
#b_str = HammingEncode(b_str)
print('b_str:' + b_str.replace(' ', ''))
TestFC = 60
TBaud = 120 #How many samples each transmitted bit takes
TestSamples = TBaud * 100

BPSK_mod = BPSK_Modulator(TestFC, TestSamples, TBaud, 1) #30 Hz, 100 Samples, 1 Volt
QPSK_mod = QPSK_Modulator(TestFC, TestSamples, TBaud, 1) # Number of Samples and Sampling Frequency should be the same. WOOPS 
QAM_mod = QAM_Modulator(TestFC, TestSamples, TBaud, 1) 

bpsk_waveform = BPSK_mod.modulate(b_str)
qpsk_waveform = QPSK_mod.modulate(b_str)
qam_waveform = QAM_mod.modulate(b_str)


plt.subplot(3, 1, 1)
plt.plot(QPSK_mod.t_array[0:len(qpsk_waveform)], qpsk_waveform)
#plt.plot(QPSK_mod.t_array, QPSK_mod.s1)
plt.title('QPSK')
plt.ylabel('Amplitude')
plt.xlabel('Time')

plt.subplot(3, 1, 2)
plt.plot(BPSK_mod.t_array[0:len(bpsk_waveform)], bpsk_waveform)
#plt.plot(BPSK_mod.t_array, BPSK_mod.s1)
plt.title('BPSK')
plt.ylabel('Amplitude')
plt.xlabel('Time')

plt.subplot(3, 1, 3)
plt.plot(QAM_mod.t_array[0:len(qam_waveform)], qam_waveform)
#plt.plot(QAM_mod.t_array, QAM_mod.s1)
plt.title('QAM')
plt.ylabel('Amplitude')
plt.xlabel('Time')

plt.show()


#print(bpsk_waveform)

#print(qpsk_waveform)

#print(qam_waveform)

bpsk_bin = BPSK_mod.demodulate(bpsk_waveform)
qpsk_bin = QPSK_mod.demodulate(qpsk_waveform)
qam_bin = QAM_mod.demodulate(qam_waveform)

'''
#print(bpsk_bin)
#print(qam_bin)

'''
#print(bpsk_bin)
print('qpsk_bin:' + str(qpsk_bin))
print('qam_bin: ' + str(qam_bin))
bpsk_str = binary_str_to_str(bpsk_bin)

qpsk_str = binary_str_to_str(qpsk_bin)

#qam_str = binary_str_to_str(None)

#print('BPSK:' + bpsk_str)
#print('QPSK:' + qpsk_str)
#print('QAM:' + qam_str)

assert qam_str == TestString
assert qpsk_str == TestString
assert bpsk_str == TestString
