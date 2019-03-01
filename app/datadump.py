import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

file = open('waveform_samples_2.txt','rb')
data = pickle.load(file)
file.close()

a = np.asarray([ data[0], data[1], data[2] ])
np.savetxt("foo3.2.csv",a)
