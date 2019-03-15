import argparse
from encoder.__main__ import Encoder, save_for_training_input
from encoder.noisy_channel import *
import sys
from run_app.functions import *
from encoder.conversions import *
import matplotlib.pyplot as plt
from Neural_Network.__main__ import *
from utility.error_metrics import ErrorMetrics

def run():
	argparser = argparse.ArgumentParser('Launch the EE132A Project')
	argparser.add_argument('-t', '--train', action='store_true', dest = 'train', default = False, help = 'Train the neural network from either a piped source or a file.')
	argparser.add_argument('-n', '--num_input_bits', type = int, dest = 'num_input_bits', default = 8, help = 'The number of bits per input WORD')
	argparser.add_argument('-i', '--inverse_rate', type = int, dest = 'inv_rate', default = 3, help = 'The ratio of CodedWordLength / UncodedWordLength used in the LDPC Encoder')
	argparser.add_argument('-d', '--filedir', type = str, dest = 'filedir', default = 'data/*.txt', help = 'Use textfiles (UTF-8 Encoded) from a file directory as source')
	argparser.add_argument('-o', '--source', type = str, dest = 'source', default = None, help = 'Use the parameter after -k as the source directory as a pipe')
	argparser.add_argument('-l', '--sink', type = str, dest = 'sink', default = None, help = 'Use the parameter after -l as the sink directory as a pipe')
	argparser.add_argument('-m', '--modulator', type = str, dest = 'modulator', default = 'BPSK_Modulator', help = 'Specify the modulation type for the encoded data')
	argparser.add_argument('-s', '--save', action = 'store_true', dest = 'save', default = True, help = 'Store the waveform samples to a file')
	argparser.add_argument('-v', '--load', action = 'store_true', dest = 'load', default = False, help = 'Load the previous session')
	argparser.add_argument('-u', '--snr', type = int, dest = 'snr', default = 40, help = 'Transmit the bitsequence over a noisy channel with the specified SNR')
	argparser.add_argument('-c', '--channel', type = str, dest = 'channel', default = 'noise', help = 'Specify whether the noisy channel is only noisy (PARAM: noise) or is noisy and fading (PARAM: fade)')
	argparser.add_argument('-e', '--encoder', type = str, dest = 'encoder', default = 'LDPC', help = 'Specify which encoding scheme to use, or to use all three in serial. By default, uses random codes, ldpc codes, and hamming code in series.')
	argparser.add_argument('-p', '--plot', type = str, dest = 'plot', default = 'plot_all', help = 'Specify what graphs to plot.')
	argparser.add_argument('-r', '--run', action = 'store_true', dest = 'run', default = False, help = 'Specify to train or to just decode the current strings in data')
	
	args = argparser.parse_args()

	try:
		app_encoder = Encoder(default_mod = args.modulator, encoding = args.encoder, invrate = args.inv_rate, LDPC_msg_size = args.num_input_bits)
		bit_stream = None
		use_trainer = False #Train neural network?
		if args.sink and not args.source:
			run_file_to_sink(bit_stream, args.source)
		elif args.sink and args.source:
			run_source_to_sink(args.sink, args.source)
		elif not args.sink and args.source:
			run_source_to_file(args.sink, args.source)
		else:
			#Run file to file
			waveform_samples = []
			if args.filedir:
				original_str = app_encoder.load_str_dir_asarray(args.filedir)
				for message in original_str:
					binary_str = str_to_binary_str(message).replace(' ', '')
					binary_str_unencoded = binary_str
					waveform_samples = app_encoder.encode_default(binary_str)
					binary_str = app_encoder.last_input #The encoded message
					#print("bin0:" + binary_str + "\nbin1: " + binary_str_unencoded)
					#sys.stderr.write("encoded:" + binary_str)
					#print(str((app_encoder.decode(waveform_samples))))
					#sys.stderr.write("unencoded: " + app_encoder.load_str_dir(args.filedir) + "\ndecoded: " + str(binary_str_to_str(app_encoder.decode(waveform_samples))))
					ChannelObject = Channel(SNRdB = args.snr, signal = waveform_samples)
					if args.channel == 'noise':
						noisy_channel = ChannelObject.w
						for index, value in enumerate(waveform_samples):
							waveform_samples[index] += noisy_channel[(0, index)]
					else:
						noisy_channel = ChannelObject.create_fading_channel(input_signal = waveform_samples)
						waveform_samples = noisy_channel #Adds N + Signal automatically in create_fading channel.
						#for index, value in enumerate(waveform_samples):
						#	waveform_samples[index] += noisy_channel[(0, index)]

					if args.save:
						sys.stderr.write(" \n Saving...")
						#print("decoded array:" + str(app_encoder.decoded_binary_pulse(binary_str_unencoded)))
						save_for_training_input(app_encoder.get_modulator_default().t_array[0:len(waveform_samples)], waveform_samples, np.array(app_encoder.decoded_binary_pulse(binary_str_unencoded)), app_encoder, file_dir = args.filedir)
						save_to_csv('waveform_samples.txt', 'csv/foo.csv') #Change the 3.2.csv to be modular	
					bin_pulse = app_encoder.get_modulator_default().binary_pulse(binary_str)
					if args.train or args.run:
						sys.stderr.write(" \n Training our NN \n")
						ErrorObject = ErrorMetrics(app_encoder.get_modulator_default())
						#original_waveform = binary_str_to_intarray(original_str)
						if args.run == False:
							s = NND("./tf.model", 50 , 128, 64, 32, 0.01, decoded_waveform = app_encoder.decoded_binary_pulse(binary_str_unencoded), ErrorObject = ErrorObject, batch_size = app_encoder.get_modulator_default().tb, waveform = waveform_samples, freq = app_encoder.freq,
								will_load = args.load, num_chars = len(message), invrate = args.inv_rate, original_bin_array = binary_str_unencoded, num_classes = args.num_input_bits)
						else:
							s = NND("./tf.model", 1 , 128, 64, 32, 0.01, decoded_waveform = app_encoder.decoded_binary_pulse(binary_str_unencoded), ErrorObject = ErrorObject, batch_size = app_encoder.get_modulator_default().tb, waveform = waveform_samples, freq = app_encoder.freq,
								will_load = args.load, num_chars = len(message), invrate = args.inv_rate, original_bin_array = binary_str_unencoded, num_classes = args.num_input_bits)
						training_epochs, nve_array, ber_array, ber_map = s.Hidden_Layers()
						#Normalize these arrays for inverse_rate * original_byte_length (8) in order to get the error rates of each epoch, and not of each epoch of each byte!
						sum_training_epochs = []
						sum_nve_array = []
						sum_ber_array = []
						sum_ber_map_array = []
						for chunk_index in range(0, len(training_epochs), args.inv_rate):
							curr_sum_training_epochs = 0
							curr_sum_nve_array = 0
							curr_ber_array = 0
							for index in range(args.inv_rate):
								curr_sum_training_epochs += training_epochs[chunk_index + index]/args.inv_rate
								curr_sum_nve_array += nve_array[chunk_index + index]/args.inv_rate
								curr_ber_array += ber_array[chunk_index + index]/args.inv_rate
							sum_training_epochs.append(curr_sum_training_epochs)
							sum_nve_array.append(curr_sum_nve_array)
							sum_ber_array.append(curr_ber_array)
							sum_ber_map_array.append(ber_map)

						#print('sum_training_epochs:' + sum_training_epochs)

						if args.plot == 'plot_error' or args.plot == 'plot_all' and args.run == False:
							plt.title("NVE and BER as a function of Epoch Indices")
							plt.plot(sum_training_epochs, sum_nve_array, 'b-', label = 'NVE')
							plt.plot(sum_training_epochs, sum_ber_array, 'r-', label = 'BER Neural')
							plt.plot(sum_training_epochs, sum_ber_map_array, 'g-', label = 'BER Map')
							plt.xlabel("Training Epochs")
							plt.ylabel("NVE / BER")
							plt.legend(loc = 'upper_left')
							plt.show()
						if args.run == True:
							print('Neural Network BER: ' + str(ber_array))
							print('Neural Network NVE: ' + str(nve_array))
							print('Neural Network BER_MAP: ' + str(ber_map))

	except KeyboardInterrupt:
		exit()

if __name__ == '__main__':
	run()
