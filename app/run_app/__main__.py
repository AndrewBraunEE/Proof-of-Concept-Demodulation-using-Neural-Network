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
	argparser.add_argument('-v', '--verbose', action='store_true', dest = 'verbose', help='Increase output and log verbosity')
	argparser.add_argument('-t', '--train', action='store_true', dest = 'train', default = False, help = 'Train the neural network from either a piped source or a file.')
	argparser.add_argument('-n', '--num_input_bits', type = int, dest = 'num_input_bits', default = 1, help = 'The number of bits per input WORD')
	argparser.add_argument('-k', '--num_output_bits', type = int, dest = 'num_output_bits', default = 1, help = 'The number of bits per output WORD')
	argparser.add_argument('-i', '--inverse_rate', type = int, dest = 'inv_rate', default = 3, help = 'The ratio of CodedWordLength / UncodedWordLength used in the LDPC Encoder')
	argparser.add_argument('-d', '--filedir', type = str, dest = 'filedir', default = 'data/*.txt', help = 'Use textfiles (UTF-8 Encoded) from a file directory as source')
	argparser.add_argument('-o', '--source', type = str, dest = 'source', default = None, help = 'Use the parameter after -k as the source directory as a pipe')
	argparser.add_argument('-l', '--sink', type = str, dest = 'sink', default = None, help = 'Use the parameter after -l as the sink directory as a pipe')
	argparser.add_argument('-m', '--modulator', type = str, dest = 'modulator', default = 'BPSK_Modulator', help = 'Specify the modulation type for the encoded data')
	argparser.add_argument('-s', '--save', action = 'store_true', dest = 'save', default = True, help = 'Store the waveform samples to a file')
	argparser.add_argument('-u', '--snr', type = int, dest = 'snr', default = 15, help = 'Transmit the bitsequence over a noisy channel with the specified SNR')
	argparser.add_argument('-c', '--channel', type = str, dest = 'channel', default = 'noise', help = 'Specify whether the noisy channel is only noisy (PARAM: noise) or is noisy and fading (PARAM: fade)')
	argparser.add_argument('-e', '--encoder', type = str, dest = 'encoder', default = 'LDPC', help = 'Specify which encoding scheme to use, or to use all three in serial. By default, uses random codes, ldpc codes, and hamming code in series.')
	argparser.add_argument('-p', '--plot', type = str, dest = 'plot', default = 'plot_channel', help = 'Specify which encoding scheme to use, or to use all three in serial. By default, uses random codes, ldpc codes, and hamming code in series.')
	
	args = argparser.parse_args()

	try:
		app_encoder = Encoder(default_mod = args.modulator, encoding = args.encoder, invrate = args.inv_rate)
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
				binary_str = str_to_binary_str(app_encoder.load_str_dir(args.filedir)).replace(' ', '')
				binary_str_unencoded = binary_str
				waveform_samples = app_encoder.encode_default(binary_str)
				binary_str = app_encoder.last_input #The encoded message
			ChannelObject = Channel(SNRdB = args.snr, signal = waveform_samples)
			if args.channel == 'noise':
				noisy_channel = ChannelObject.w
				for index, value in enumerate(waveform_samples):
					waveform_samples[index] += noisy_channel[(0, index)]
			else:
				noisy_channel = ChannelObject.create_fading_channel(input_signal = waveform_samples)
				waveform_samples = noisy_channel

			if args.save:
				sys.stderr.write(" \n Saving...")
				save_for_training_input(app_encoder.get_modulator_default().t_array[0:len(waveform_samples)], waveform_samples, app_encoder.decoded_binary_pulse(binary_str_unencoded), app_encoder, file_dir = args.filedir)
				save_to_csv('waveform_samples.txt', 'csv/foo.csv') #Change the 3.2.csv to be modular	
		bin_pulse = app_encoder.get_modulator_default().binary_pulse(binary_str)
		if args.plot == 'plot_channel':
			plt.plot(app_encoder.get_modulator_default().t_array[0:len(waveform_samples)], waveform_samples, 'b')
			plt.plot(app_encoder.get_modulator_default().t_array[0:len(bin_pulse)], bin_pulse, 'r')
			plt.show()
		if args.train:
			sys.stderr.write(" \n Training our NN \n")
			ErrorObject = ErrorMetrics(app_encoder.get_modulator_default())
			s = NND(5, 128, 64, 32, 0.001)
			training_epochs, nve_array, ber_array = s.Hidden_Layers(decoded_waveform = bin_pulse, ErrorObject = ErrorObject, batch_size = app_encoder.get_modulator_default().tb) 
			if args.plot == 'plot_error':
				plt.title("NVE and BER as a function of Epoch Indices")
				plt.plot(training_epochs, nve_array, 'b-', label = 'NVE')
				plt.plot(training_epochs, ber_array, 'r-', label = 'BER')
				plt.xlabel("Training Epochs")
				plt.ylabel("NVE")
				plt.legend(loc = 'upper_left')
				plt.show()

	except KeyboardInterrupt:
		exit()

if __name__ == '__main__':
	run()
