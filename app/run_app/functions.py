from pipes import *

#The OFDM Module they will provide will have 32 carriers in a single channel.
#Meaning that 

def run_file_to_sink(ascii_message, sink = "/dev/stdout"):
	#Assume OFDMModule is a sink
	encoded_msg = Encoder.encode(ascii_message)
	waveform_samples = encoded_msg >> Encoder.modulate(encoded_msg)

def run_source_to_sink(source = "/dev/stdin", sink = "/dev/stdout"):
	ascii_message = cat(source)
	encoded_msg = Encoder.encode(ascii_message)
	waveform_samples = encoded_msg >> Encoder.modulate(encoded_msg) #TODO: Make this a sink


def run_source_to_file(source = "/dev/stdin", sink = "/dev/stdout"):
	ascii_message = cat(source)
	encoded_msg = Encoder.encode(ascii_message)
	waveform_samples = Encoder.modulate(encoded_msg) #TODO: Make this a sink
	Encoder.save_to_file()


if __name__ == 'main':
	try:
		run_pipes(args)
	except KeyboardInterrupt:
		exit()