import argparse
import app.__main__
import sys
argparser = argparse.ArgumentParser('Launch the EE132A Project')
argparser.add_argument('-v', '--verbose', action='store_true',
                           help='Increase output and log verbosity')
argparser.add_argument('-t', '--train' ,action='store_true', help = 'Train the neural network from either a piped source or a file.')
argparser.add_argument('-d', '--filedir', action = 'store_true', help = 'Use textfiles (UTF-8 Encoded) from a file directory as source')
argparser.add_argument('-k', '--source', action = 'store_true', help = 'Use the parameter after -k as the source directory as a pipe')
argparser.add_argument('-l', '--sink', action = 'store_true', help = 'Use the parameter after -l as the sink directory as a pipe')
argparser.add_argument('-m', '--modulator', action = 'store_true', help = 'Specify the modulation type for the encoded data')
argparser.add_argument('-s', '--save', action = 'store_true', help = 'Store the waveform samples to a file')
args = argparser.parse_args()

try:
	encoder = Encoder()
	bit_stream = None
	use_trainer = False #Train neural network?
	sys.stderr.write(args)
	print(args) #todo, delete.
	if args.m:
		encoder.re_init(default_mod = args.m)
    if args.t: #Train
    	use_trainer = args.t
	if args.d:
		bit_stream = encoder.load_str_dir(args.d)
	if args.s:
		encoder.save_to_file(args.s)
	elif args.l and not args.k:
		run_file_to_sink(bit_stream, args.k)
	elif args.l and args.k:
		run_source_to_sink(args.l, args.k)
	elif not args.l and args.k:
		run_source_to_file(args.l, args.k)

except KeyboardInterrupt:
    exit()
