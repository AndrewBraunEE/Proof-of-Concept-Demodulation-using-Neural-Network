from . import hamming

def str_to_binary_str(input):
	return ' '.join(format(ord(x), 'b') for x in input)

def binary_str_to_str(input):
	return input.decode('ascii')
