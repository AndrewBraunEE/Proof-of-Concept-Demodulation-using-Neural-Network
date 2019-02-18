import struct
from . import hamming
from textwrap import wrap

def str_to_binary_str(input=None):
	return ' '.join(format(ord(x), 'b') for x in input)

def binary_str_to_str(input=None):
	string = ''
	binary = wrap(input, 8)
	for element in binary:
		string = string + chr(int(element, 2)).encode('ascii', errors='replace').decode("utf-8")
	#print(binary)
	#print(string)
	return string