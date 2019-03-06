from encoder.__main__ import Hamming, LDPC, RandomCodes
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import textwrap
import pyldpc
from pyldpc.ldpcalgebra import*
__all__ = ['BinaryProduct', 'InCode', 'BinaryRank','Coding_random','Coding']
import itertools

from encoder.BPSK import *
from encoder.QPSK import *
from encoder.QAM import *
from encoder.conversions import *
import tensorflow as tf


def hamming_verification():
	HammingEncoder = Hamming()
	string_msg = 'hey_im_hamming'
	binstr = str_to_binary_str(string_msg).replace(' ', '')
	encoded_msg = HammingEncoder.encode(binstr)
	decoded_msg = HammingEncoder.decode(encoded_msg)
	decoded_str = binary_str_to_str(decoded_msg)
	print('[UNENCODED]: ' + binstr +  ' [ENCODED MSG]: ' + encoded_msg + ' [DECODED MSG]: ' + decoded_msg)
	print('[ENCODED MSG]: ' + string_msg + ' [DECODED MSG]: ' + decoded_str)

def ldpc_verification(): 
	Encoder = LDPC()
	string_msg = 'hey_im_LDPC'
	binstr = str_to_binary_str(string_msg).replace(' ', '')
	print(len(binstr))
	original_len = len(binstr)
	d_v = 8
	d_c = 16
	rate = d_c
	codeword_len = len(binstr)
	print(codeword_len)
	H = Encoder.construct_h_matrix(codeword_len, d_v, d_c)#7 chosen because char = 7 bits
	tg = Encoder.construct_tg_matrix(H)
	encoded_msg = Encoder.encode(binstr)
	print(len(encoded_msg))
	decoded_msg = Encoder.decode(encoded_msg, tg_matrix = tg)
	decoded_str = binary_str_to_str(decoded_msg)
	print('[UNENCODED]: ' + binstr +  ' [ENCODED MSG]: ' + encoded_msg + ' [DECODED MSG]: ' + decoded_msg)
	print('[ENCODED MSG]: ' + string_msg + ' [DECODED MSG]: ' + decoded_str)

def ldpc_verification_new():
	Encoder = LDPC()
	string_msg = 'hey_im_LDPCTESTTTIN     WHOA'
	binstr = str_to_binary_str(string_msg).replace(' ', '')
	print("binstr: " + binstr)
	##LDPC Parameters
	invrate = 3
	#k = len(binstr)
	k = 8
	h, tg = Encoder.construct_use_invrate(invrate, k)
	binstr = Encoder.binstr_to_messagesize(binstr)
	encoded_msg = Encoder.encode(binstr)
	#print(h, tg)
	#print(str(len(encoded_msg)) +'    ' + str(len(binstr)))
	#print(len(encoded_msg))
	#print("encoded_msg: " + str(encoded_msg))
	encoded_msg = Encoder.binstr_to_messagesize_inverse(encoded_msg)
	#print("encoded_msg: " + str(encoded_msg))
	decoded_msg = Encoder.decode(encoded_msg, tg_matrix = tg)
	decoded_str = binary_str_to_str(decoded_msg)
	print('[UNENCODED]: ' + str(binstr) +  ' [ENCODED MSG]: ' + str(encoded_msg) + ' [DECODED MSG]: ' + str(decoded_msg))
	print('[ENCODED MSG]: ' + str(string_msg) + ' [DECODED MSG]: ' + str(decoded_str))

def random_codes_verification():
	string_msg = 'hey_im_random'
	binstr = str_to_binary_str(string_msg).replace(' ', '')
	RandomEncoder = RandomCodes(2, 3)
	encoded_msg = RandomEncoder.encode(binstr)
	decoded_msg = RandomEncoder.decode(encoded_msg)
	decoded_str = binary_str_to_str(decoded_msg)
	print('[UNENCODED]: ' + str(binstr) +  ' [ENCODED MSG]: ' + str(encoded_msg) + ' [DECODED MSG]: ' + str(decoded_msg))
	print('[ENCODED MSG]: ' + str(string_msg) + ' [DECODED MSG]: ' + str(decoded_str))

def main():
	#hamming_verification()
	ldpc_verification_new()
	#random_codes_verification()

if __name__ != 'main':
	try:
		print("MAIN")
		main()
	except KeyboardInterrupt:
		exit()