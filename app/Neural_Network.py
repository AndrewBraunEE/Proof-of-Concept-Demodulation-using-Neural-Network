import tensorflow as tf
import numpy as np


class NND:
    def __init__(training_epochs, n_h1, n_h2, n_h3, lr, n_f, n_c):        
        self.training_epochs =  training_epochs #number of iterations
        self.n_neurons_in_h1 = n_h1
        self.n_neurons_in_h2 = n_h2
        self.n_neurons_in_h3 = n_h3
        self.learning_rate = lr
        self.n_features = n_f
        self.n_classes = n_c

        
def Hidden_Layers(self):
    #inputs, will be arguments 
    X = tf.placeholder(tf.float32, [None, self.n_features], name='features')
    Y = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')

    #Weights and bias for hidden layer 1xzd232wee
    W1 = tf.Variable(tf.truncated_normal([self.n_features,self.n_neurons_in_h1],mean=0,sttdev=1/np.sqrt(self.n_features)), name='weights1')
    B1 = tf.Variable(tf.truncated_normal([self.n_neurons_in_h1],mean=0,sttdev=1/np.sqrt(self.n_features)), name='biases1')
    #activation layer for H1, used as input for activation layer 2
    sig1 = tf.nn.sigmoid((tf.matmul(X,W1)+B1),name ='activationLayer1')

    #weights and bias for hidden layer 2
    W2 = tf.Variable(tf.truncated_normal([self.n_features,self.n_neurons_in_h2],mean=0,sttdev=1/np.sqrt(self.n_features)), name='weights2')
    B2 = tf.Variable(tf.truncated_normal([self.n_neurons_in_h2],mean=0,sttdev=1/np.sqrt(self.n_features)), name='biases2')
    #activation layer for H2, used as input for activation layer 3
    sig2 = tf.nn.sigmoid((tf.matmul(sig1,W2)+B2),name ='activationLayer2')

    #weights and bias for hidden layer 3
    W3 = tf.Variable(tf.truncated_normal([self.n_features,self.n_neurons_in_h3],mean=0,sttdev=1/np.sqrt(self.n_features)), name='weights3')
    B3 = tf.Variable(tf.truncated_normal([self.n_neurons_in_h3],mean=0,sttdev=1/np.sqrt(self.n_features)), name='biases3')
    #activation layer for H3, output of NN
    sig3 = tf.nn.sigmoid((tf.matmul(sig2,W3)+B3),name ='activationLayer3')
    
    return sig3

def Hidden_layers_output(self):
    #output layer 
    sig3 = Hidden_Layers(self)
    Wo = tf.Variable(tf.random_normal([self.n_neurons_in_h3, self.n_classes], mean=0,sttdev=1/np.sqrt(self.n_features)), name='weightsOut')
    Bo = tf.Variable(tf.randmom_normal([self.n_classes], mean=0, stddev=1/np.sqrt(self.n_features)),name='biasesOut')
    a = tf.nn.softmax((tf.matmul(sig3,Wo)+bo), name='activationOutputLayer')
    return a

