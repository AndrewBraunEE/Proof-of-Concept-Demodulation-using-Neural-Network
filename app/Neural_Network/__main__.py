
import tensorflow as tf
import numpy as np
from numpy import array
import pickle 
import matplotlib.pyplot as plt

import os

N = 16
r = .5


#CREDIT TO CSM146 HW 2 CODE
class Data :
    
    def __init__(self, X=None, Y=None) :
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.Y = Y
    
    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        dir = os.path.abspath('')
        f = os.path.join(dir,filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")
        
        # separate features and labels
        self.X = data[0]
        self.Y = data[1]
        #self.Z = data[3]
    
    def plot(self, **kwargs) :
        """Plot data."""
        
        if 'color' not in kwargs :
            kwargs['color'] = 'b'
        
        plt.scatter(self.X, self.Y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, Y, **kwargs) :
    data = Data(X, Y)
    data.plot(**kwargs)

epochs_completed = 0
index_in_epoch = 0

def next_batch(num, X_train, Y_train):
    '''
    #global X_train,y_train
    global index_in_epoch, epochs_completed

    start = index_in_epoch
    index_in_epoch +=batch_size

    if index_in_epoch > num_examples:
        epochs_completed +=1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return X_train[start:end], y_train[start:end]
    '''
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(X_train))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [X_train[ i] for i in idx]
    labels_shuffle = [Y_train[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    

class NND:
    def __init__(self,training_epochs, n_h1, n_h2, n_h3, lr):      
        self.training_epochs =  training_epochs #number of iterations
        self.n_neurons_in_h1 = n_h1
        self.n_neurons_in_h2 = n_h2
        self.n_neurons_in_h3 = n_h3
        self.learning_rate = lr
        self.train_data = load_data('foo3.csv')
        self.X_train = self.train_data.X
        self.Y_train = self.train_data.Y
        print(self.Y_train)
        print(self.X_train)
        print(len(self.X_train))
        self.n_features = len(self.X_train)
        self.n_classes = len(self.X_train) #2**(N*r)#should be 2^(N*r)
        #self.n_features = n_f
        #self.n_classes = n_c
        self.X = tf.placeholder(tf.float32, [None, self.n_features], name='training')
        self.Y = tf.placeholder(tf.float32, [None, self.n_classes], name='test')
        
    def Hidden_Layers(self):
        #inputs, will be arguments 


        #Weights and bias for hidden layer 1xzd232wee
        #W1 = tf.Variable(tf.truncated_normal([self.n_neurons_in_h1, self.n_features],mean=0,stddev=1/np.sqrt(self.n_neurons_in_h1)), name='weights1')
        #B1 = tf.Variable(tf.truncated_normal([self.n_features],mean=0,stddev=1/np.sqrt(self.n_features)), name='biases1')
        W1 = tf.Variable(tf.random_normal([self.n_features, self.n_neurons_in_h1], stddev=1), name = 'W1')
        B1 = tf.Variable(tf.random_normal([self.n_neurons_in_h1]),name ='B1')
        #activation layer for H1, used as input for activation layer 2
        sig1 = tf.nn.sigmoid((tf.matmul(self.X,W1)+B1),name ='activationLayer1')

        #weights and bias for hidden layer 2
        W2 = tf.Variable(tf.random_normal([self.n_neurons_in_h1, self.n_neurons_in_h2], stddev=1), name = 'W1')
        B2 = tf.Variable(tf.random_normal([self.n_neurons_in_h2]),name ='B2')
        #activation layer for H2, used as input for activation layer 3
        sig2 = tf.nn.sigmoid((tf.matmul(sig1,W2)+B2),name ='activationLayer2')

        #weights and bias for hidden layer 3
        W3 = tf.Variable(tf.random_normal([self.n_neurons_in_h2, self.n_neurons_in_h3], stddev=1), name = 'W1')
        B3 = tf.Variable(tf.random_normal([self.n_neurons_in_h3]),name ='B3')
        #activation layer for H3, output of NN
        sig3 = tf.nn.sigmoid((tf.matmul(sig2,W3)+B3),name ='activationLayer3')

        #Wo = tf.Variable(tf.random_normal([self.n_ner]))

        #Wo = tf.Variable(tf.random_normal([self.n_features, self.n_classes], mean=0,stddev=1/np.sqrt(self.n_neurons_in_h3)), name='weightsOut')
        #Bo = tf.Variable(tf.random_normal([self.n_classes], mean=0, stddev=1/np.sqrt(self.n_features)),name='biasesOut')
        #a = tf.nn.softmax((tf.matmul(sig3,W3)+B3), name='activationOutputLayer')
   #     return sig3

  #  def accuracy_prediction(self):
        output = sig3
        learning_rate = self.learning_rate
        out_clipped = tf.clip_by_value(output,1e-10,0.9999999)#to avoid log(0) error
        #we will be using the cross entropy cost function of the form y*log(y)+(1+y)*log(1-y) to measure performance
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.Y * tf.log(out_clipped) + (1-self.Y)*tf.log(1-out_clipped), axis=1))
        #print('CO: ',cross_entropy)
        #Gradient Descent Optimizer 
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
        init_op = tf.global_variables_initializer()
        correct_prediction = tf.equal(tf.argmax(self.Y,1), tf.argmax(output,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #Using to test the neural network
        #The dimenstions of this is 784 pixels- picture data
        #Need to train on more relevent data for the project 
        '''
            list_txt_files = glob.glob('waveform_samples.txt')
            data = ''
            for file in list_txt_files:
                with open(file, 'r') as file_resource:
                    data+=file_resource.read().replace('\n', '')

        from data import input_data
        mnist = input_data.read_data_sets()                                
        '''        
        #train_data = load_data('Wave_train')
        #test_data = load_data('Wave_test')
    

        #train the model
        with tf.Session() as sess:
            sess.run(init_op)
            total_batch = int(len(self.X_train) / 32)
            #print(len(self.X_train))
            num_examples = self.X_train.shape[0]
            print(num_examples)
            #print(len(self.X_train))
            for self.training_epochs in range(self.training_epochs):
                    avg_cost = 0
                    #print(total_batch)
                    for i in range(total_batch):
                            X_, y_ = next_batch(num_examples, self.X_train, self.Y_train)
                            #print(X_)
                            #print(y_)
                            #X_ = np.transpose(X_)
                            #y_ = np.transpose(y_)
                            #y_train = np.reshape(y_train.shape[0],1)
                            #y_train = np.concatenate((1-y_train,y_train),axis =1)
                            #X_train_temp = array(X_train).reshape(3)
                            _,c = sess.run([optimiser, cross_entropy], feed_dict={self.X: X_, self.Y: y_}) #ERROR HERE
                            #print(c)
                            avg_cost += c / total_batch
                            print("AVG COST: ", avg_cost)
                    print("Epoch:",(self.training_epochs+1),"cost =", "{:.3f}".format(avg_cost))
            #print(sess.run(accuracy, feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels}))  
    
if __name__ == '__main__':
    try:
        s = NND(10, 128, 64, 68640, 0.01)
        s.Hidden_Layers()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit()

