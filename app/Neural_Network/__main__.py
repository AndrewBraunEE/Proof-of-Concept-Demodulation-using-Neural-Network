import tensorflow as tf
import numpy as np
from numpy import array
import pickle 
import matplotlib.pyplot as plt
import json

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
    
    def load_from_data(self, input_filename = 'waveform_samples.txt'):
        file = open(input_filename,'rb')
        data = pickle.load(file)
        file.close()
        self.X = data[1]
        self.Y = data[2]

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
        self.X = data[1]
        self.Y = data[2]
        
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
    #data = Data()
    #data.load(filename)
    data = Data()
    data.load_from_data()
    return data

def plot_data(X, Y, **kwargs) :
    data = Data(X, Y)
    data.plot(**kwargs)

epochs_completed = 0
index_in_epoch = 0

'''
def next_batch(num_X, num_Y, X_train, Y_train):
    
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
 
    Return a total of `num` random samples and labels. 
    idx = np.arange(0 ,len(X_train))
    idy = np.arange(0, len(Y_train))
    np.random.shuffle(idx)
    np.random.shuffle(idy)
    idx = idx[:num_X]
    idy = idy[:num_Y]
    data_shuffle = X_train[idx]
    labels_shuffle = Y_train[idy]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
'''
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    idy = np.arange(0, len(labels))
    idx = idx[:num]
    idy = idy[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idy]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

'''
def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]
'''
class NND:
    def __init__(self, savefile,training_epochs, n_h1, n_h2, n_h3, lr,D=None,K=None,**kwargs):
        self.decoded_waveform = kwargs.get('decoded_waveform', None) #This should be the binary pulse, not the binary string
        self.ErrorObject = kwargs.get('ErrorObject', None) 
        self.batch_size = kwargs.get('batch_size', None)
        
        
        self.savefile = savefile
        
        self.training_epochs =  training_epochs #number of iterations
        self.n_neurons_in_h1 = n_h1
        self.n_neurons_in_h2 = n_h2
        self.n_neurons_in_h3 = n_h3
        self.learning_rate = lr
        self.train_data = load_data('csv/foo.csv')
        self.X_train = self.train_data.X
        self.Y_train = self.train_data.Y
        #print(self.Y_train)
        #print(self.X_train)
        #print(len(self.X_train))
        self.n_features = 200
        self.n_classes =  len(self.decoded_waveform) #2**(N*r)#should be 2^(N*r)
        self.D = self.n_features
        self.K = self.n_classes

        print("The length is: ", len(self.decoded_waveform))
        #self.n_features = n
        #self.n_classes = n_c
        self.X = tf.placeholder(tf.float32, [None, self.n_features], name='training')
        self.Y = tf.placeholder(tf.float32, [None, self.n_classes], name='test')

    '''
    def save(self, filename):
        j = {
          'D': self.D,
          'K': self.K,
          'model': self.savefile
        }
        with open(filename, 'w') as f:
          json.dump(j, f)
    '''

    def Hidden_Layers(self, **kwargs):
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

        Wo = tf.Variable(tf.random_normal([self.n_neurons_in_h3, self.n_classes], stddev=1), name='weightsOut')
        Bo = tf.Variable(tf.random_normal([self.n_classes]), name='biasesOut')
        output = tf.nn.sigmoid((tf.matmul(sig3,Wo)+Bo),name ='Output_Layer')

        #output = sig3
        learning_rate = self.learning_rate
        out_clipped = tf.clip_by_value(output,1e-10,0.9999999)#to avoid log(0) error
        print("Size of outclipped: ", output)
        #we will be using the cross entropy cost function of the form y*log(y)+(1-y)*log(1-y) to measure performance
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
            saver = tf.train.Saver()
            saver.restore(sess,self.savefile)
            total_batch = int(self.n_features/self.batch_size)
            print(len(self.X_train))
            print(total_batch)
            #print(len(self.X_train))
            for self.training_epochs in range(self.training_epochs):
                    avg_cost = 0
                    #print(total_batch)
                    for i in range(total_batch):
                            #X_, y_ = next_batch(self.n_features, self.n_classes, self.X_train, self.Y_train)
                            X_,y_ = next_batch(self.batch_size, self.X_train, self.Y_train)
                            X_ = np.expand_dims(X_, axis = 0)
                            y_ = np.expand_dims(y_, axis = 0)
                            _, c = sess.run([optimiser, cross_entropy], feed_dict={self.X: X_, self.Y: y_}) #ERROR HERE
                            avg_cost += c / total_batch
                            #print("AVG COST: ", avg_cost)
                            '''
                            if decoded_waveform != None and ErrorObject != None:
                                NVE_Val = ErrorObject.NVE(decoded_waveform,self.Y)
                                NVE_Array.append(NVE_Val)
                                BER_Val = ErrorObject.BER(decoded_waveform,self.Y)
                                BER_Array.append(BER_Val)
                            '''
                            #print("AVG COST: ", avg_cost)
                    print("Epoch:",(self.training_epochs+1),"cost =", "{:.3f}".format(avg_cost))
            #print(sess.run(accuracy, feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels}))  
            saver.save(sess, self.savefile)
        return ([],[],[])
    
if __name__ == '__main__':
    try:
        s = NND(100, 128, 64, 32, 0.01)
        s.Hidden_Layers(batch_size = 100)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit()
