
import tensorflow as tf
import numpy as np


class NND:
    def __init__(training_epochs, n_h1, n_h2, n_h3, lr, n_f, n_c):        
        self.training_epochs =  training_epochs #number of iterations
        self.n_neurons_in_h1 = n_h1
        self.n_neurons_in_h2 = n_h2
        self.n_neurons_in_h3 = n_h3
        self.learning_rate = lr
        #self.n_features = n_f
        #self.n_classes = n_c
        self.X = tf.placeholder(tf.float32, [None, n_f], name='training')
        self.Y = tf.placeholder(tf.float32, [None, n_c], name='test')

        
def Hidden_Layers(self):
    #inputs, will be arguments 
    

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

#check if this is necessary
def Hidden_layers_output(self):
    #output layer 
    sig3 = Hidden_Layers(self)
    Wo = tf.Variable(tf.random_normal([self.n_neurons_in_h3, self.n_classes], mean=0,sttdev=1/np.sqrt(self.n_features)), name='weightsOut')
    Bo = tf.Variable(tf.randmom_normal([self.n_classes], mean=0, stddev=1/np.sqrt(self.n_features)),name='biasesOut')
    a = tf.nn.softmax((tf.matmul(sig3,Wo)+bo), name='activationOutputLayer')
    return a

def accuracy_prediction(self):
    output = Hidden_layers_output(self)
    learning_rate = self.learning_rate
    out_clipped = tf.clip_by_value(output,1e-10,0.9999999)#to avoid log(0) error
    #we will be using the cross entropy cost function of the form y*log(y)+(1+y)*log(1-y) to measure performance
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.Y * tf.log(out_clipped) + (1-self.Y)*tf.log(1-out_clipped), axis=1)
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    init_op = tf.global_variables_initializer()
    correct_prediction = tf.equal(tf.argmax(self.Y,1), tf.argmax(output,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
       
    #Using to test the neural network                                
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)                                
                                    
    #train the model
    with tf.Session() as sess:
        sess.run(init_op)
        total_batch = int(len(mnist.train.lables) / 100)
        for self.training_epochs in range(self.training_epochs):
                avg_cost = 0
                for i in range(total_batch):
                        X_train, y_train = mnist.train.next_batch(batch_size=batch_size)
                        _, c = sess.run([optimiser, cross_entropy], feed_dict=[x: X_train, y:batch_y])
                        avg_cost += c / total_batch
                print("Epoch:",(self.training_epoch+1),"cost =", "{:.3f}".format(avg_cost))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))  
