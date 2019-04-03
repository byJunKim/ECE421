#the functions were run in jupyer notebook
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import math

# Loading data
data = np.load('data2D.npy')
data100D = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)
is_valid = True
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO

    vector = []
    for i in range(MU.shape[0]):
        temp_vec = tf.math.square(tf.math.subtract(X, MU[i])) #elementwise subtraction
        temp_vec = tf.math.reduce_sum(temp_vec, 1)
        vector.append(temp_vec)
    vector = tf.convert_to_tensor(vector)
    return tf.transpose(vector) #final shape N x K

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    # TODO
    
    log_Gaussian = []
    dim = int(X.shape[1])
    det = tf.math.pow(sigma, dim)
    first_term = tf.math.log(1.0/(((2.0*math.pi)**(dim/2))*det))
    second_term = -0.5*distanceFunc(X, mu)/(2.0*tf.transpose(sigma*sigma))
    result = tf.transpose(first_term) + second_term 
    
    return result

def log_posterior(log_PDF, log_pi):
    #compute log(P(z|x)) = log(z) + log(P(x|z)) - log(sump(x|z)) = log_pi + log_PDF - logsum(log_PDF)
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1
    # Outputs
    # log_post: N X K
    a = ((tf.add(tf.broadcast_to(log_pi, shape = (log_PDF.shape[1], log_PDF.shape[0])), tf.transpose(log_PDF))))
    b = reduce_logsumexp(log_PDF, keep_dims= True)
    b = tf.broadcast_to(b, shape = (log_PDF.shape[0], log_PDF.shape[1]))
    b = tf.transpose(b)
    return (tf.subtract(a,b))



def a3_train(trainData, validData, k, lr = 0.1, num_epochs = 700):
    trainData = trainData.astype('float32')
    
    
    train_losses = np.zeros(num_epochs)
    valid_losses = np.zeros(num_epochs)
    
    dim = trainData.shape[1]
    #placeholders
    data_holder = tf.placeholder(tf.float32, shape = (trainData.shape[0], dim))
    valid_data_holder = tf.placeholder(tf.float32, shape = (validData.shape[0], dim))
    
    MU = tf.Variable(tf.random_normal(shape = (k, dim)))
    phi = tf.math.exp(tf.Variable(tf.random_normal(shape = (k, 1))))
    
    pi =  tf.math.exp(tf.Variable(tf.random_normal(shape = (k, 1))))
    pi = logsoftmax(pi)


    #calculate loss
    train_log_PDF = log_GaussPDF(data_holder, MU, tf.math.sqrt(phi)) #Prbability X|Z

    train_log_pi = tf.broadcast_to(pi, shape = (k, trainData.shape[0]))

    train_log_pi = tf.transpose(train_log_pi)

    train_PX = reduce_logsumexp(tf.math.add(train_log_PDF, train_log_pi), keep_dims = True)
    test = tf.reduce_sum(train_PX, 0)
    train_loss_func = tf.math.negative((tf.reduce_sum(train_PX, 0)))
    
    
    valid_log_PDF = log_GaussPDF(valid_data_holder, MU, tf.math.sqrt(phi)) #Prbability X|Z
    valid_log_pi = tf.broadcast_to(pi, shape = (k, validData.shape[0]))
    valid_log_pi = tf.transpose(valid_log_pi)
    valid_PX = reduce_logsumexp(tf.math.add(valid_log_PDF, valid_log_pi), keep_dims = True)
    
    valid_loss_func = tf.math.negative((tf.reduce_sum(valid_PX, 0)))
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.9, beta2 =0.999, epsilon = 1e-8)
    minimize_opt = optimizer.minimize(train_loss_func)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(init)
        
        for epoch in range(num_epochs):
            loss = 0.0
            v_loss = 0.0    
            
            #x_training = np.split(trainData[:9984], 312)
            #y_training, y_valid, y_test = convertOneHot(trainTarget, validTarget, testTarget)
            #y_training = np.split(y_training[:9984], 312)
            
            #x_valid = validData
            #y_valid = y_valid

            #labels = y_training[i]
            #print("train_log_PDF", sess.run(train_log_PDF, feed_dict = {data_holder: trainData}))
            #print("log pi", sess.run(train_log_pi, feed_dict = {data_holder: trainData}))
            #print("PX", sess.run(train_PX, feed_dict = {data_holder: trainData}))
            #print("test", sess.run(test, feed_dict = {data_holder: trainData}))
            loss = sess.run(train_loss_func, feed_dict={data_holder: trainData})

            v_loss = sess.run(valid_loss_func, feed_dict={valid_data_holder: validData})
            sess.run(minimize_opt, feed_dict = {data_holder: trainData})
            
            
            
            train_losses[epoch] = loss#/trainData.shape[0]
            valid_losses[epoch] = v_loss#/validData.shape[0]
            print("Epoch:", epoch+1)
            print("train loss:", train_losses[epoch], "valid loss:", valid_losses[epoch])
            
            #shuffle data here
            #trainData, trainTarget = shuffle(trainData, trainTarget)
        clusters = MU.eval()
        assignments = sess.run(get_clusters(validData, MU))
    return train_losses, valid_losses, clusters, assignments



def plot_error(errorsList):

    def plot_loss():
        plt.figure(2)
        plt.plot(errorsList, label = "Loss")
        plt.xlabel("Epoch")
        plt.title("Total Validation Loss")
        plt.legend()

    plot_loss()

#build the scatterplot. Clusters = centers, assignments = assignments of each point to a centerpoint
def plot_data(dataset, clusters, assignments):
  k = len(clusters)
  plt.figure(1)
  plt.scatter(dataset[:, 0], dataset[:, 1], c=assignments, cmap=plt.get_cmap('Set1'), s=25, alpha=0.6)
  plt.scatter(clusters[:, 0], clusters[:, 1], marker='o', c=range(k), cmap=plt.get_cmap('Set1'), s=150, linewidths=1)
  plt.title('K-Means Clustering')
  plt.xlabel('X1')
  plt.ylabel('X2')
  plt.grid()

def square_distance(x, y): # return pairwise square dist

    dot_prod = tf.matmul(y, tf.transpose(x))
    sum = tf.transpose(norm(x)) + norm(y)

    return sum - (2 * dot_prod)


def get_clusters(dataset, clusters):
    min_dist = tf.argmin(square_distance(clusters, dataset.astype('float32')), 1)
    #print(min_dist)
    return min_dist

def norm(x):
    return tf.reduce_sum(x*x, 1, keep_dims=True)

#this functions were run in jupyer notebook
