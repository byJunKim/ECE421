import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from collections import Counter, defaultdict

def analyze_data(classifications): # get unique labels (cluster) and # of pts in each label 

    print(np.unique(classifications,return_counts=True))

def plot_error(errorsList):

    def plot_loss():
        plt.figure(2)
        plt.plot(errorsList, label = "Error")
        plt.xlabel("Epoch")
        plt.ylim((0,1))
        plt.title(f"Error against Iterations")
        plt.legend()

    plot_loss()

def plot_data(dataset, clusters, classes):
    k = len(clusters)
    plt.figure(1)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=classes, cmap=plt.get_cmap('Set1'), s=25, alpha=0.6)
    plt.scatter(clusters[:, 0], clusters[:, 1], marker='o', c=range(k), cmap=plt.get_cmap('Set1'), s=150, linewidths=1)
    plt.title('K-Means Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid()

def norm(x):
    return tf.reduce_sum(x*x, 1, keep_dims=True)

def square_distance(x, y): # return pairwise square dist

    dot_prod = tf.matmul(y, tf.transpose(x))
    sum = tf.transpose(norm(x)) + norm(y)

    return sum - (2 * dot_prod)

def loss_function(x,mu):
    return tf.reduce_mean(tf.reduce_min(square_distance(mu, x), 1))

def get_clusters(dataset, clusters):
    min_dist = tf.argmin(square_distance(clusters, dataset), 1)
    #print(min_dist)
    return min_dist

def k_means(data, k, iterations):

    dim = len(data[0])
    dataset = tf.placeholder(tf.float32, [None, dim], name="Data")
    mu = tf.Variable(tf.random_normal([k, dim]), name="Centroids")

    # Training specification
    cost = loss_function(mu, dataset)
    optimizer = tf.train.AdamOptimizer(0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(cost)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        costs = []
        last_cost = float('inf')
       # min_valid_cost = 100000
        for i in range(iterations):
            iter_cost = sess.run([cost, optimizer], feed_dict={dataset: data})[0]
            #if valid_data.any():
                #valid_cost = sess.run([cost, optimizer], feed_dict={dataset: valid_data})[0]
                #min_valid_cost = min(min_valid_cost, valid_cost)
            costs.append(iter_cost)

            print ("Iteration:", i, "Loss:", iter_cost, "\n")

            if abs(iter_cost - last_cost) == 0 :
               # if valid_data.any():
                   # print ("Converged! Lowest validation loss is:", min_valid_cost, "ending validation cost is:", valid_cost)
                clusters = mu.eval()
                break
            else:
                last_cost = iter_cost

        assignments = sess.run(get_clusters(dataset, mu), feed_dict={dataset: data})

    return clusters, assignments, costs


def a3part1(data, K, is_valid, iterations):
        # For Validation set
    if is_valid:
      valid_batch = int(data.shape[0] / 3.0)
      np.random.seed(45689)
      rnd_idx = np.arange(data.shape[0])
      np.random.shuffle(rnd_idx)
      val_data = data[rnd_idx[:valid_batch]]
      data = data[rnd_idx[valid_batch:]]

    #print(val_data.shape)

    clusters, assignments, costs = k_means(data, K, iterations)
    print("clusters shape", clusters.shape,"assignments shape", assignments.shape)
    plot_data(data, clusters, assignments)
    analyze_data(assignments)
    plot_error(costs)
    
if __name__ == '__main__':

    # Loading data
    data = np.load('data2D.npy')

    K = 3
    is_valid = False
    iterations = 10000
    
    #run part one of assignment 3
    a3part1(data, K, is_valid, iterations)
