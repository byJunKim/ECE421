import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from collections import Counter, defaultdict

def analyze_data(classifications):
    
    print(np.unique(classifications,return_counts=True))
        
def ploterror(errorsList):
    
    def plot_loss():
        plt.figure(2)
        plt.plot(errorsList, label = "Error")
        plt.xlabel("Epoch")
        plt.ylim((-1,1))
        plt.title(f"Error against Iterations")
        plt.legend()
        
    plot_loss()

        
        
def plot_data(dataset, clusters, classes):
    '''

    Draws a scatterplot of the clusterized data
    Args:
        dataset: Dataset to be plotted
        clusters: Clusters centers
        classes: Clases of each point in the dataset

    '''
    k = len(clusters)
    plt.figure(1)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=classes, cmap=plt.get_cmap('Set1'), s=25, alpha=0.6)
    plt.scatter(clusters[:, 0], clusters[:, 1], marker='*', c=range(k), cmap=plt.get_cmap('Set1'), s=500, linewidths=3)
    plt.title('K-Means Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid()
    
def matrix_norm_column(x):
    '''
    Calculates the norm of each vector in a matrix, returning a column matrix
    '''
    x_sq = x*x
    norm = tf.reduce_sum(x_sq, 1, keep_dims=True)
    return norm

def square_distance(x, y):
    '''
    Returns pairwise squared distance between two matrices:
    For points xi belonging to X and yj belonging to Y, we have:
        d(xi,yj)^2 = ||xi - yj||^2 = ||xi||^2 + ||xj||^2 - 2 * (yj . xi)
    Expanding to the entire matrix, we get:
        d(X, Y)^2 = (||X||^2 .+ ||Y||^2) - 2*dot(Y,X')
            Where .+ is the outer (element-wise) sum
    '''

    x_norm = matrix_norm_column(x)
    y_norm = matrix_norm_column(y)
    dot_prod = tf.matmul(y, tf.transpose(x))

    outer_sum = tf.transpose(x_norm) + y_norm

    return outer_sum - 2 * dot_prod

def loss_function(x,mu):
    dists = square_distance(mu, x) #||x - mu||
    min_dist = tf.reduce_min(dists, 1) # min ||x - mu||
    cost = tf.reduce_mean(min_dist) # sum(min ||x - mu||)
    return cost



def assign_data(dataset, clusters):
    '''

    Calculates the cluster assignments given a dataset and cluster centers
    Args:
        dataset: Set of points
        clusters: Centers of clusters

    Returns:
        min_dist: List of point classes in the same order they appear in the dataset

    '''
    dists = square_distance(clusters, dataset)  # ||x - mu||
    min_dist = tf.argmin(dists, 1)  # argmin ||x - mu||
    return min_dist




def k_means(data, k, EXP=1e-6):
    '''

    Performs K-Means clusterization on a dataset
    Args:
        data: Set of points
        k: Number of clusters to use
        EXP: Convergence criteria (minimum difference between iteration cost before stopping

    Returns:
        clusters: Cluster centers
        assignments: Class assignment of each point
        costs: Cost history throughout training

    '''
    data_len = len(data)
    assert (data_len > 0), "Dataset is empty"
    assert (k < data_len), "Invalid value of K for size of dataset"

    dim = len(data[0])

    # Input data and Clusters
    dataset = tf.placeholder(tf.float32, [None, dim], name="Data")
    mu = tf.Variable(tf.random_normal([k, dim]), name="Centroids")

    # Training specification
    cost = loss_function(mu, dataset)
    iter_var = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(cost, global_step=iter_var)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        costs = []
        last_cost = float('inf')
        for i in range(1000):
            iter_cost = sess.run([cost, optimizer], feed_dict={dataset: data})[0]

            costs.append(iter_cost)

            print ("Iteration:", i, "Loss:", iter_cost, "\n")

            if abs(iter_cost - last_cost) == 0 :
                print ("Converged!")
                clusters = mu.eval()
                break
            else:
                last_cost = iter_cost

        assignments = sess.run(assign_data(dataset, mu), feed_dict={dataset: data})

    return clusters, assignments, costs


if __name__ == '__main__':
    
    # Loading data
    data = np.load('data2D.npy')
    
    K = 2
    is_valid = False
    iterations = 10000
    
    
    clusters, assignments, costs = k_means(data, K, iterations)
    print("clusters shape", clusters.shape,"assignments shape", assignments.shape)
    print(assignments)
    plot_data(data, clusters, assignments)
    analyze_data(assignments)
    #ploterror(costs)