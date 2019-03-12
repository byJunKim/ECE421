import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    return x.clip(min=0)

def softmax(x):
    exp = np.exp(x)
    denom = np.sum(exp)
    return(exp/denom)


def computeLayer(X, W, b):
    # TODO
    return(X @ W + b)

#prediction = 10,000 x 10
#target - 10,000x1
def CE(target, prediction):
    N = target.shape[0] #get number of rows
    return np.sum((-1)*np.log(prediction) @ target)/N


def gradCE(target, prediction):
    N = target.shape[0]
    return((-1)/N)*np.divide(target,softmax(prediction))
    #return np.subtract(softmax(prediction),target)/N

def init():
    STD1 = (2/(784+1000)**0.5)
    STD2 = (2/(1000+10)**0.5)
    w1 = np.random.normal(0,STD1,(784,1000))
    w2 = np.random.normal(0,STD2,(1000,10))
    b2 = np.zeros((1,10))
    b1 = np.full((1,1000),0.01) #init to small value for RELU
    
    return w1,w2,b1,b2

def forward(w1,w2,b1,b2,x):
    # input layer
    S = computeLayer(x,w1,b1) # result: 10,000 x 1000
    #hidden layer
    RL = relu(S)
    #output layer
    Z  = computeLayer(RL,w2,b2) # result: 10,000 x 10
    return(softmax(Z))
    
def train(w1,w2,b1,b2,trainData,trainTarget):
    res_prob = forward(w1,w2,b1,b2,trainData)
    error = CE(res_prob, trainTarget)
#data - 10000 x (784)
#target - 10,000 x 1   
if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();
    trainData = np.reshape(trainData, (trainData.shape[0], -1))
    w1,w2,b1,b2 = init()
    
    train(w1,w2,b1,b2,trainData,trainTarget)
    
    
