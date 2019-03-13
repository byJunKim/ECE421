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
    denom = np.sum(exp,axis=1).reshape(x.shape[0], 1)
    return(exp/denom)
    
    
def safelog(x, minval=np.finfo(np.float).eps):
    return np.log(x.clip(min=minval))

def computeLayer(X, W, b):
    return(X @ W + b)
    
#prediction = 10,000 x 10
#target - 10,000x10 w/ one-hot
def CE(X,y):
    n = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(n),y])
    loss = np.sum(log_likelihood) / n
    return loss

def gradCE(target, prediction):
    N = target.shape[0]
    return((-1)/N)*np.divide(target,prediction)

def init():
    STD1 = ((2/(784+1000))**0.5)
    STD2 = ((2/(1000+10))**0.5)
    w1 = np.random.normal(0,STD1,(784,1000))
    w2 = np.random.normal(0,STD2,(1000,10))
    b2 = np.zeros((1,10))
    b1 = np.full((1,1000),0.01) #init to small value for RELU
    v1 = np.full((784,1000), np.exp(-5))
    v2 = np.full((1000,10), np.exp(-5))
    v_b1 = np.full((1,1000),np.exp(-5))
    v_b2 = np.full((1,10),np.exp(-5))
    
    return w1,w2,b1,b2,v1,v2,v_b1,v_b2
    
def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad


def train(w1,w2,b1,b2,v1,v2, v_b1, v_b2,trainData,trainTarget,LR, gamma, validTarget, testTarget,epochs,validData, testData):
    Z_1 = 0
    Z_2 = 0
    a = 0
    result = 0
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
    
    def calcAcc (x,y):
        bestProbs = np.argmax(x,axis = 1)
        #print(bestProbs.shape,np.argmax(y,axis = 1).shape)
        return np.sum(bestProbs == np.argmax(y,axis = 1))/y.shape[0]
    
    def forward(w1,w2,b1,b2,x):
        #print("B1's shape is:", b1.shape)
        Z_1 = computeLayer(x,w1,b1) # result: 10,000 x 1000
        a = relu(Z_1)
        Z_2  = computeLayer(a,w2,b2) # result: 10,000 x 10
        return softmax(Z_2),Z_1,a,Z_2
    
    def gradRelu(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def backprop(w1,w2,b1,b2,predic,x,y):
        delta_3 = np.subtract(predic,y) # 1 x 10 FOR 1 PT
        delta_2 = delta_3 @ w2.transpose() * gradRelu(Z_1) # 1 x 1000
        gradW_h = x.transpose() @ delta_2 # 784 x 1 FOR ONE, 784 x 1000
        gradW_o = a.transpose() @ delta_3 # 1000 x 10
        #Note: delta_2 = gradB_h delta_1 = grad_B_o
        delta_2 = np.sum(delta_2,axis=0)
        delta_3 = np.sum(delta_3,axis=0)
        #print("delta 2 shape:" ,delta_2.shape)
        Vh_new = LR*gradW_h + gamma*v1 #momentum for w1
        Vo_new = LR*gradW_o + gamma*v2 #momentum for w2
        V_b1_new = LR*delta_2 + gamma*v_b1
        V_b2_new = LR*delta_3 + gamma*v_b2
        
        #updates
        w1_new = w1 - Vh_new
        w2_new = w2 - Vo_new
        #print("b1 shape original is", b1.shape)
        b1_new = b1 - V_b1_new
        #print("b1 shape new is", b1_new.shape)
        b2_new = b2 - V_b2_new
        
        return w1_new,w2_new,b1_new,b2_new,Vh_new,Vo_new,V_b1_new,V_b2_new
    
    for i in range(epochs):
        #print("B1's shape is:", b1.shape)
        result,Z_1,a,Z_2 = forward(w1,w2,b1,b2,trainData)
        result_validation, Z1_valid,a_valid,Z2_valid = forward(w1,w2,b1,b2,validData)
        result_test, Z1_test,a_test,Z2_test = forward(w1,w2,b1,b2,testData)
        error = CE(Z_2,trainTarget)
        error_valid = CE(Z2_valid,validTarget)
        error_test = CE(Z2_test,testTarget)
        accuracy = calcAcc(result,newtrain)
        accuracy_validation = calcAcc(result_validation, newvalid)
        accuracy_test = calcAcc(result_test, newtest)
        print("epoch:" ,i, "Training error:",error," Training accuracy:",accuracy)
        print("validation error:",error_valid, "validation accuracy", accuracy_validation)
        print("test error:",error_test, "test accuracy", accuracy_test)
        ############# backward propagation ####################
        w1,w2,b1,b2,v1,v2,v_b1,v_b2 = backprop(w1,w2,b1,b2,result,trainData,newtrain)
        
#data - 10000 x (784)
#target - 10,000 x 1
if __name__ == "__main__":

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData();
    trainData = np.reshape(trainData, (trainData.shape[0], -1))
    testData = np.reshape(testData, (testData.shape[0],-1))
    validData = np.reshape(validData, (validData.shape[0],-1))
    w1,w2,b1,b2,v1,v2,v_b1,v_b2 = init()
    
    LR=0.00005
    gamma = 0.9
    epochs = 200
    train(w1,w2,b1,b2,v1,v2,v_b1,v_b2,trainData,trainTarget, LR, gamma, validTarget, testTarget, epochs, validData, testData)
    
    