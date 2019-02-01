import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        print("Success")
    return trainData, validData, testData, trainTarget, validTarget, testTarget

    # for training: W^T must be a 28x28 weight matrix
    # x[i] = 28x28 (picture)
def MSE(W, b, x, y, reg):
    mse = 0
    for i in range(len(y)):
        mse+=(1/(2*len(y)))*(W.dot(np.transpose(x[i].flatten()))+b-y[i])**2 + reg/2*np.linalg.norm(W)**2
    return mse

def gradMSE(W, b, x, y, reg):
    #calculate gradient with respect to W
    gradW=0
    gradB=0
    for i in range(len(y)):
        gradW += (1/len(y))*((W.dot(x[i].flatten().transpose())+b-y[i])*x[i].flatten().transpose()) + reg*W
        gradB += 1/len(y)*(W.dot(x[i].flatten().transpose())+b-y[i])
    #calculate gradient with respect to regularization-tor
    return gradW, gradB
    
def crossEntropyLoss(W, b, x, y, reg):
    pass

def gradCE(W, b, x, y, reg):
    pass

# While the error is above threshold, keep updating weight matrix
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    errorList = []
    error = MSE(W,b,trainingData,trainingLabels,reg)
    errorList.append(error)
    while( error > EPS):
        # idk if this syntax is right gotta go check later
        print(error)
        gradW, gradB = gradMSE(W,b,trainingData,trainingLabels,reg)
        W = np.subtract(W,alpha*gradW)
        b = b - alpha*gradB
        error = MSE(W,b,trainingData,trainingLabels,reg)
        errorList.append(error)
        
    return errorList

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    pass

def plotLab1(errorsList):
    plt.plot(errorsList)
    plt.ylabel("Error/Loss")
    plt.xlabel("Epoch")
    plt.title("Lab 1 Part 1: Plotting loss against epochs")
    plt.savefig('lab1part1plot.png')
    

def main():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W = np.ones(shape=(784))
    b = 0
    errorList = grad_descent(W,b,trainData, trainTarget, 0.001, 5000, 0, 10**(-7)) #10**(-7)
    plotLab1(errorList)
    
if __name__ == "__main__":
    main()