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


def MSE(W, b, x, y, reg):
    mse = 0
# =============================================================================
#   Checking the dimensions of the matrices
#     print(np.shape(x))
#     print(np.shape(W))
#   Note: it's faster to multiple than divide by 5%
# =============================================================================
    mse = np.sum(((W.transpose() @ x.transpose())+b-y.transpose())**2)*(1/(2*len(y)))
    mse+= (reg/2)*(np.linalg.norm(W)**2)
    return mse

def gradMSE(W, b, x, y, reg):
 
# =============================================================================
#     gradB is 1x3500 array
#     x is 3500x784 array
# =============================================================================
    
    gradB= (1/len(y))*((W.transpose() @ x.transpose()) + b - y.transpose())
#    print(np.shape(gradB)) # results above
    gradW= (gradB @ x).transpose() + reg*W
    gradB = np.sum(gradB)
    
    return gradW, gradB
    
def crossEntropyLoss(W, b, x, y, reg):
    pass

def gradCE(W, b, x, y, reg):
    pass

def calcAcc(W,x,y,b):
    return ((((W.transpose() @ x.transpose()) + b).transpose() > 0.5).astype(int) == y).astype(int).sum() / y.shape[0]
# While the error is above threshold, keep updating weight matrix
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    print (np.shape(trainingData))
    print(np.shape(W))
    
# =============================================================================
#     Flatten the matrices
# =============================================================================
    N = trainingData.shape[0]
    x = np.reshape(trainingData, (N, -1))
    W = np.reshape(W, (-1, 1))
# =============================================================================
#     Make new variables (lists to store accuracy and loss)
# =============================================================================
    errorList = []
    accuracyList = []
    wDiff = 100
# =============================================================================
#     Initial calculations and storage
# =============================================================================
    
    accuracy = calcAcc(W,x,trainingLabels,b)
    print(accuracy)
    error = MSE(W,b,x,trainingLabels,reg)
    errorList.append(error)
    accuracyList.append(accuracy)
# =============================================================================
#     Gradient Descent
# =============================================================================
    for i in range(iterations):
        if wDiff < EPS:
            break
        
        gradW, gradB = gradMSE(W,b,x,trainingLabels,reg)
        W_new = np.subtract(W,alpha*gradW)
        wDiff = np.linalg.norm(W_new-W)
 #       print(np.shape(W_new),np.shape(W))
        W = W_new
        b = b - alpha*gradB
   
# =============================================================================
#     Update lists
# =============================================================================
        error= MSE(W,b,x,trainingLabels,reg)
        errorList.append(error)
        accuracy = calcAcc(W,x,trainingLabels,b)
        accuracyList.append(accuracy)
        
        print("Epoch: " ,i, " Error: ", error, " Accuracy: ",accuracy)
    return W,b,errorList,accuracyList

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    pass

def plotLab1(errorsList,accuracyList,learningRate):
    plt.plot(errorsList)
    plt.ylabel("Error/Loss")
    plt.xlabel("Epoch")
    plt.title(f"Lab 1 Part 1: Loss w/ LR {learningRate}")
    plt.savefig(f'lab1part1LOSSplotLR{learningRate}.png')
    
    plt.plot(errorsList)
    plt.ylabel("Accuracy of predictions")
    plt.xlabel("Epoch")
    plt.title(f"Lab 1 Part 1: Accuracy w/ LR {learningRate}")
    plt.savefig(f'lab1part1ACCURACYplotLR{learningRate}.png')
    

def main():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W = np.ones(shape=(784))
    b = 0
    LR = [0.0001, 0.001, 0.005]
    for lr in LR:
        W,b,errorList,accuracyList = grad_descent(W,b,trainData, trainTarget, lr, 5000, 0, 10**(-7)) #10**(-7)
        plotLab1(errorList,accuracyList,lr)
    
if __name__ == "__main__":
    main()