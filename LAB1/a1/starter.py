import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from script import calcClosedFormSolution

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

def sigmoidZ(x,W,b):
    #print(np.shape(W),np.shape(x))
    return(1/(1+np.exp(-(W.transpose() @ x.transpose() + b))))

def safelog(x, minval=np.finfo(np.float).eps):
    return np.log(x.clip(min=minval))
            
def crossEntropyLoss(W, b, x, y, reg):
    
# =============================================================================
#     x= np.reshape(x,(x.shape[0],-1))
#     W.reshape((-1,1))
# =============================================================================
    #print(np.shape(x), np.shape(W), np.shape(y))
    loss =((-1*y.transpose() * safelog(sigmoidZ(x,W,b))) - ((1-y.transpose()) * safelog(1-sigmoidZ(x,W,b))))
    #print(np.shape(loss))
    loss = np.sum(loss)*(1/len(y))
    loss += (reg/2)*(np.linalg.norm(W)**2)
    #print("Loss is: ",loss)
    return loss

def gradCE(W, b, x, y, reg):

    # =============================================================================
    sigmoid = sigmoidZ(x,W,b).clip(min=np.finfo(np.float).eps,max = 1 - np.finfo(np.float).eps)
    gradB = (1/len(y))*(sigmoid ** 2) *np.exp(-1 *((W.transpose() @ x.transpose()) + b))
    gradB *= (((-1 * y.transpose()) /sigmoid)+((1-y.transpose())/(1-sigmoid)))
    gradW = gradB * x.transpose()
    gradB = np.sum(gradB)
    gradW = np.sum(gradW, axis=1)
    gradW = np.expand_dims(gradW, axis=1)
    #print("Shape of gradW is: ", np.shape(gradW), "shape of W is: ", np.shape(W))
    gradW += reg*W
    return gradW, gradB
# =============================================================================


def calcAcc(W,x,y,b):
    return np.sum(((((W.transpose() @ x.transpose()) + b).transpose() > 0.5).astype(int) ==y).astype(int)) / x.shape[0]
# While the error is above threshold, keep updating weight matrix
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType):
    #print (np.shape(trainingData))
    #print(np.shape(W))
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
#     LINEAR REGRESSION
# =============================================================================
    if lossType is "LIN":
        accuracy = calcAcc(W,x,trainingLabels,b)
        print(accuracy)
        error = MSE(W,b,x,trainingLabels,reg)
        errorList.append(error)
        accuracyList.append(accuracy)
    # =========================================================================
    #     Gradient Descent
    # =========================================================================
        for i in range(iterations):
            if wDiff < EPS:
                break
            
            gradW, gradB = gradMSE(W,b,x,trainingLabels,reg)
            W_new = np.subtract(W,alpha*gradW)
            wDiff = np.linalg.norm(W_new-W)
     #       print(np.shape(W_new),np.shape(W))
            W = W_new
            b = b - alpha*gradB
       
    # =========================================================================
    #     Update lists
    # =========================================================================
            error= MSE(W,b,x,trainingLabels,reg)
            errorList.append(error)
            accuracy = calcAcc(W,x,trainingLabels,b)
            accuracyList.append(accuracy)
            
            print("Epoch: " ,i, " Error: ", error, " Accuracy: ",accuracy)
            
# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================
    elif lossType is "LOG":
        
        error = crossEntropyLoss(W,b,x,trainingLabels,reg)
        accuracy = calcAcc(W,x,trainingLabels,b)
        errorList.append(error)
        accuracyList.append(accuracy)
        
        # Gradient Descent Cross Entropy
        for i in range(iterations):
            if wDiff < EPS:
                break
            gradW, gradB = gradCE(W,b,x,trainingLabels,reg)
            W_new = np.subtract(W, alpha*gradW)
            wDiff = np.linalg.norm(W-W_new)
            W = W_new
            b = b - alpha*gradB
            
            error = crossEntropyLoss(W,b,x,trainingLabels,reg)
            errorList.append(error)
            accuracy = calcAcc(W,x,trainingLabels,b)
            accuracyList.append(accuracy)
        
            print("Epoch: " ,i, " Error: ", error, " Accuracy: ",accuracy)
            
            
    return W,b,errorList,accuracyList

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    pass

def plotLab(errorsList,accuracyList,learningRate, lab_part, _label):
    
    def plot_loss():
        plt.figure(1)
        line = plt.plot(errorsList,label = _label)
        plt.xlabel("Epoch")
        plt.title(f"Lab 1 Part {lab_part}: Loss w/ LR {learningRate}")
        plt.savefig(f'lab1part{lab_part}LOSSplotLR{learningRate}.png')
        plt.legend()
        
    
    def plot_acc():
        plt.figure(2)
        line = plt.plot(accuracyList, label = _label)
        plt.ylabel("Accuracy of predictions")
        plt.xlabel("Epoch")
        plt.title(f"Lab 1 Part {lab_part}: Accuracy w/ LR {learningRate}")
        plt.savefig(f'lab1part{lab_part}ACCURACYplotLR{learningRate}.png')
        plt.legend()
        
    plot_loss()
    plot_acc()
    
def plotLab1REG(errorsList,accuracyList,reg, lab_part):
    plt.subplots(errorsList)
    plt.xlabel("Epoch")
    plt.title(f"Lab 1 Part {lab_part}: Loss w/ REG {reg}")
    plt.savefig(f'lab1part{lab_part}LOSSplotREG{reg}.png')
    
    plt.subplots(accuracyList)
    plt.ylabel("Accuracy of predictions")
    plt.xlabel("Epoch")
    plt.title(f"Lab 1 Part {lab_part}: Accuracy w/ REG {reg}")
    plt.savefig(f'lab1part{lab_part}ACCURACYplotREG{reg}.png')
    plt.legend()

def test_closed_form(X,Y):
    
    W_l = calcClosedFormSolution(X, Y)
    
    N = X.shape[0]
    X = np.reshape(X, (N, -1))
    W_l = np.reshape(W_l, (-1, 1))

    error = MSE(W_l, 0, X, Y, 0)
    accuracy = calcAcc(W_l,X,Y,0)
    print("W_l Error: ", error, "| W_l accuracy: ", accuracy)
    
def main():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W = W = np.random.rand(784,1)
    W_2 = W
    b = 0

    W,b, errors, accuracies = grad_descent(W, b, trainData, trainTarget, 0.005, 5000, 0.1, 10**-7, "LOG")
    plotLab(errors,accuracies,0.005,"2: LIN VS LOG", " logistic regression")
    b_2 = 0
    W_2,b_2, errors, accuracies = grad_descent(W_2, b_2, trainData, trainTarget, 0.005, 5000, 0, 10**-7, "LIN")
    plotLab(errors,accuracies,0.005,'2: LIN VS LOG', " linear regression")
    
 #This section finds the closed form solution and computes its error & accuracy   
# =============================================================================
    #test_closed_form(trainData,trainTarget)
# =============================================================================
    
    
if __name__ == "__main__":
    main()