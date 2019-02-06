import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from script import calcClosedFormSolution
import timeit

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
    mse = np.sum(((x@W)+b-y)**2)*(1/(2*len(y)))
    mse+= (reg/2)*(np.linalg.norm(W)**2)
    return mse

def gradMSE(W, b, x, y, reg):
 
# =============================================================================
#     gradB is 1x3500 array
#     x is 3500x784 array
# =============================================================================
    
    gradB= (1/len(y))*((x@W) + b - y)
#    print(np.shape(gradB)) # results above
    gradW= (x.transpose() @ gradB) + reg*W
    gradB = np.sum(gradB)
    
    return gradW, gradB

def sigmoidZ(x,W,b):
    #print(np.shape(W),np.shape(x))
    return(1/(1+np.exp(-(x@W + b))))

def safelog(x, minval=np.finfo(np.float).eps):
    return np.log(x.clip(min=minval))
            
def crossEntropyLoss(W, b, x, y, reg):
    
# =============================================================================
#     x= np.reshape(x,(x.shape[0],-1))
#     W.reshape((-1,1))
# =============================================================================
    #print(np.shape(x), np.shape(W), np.shape(y))
    loss =(((-1)*(y * safelog(sigmoidZ(x,W,b)))) - ((1-y) * safelog(1-sigmoidZ(x,W,b))))
    #print(np.shape(loss))
    loss = np.sum(loss)*(1/len(y))
    loss += (reg/2)*(np.linalg.norm(W)**2)
    #print("Loss is: ",loss)
    return loss

def gradCE(W, b, x, y, reg):

    # =============================================================================
    sigmoid = sigmoidZ(x,W,b).clip(max = 1 - np.finfo(np.float).eps)
    gradB = (1/len(y))*sigmoid *((-1)*y*np.exp(-(x@W + b)) + 1-y)
    gradW = x.transpose() @ gradB
    gradB = np.sum(gradB)
    #print("Shape of gradW is: ", np.shape(gradW), "shape of W is: ", np.shape(W))
    gradW += reg*W
    #print(gradW)
    return gradW, gradB
# =============================================================================


def calcAcc(W,x,y,b):
    return np.sum(((((x @ W) + b) > 0.5).astype(int) ==y).astype(int)) / x.shape[0]


def grad_descent(W, b, trainingData, trainingLabels, validData, validTarget, testData, testTarget, alpha, iterations, reg, EPS, lossType):
    print (np.shape(trainingData))
    print(np.shape(W))
    
# =============================================================================
#     Flatten the matrices
# =============================================================================
    N = trainingData.shape[0]
    M = validData.shape[0]
    O = testData.shape[0]
    x = np.reshape(trainingData, (N, -1))
    validData = np.reshape(validData, (M, -1))
    testData =np.reshape(testData, (O, -1))
    W = np.reshape(W, (-1, 1))
    print("1")
# =============================================================================
#     Make new variables (lists to store accuracy and loss)
# =============================================================================
    errorList = []
    accuracyList = []
    valLossList = []
    testLossList = []
    wDiff = 100
    print("2")
    if lossType is "LIN":
    # =============================================================================
    #     Initial calculations and storage
    # =============================================================================
    
        accuracy = calcAcc(W,x,trainingLabels,b)
        print("accuracy: %d" %accuracy)
        error = MSE(W,b,x,trainingLabels,reg)
        errorList.append(error)
        accuracyList.append(accuracy)
        val_loss = MSE(W, b, validData, validTarget, reg)
        valLossList.append(val_loss)
        test_loss = MSE(W, b, testData, testTarget, reg)
        testLossList.append(test_loss)
    # =============================================================================
    #     Gradient Descent
    # =============================================================================
        for i in range(iterations):
            if wDiff < EPS:
                break

            gradW, gradB = gradMSE(W, b, x, trainingLabels, reg)
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
            val_loss = MSE(W, b, validData, validTarget, reg)
            valLossList.append(val_loss)
            test_loss = MSE(W, b, testData, testTarget, reg)
            testLossList.append(test_loss)

            print("Epoch: " ,i, "Test Error: ", error, " Validation Loss: ", val_loss, " Test Loss: ", test_loss, " Accuracy: ",accuracy)
        
    elif lossType is "LOG":
        #TODO: needs to include validtion and test losses
        error = crossEntropyLoss(W,b,x,trainingLabels,reg)
        accuracy = calcAcc(W,x,trainingLabels,b)
        errorList.append(error)
        accuracyList.append(accuracy)
        val_loss = crossEntropyLoss(W, b, validData, validTarget, reg)
        valLossList.append(val_loss)
        test_loss = crossEntropyLoss(W, b, testData, testTarget, reg)
        testLossList.append(test_loss)
        
        # Gradient Descent Cross Entropy
        for i in range(iterations):
            if wDiff < EPS:
                break
            gradW, gradB = gradCE(W,b,x,trainingLabels,reg)
            W_new = np.subtract(W,alpha*gradW)
            wDiff = np.linalg.norm(W-W_new)
            
            W = W_new
            b = b - alpha*gradB
            
            error = crossEntropyLoss(W,b,x,trainingLabels,reg)
            errorList.append(error)
            accuracy = calcAcc(W,x,trainingLabels,b)
            accuracyList.append(accuracy)
            val_loss = crossEntropyLoss(W, b, validData, validTarget, reg)
            valLossList.append(val_loss)
            test_loss = crossEntropyLoss(W, b, testData, testTarget, reg)
            testLossList.append(test_loss)

            print("Epoch: " ,i, "Test Error: ", error, " Validation Loss: ", val_loss, " Test Loss: ", test_loss, " Accuracy: ",accuracy)
                  
    return W,b,errorList,accuracyList,valLossList,testLossList


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    pass

def plotLab(errorsList,accuracyList,learningRate, lab_part, plotAcc,plotLoss, _label, hyperparam):
    
    def plot_loss():
        plt.figure(1)
        plt.plot(errorsList, label = _label)
        plt.xlabel("Epoch")
        plt.ylim((0,30))
        plt.title(f"Lab 1 Part {lab_part}: Loss w/ {hyperparam} {learningRate}")
        plt.savefig(f'lab1part{lab_part}LOSSplot{hyperparam}{learningRate}.png')
        plt.legend()
        
    
    def plot_acc():
        plt.figure(2)
        plt.plot(accuracyList, label = _label)
        plt.ylabel("Accuracy of predictions")
        plt.xlabel("Epoch")
        plt.title(f"Lab 1 Part {lab_part}: Accuracy w/ {hyperparam} {learningRate}")
        plt.savefig(f'lab1part{lab_part}ACCURACYplot{hyperparam}{learningRate}.png')
        
    if (plotLoss):    
        plot_loss()
    
    if plotAcc:
        plot_acc()
    


def test_closed_form(X,Y):
    
    W_l = calcClosedFormSolution(X, Y)
    
    N = X.shape[0]
    X = np.reshape(X, (N, -1))
    W_l = np.reshape(W_l, (-1, 1))

    error = MSE(W_l, 0, X, Y, 0)
    accuracy = calcAcc(W_l,X,Y,0)
    print("W_l Error: ", error, "| W_l accuracy: ", accuracy)
    
def test_time_CF():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W = calcClosedFormSolution(trainData,trainTarget)
        
def test_time_batch():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W_ = W = np.random.rand(784,1)
    b_ = 0
    W_,b_, errors, accuracies, valLosses, testLosses = grad_descent(W_, b_, trainData, trainTarget, validData, validTarget, testData, testTarget, 0.005, 5000, 0.001, 10**-7, "LIN")
        
def main():
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W = np.random.rand(784,1)
    W_2 = W
    b = 0
    b_2 = 0
    
    def MSE2(W, b, x, y, reg):
        x = np.reshape(x, (x.shape[0], -1))
        mse = np.sum(((W.transpose() @ x.transpose())+b-y.transpose())**2)*(1/(2*len(y)))
        mse+= (reg/2)*(np.linalg.norm(W)**2)
        return mse

    def calcAcc2 (W,x,y,b):
        x = np.reshape(x,(x.shape[0],-1))
        return np.sum(((((W.transpose() @ x.transpose()) + b).transpose() > 0.5).astype(int) ==y).astype(int)) / x.shape[0]
    
    W,b, errors, accuracies, valErr, testErr = grad_descent(W, b, trainData, trainTarget, validData, validTarget, testData, testTarget, 0.005, 5000, 0, 10**-7, "LOG")
    plotLab(errors,accuracies,0.1,"2",False, True, "logistic regression", "Reg")
    b_2 = 0
    W_2,b_2, errors, accuracies, valErr, testErr = grad_descent(W_2, b_2, trainData, trainTarget, validData, validTarget, testData, testTarget, 0.005, 5000, 0, 10**-7, "LIN")
    plotLab(errors,accuracies,0,'2', False, True,  "linear regression", "LIN VS LOG w Reg")
    
    # Tuning REG
# =============================================================================
    #reg = 0.1
  
    #W,b, errors, accuracies, valLosses, testLosses = grad_descent(W, b, trainData, trainTarget, validData, validTarget, testData, testTarget, 0.005, 5000, reg, 10**-7, "LOG")
    #plotLab(errors,accuracies,reg, "2", True,True,"training error" , "Regularization")
    #plotLab(valLosses,accuracies,reg, "2", True,True,"validation error" , "Regularization")
    #plotLab(testLosses,accuracies,reg, "2", True,True,"test error" , "Regularization")
         
         #print(f"For LR = {LR} accuracy is:" ,accuracies[len(accuracies)-1])
#         
# =============================================================================
# =============================================================================
#     # Tuning Reg
#     Regs = [0.001,0.1,0.5]
#     for reg in Regs:
#         W=np.random.rand(784,1)
#         b = 0      
#         W,b, errors, accuracies, valLosses, testLosses = grad_descent(W, b, trainData, trainTarget, validData, validTarget, testData, testTarget, 0.005, 5000, reg, 10**-7, "LIN")
#         print(f"For Reg = {reg} accuracy is:" ,accuracies[len(accuracies)-1])
# =============================================================================
# =============================================================================
#     # Comparing Closed Form and Batch Losses
#     
#     W = calcClosedFormSolution(trainData,trainTarget)
#     W_2,b_2, errors, accuracies, valLosses, testLosses = grad_descent(W_2, b_2, trainData, trainTarget, validData, validTarget, testData, testTarget, 0.005, 5000, 0.001, 10**-7, "LIN")
#     
#     cfError = MSE2(W,0,trainData,trainTarget,0)
#     cfAcc = calcAcc2(W,trainData,trainTarget,b)
#     
#     ofError = MSE2(W_2,b_2,trainData,trainTarget,0)
#     ofAcc = calcAcc2(W_2,trainData,trainTarget,b_2)
#     print("Errors: Closed Form- ", cfError, "Batch- " ,ofError)
#     print("Accuracies: Closed Form- ", cfAcc,"Batch- " ,ofAcc)
# =============================================================================
    
    
    #plotLab(errors,0,0,'1', False,True, "training loss", "regularization")
    
# =============================================================================
#     plotting validatio and test losses
#     plotLab(valLosses,0,0.001,'1', False,True, "training loss", "regularization")
#     plotLab(testLosses,0,0.001,'1', False,True, "training loss", "regularization")
# =============================================================================

    # Testing time
    
    #print(timeit.timeit(stmt="test_time_CF()",setup="from __main__ import test_time_CF", number=3)/3)
    #print(timeit.timeit(stmt="test_time_batch()",setup="from __main__ import test_time_batch", number=3)/3)
    
 #This section finds the closed form solution and computes its error & accuracy   
# =============================================================================
    #test_closed_form(trainData,trainTarget)
# =============================================================================
    
# =============================================================================
#     plotLab(errors,accuracies,0.005,"2: LIN VS LOG", " logistic regression")
# =============================================================================
    
if __name__ == "__main__":
    
    main()