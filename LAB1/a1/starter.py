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

def plotLab1(errorsList, validationErrorList, testErrorList,learningRate, train_accuracyList = None, valid_accuracyList = None, test_accuracyList = None ):
    plt.plot(errorsList, label = "Training Loss")
    plt.plot(validationErrorList, label = "Validation Loss")
    plt.plot(testErrorList, label = "Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(("Lab 1 Part 1: Training and Validation Loss w/ LR={}".format(learningRate)))
    plt.legend(loc='best')
    plt.show()
    plt.savefig(('lab1_part1_training_loss_plot_lr={}.png').format(learningRate))
    
    plt.plot(train_accuracyList, label = "Test Accuracy")
    plt.plot(valid_accuracyList, label = "Test Accuracy")
    plt.plot(test_accuracyList, label = "Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(("Lab 1 Part 1: Accuracy w/ LR={}".format(learningRate)))
    #plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.show()
    plt.savefig(('lab1_part1_accuracy_plot_lr={}.png').format(learningRate))
    
def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, optimizer = None, learning_rate=0.001, batch_size = 500):
    #Initialize weight and bias tensors
    x, validData, testData, y, validTarget, testTarget = loadData()
    tf.set_random_seed(421)
    W = tf.truncated_normal(shape = (28*28, 1), stddev = 0.5, dtype = tf.float32, seed = 421, name = "weight")
    B = tf.truncated_normal(shape = (1,1), stddev = 0.5, dtype = tf.float32, seed = 421, name = "bias")
    reg = 0
    epoch = 700
    if lossType == "MSE":
        return SGD_MSE(x, y, validData, validTarget, testData, testTarget, W, B, learning_rate, reg, epoch, batch_size, opt = optimizer, beta1_ = beta1, beta2_ = beta2, epsilon_ = epsilon)
    elif lossType == "CE":
        return SGD_CE(x, y, validData, validTarget, testData, testTarget, W, B, learning_rate, reg, epoch, batch_size, opt = optimizer, beta1_ = beta1, beta2_ = beta2, epsilon_ = epsilon)

def data_shuffle(data, label):
    s = np.arange(int(data.shape[0]))
    np.random.shuffle(s)
    shuffled_data = data[s]
    shuffled_label = label[s]

    return shuffled_data, shuffled_label

def SGD_MSE(trainData, trainTarget, validData, validTarget, testData, testTarget, W, b, lr, reg, epoch, mini_batch, opt = None, beta1_ = None, beta2_ = None, epsilon_ = None):  
    
    
    #flattening the data
    trainData = np.reshape(trainData, (trainData.shape[0], -1))
    validData = np.reshape(validData, (validData.shape[0], -1))
    testData = np.reshape(testData, (testData.shape[0], -1))
    
    
    #intializing
    SGD_train_losses = []
    SGD_valid_losses = []
    SGD_test_losses = []
    SGD_train_accuracy = []
    SGD_valid_accuracy = []
    SGD_test_accuracy = []
    W = tf.Variable(W)
    b = tf.Variable(b)
    reg = tf.Variable(reg)
    
    # ~ Calculations ~
    #num_of_TrImages = trainData[0]
    #num_of_batches = num_of_TrImages / mini_batch
    
    

    #select fist 500 images that are shuffled
    x_training = trainData[0:(mini_batch-1)]
    y_training = trainTarget[0:(mini_batch -1)]
    y_training_prime = tf.matmul(tf.cast(x_training, tf.float32), W) #might need to add bias here????
    training_loss = tf.losses.mean_squared_error(y_training, predictions = y_training_prime)
    train_acc = tf.math.divide(tf.math.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.math.greater(y_training_prime, tf.convert_to_tensor(0.5)), tf.int32), y_training), tf.int32)), y_training.shape[0])
    
    if opt is None:
        init = tf.global_variables_initializer()
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(training_loss)
    elif opt is "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate = lr, beta1 = beta1_, beta2 =beta2_, epsilon = epsilon_).minimize(training_loss)
        init = tf.global_variables_initializer()
    
    
    x_valid = validData[0:99]
    y_valid = validTarget[0:99]
    y_valid_prime = tf.matmul(tf.cast(x_valid, tf.float32), W)
    valid_loss = tf.losses.mean_squared_error(y_valid, predictions = y_valid_prime)
    valid_acc = tf.math.divide(tf.math.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.math.greater(y_valid_prime, tf.convert_to_tensor(0.5)), tf.int32), y_valid), tf.int32)), y_valid.shape[0])
    
    
    x_test = testData[0:144]
    y_test = testTarget[0:144]
    y_test_prime = tf.matmul(tf.cast(x_test, tf.float32), W)
    test_loss = tf.losses.mean_squared_error(y_test, predictions = y_test_prime)
    test_acc = tf.math.divide(tf.math.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.math.greater(y_test_prime, tf.convert_to_tensor(0.5)), tf.int32), y_test), tf.int32)), y_test.shape[0])
    
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(init)
        
        for i in range(epoch):
            sess.run(optimizer)
            
            SGD_train_losses.append(sess.run(training_loss))
            SGD_valid_losses.append(sess.run(valid_loss))
            SGD_test_losses.append(sess.run(test_loss))

            
            trainAcc = sess.run(train_acc)
            validAcc = sess.run(valid_acc)
            testAcc = sess.run(test_acc)
            
            SGD_train_accuracy.append(trainAcc)
            SGD_valid_accuracy.append(validAcc)
            SGD_test_accuracy.append(testAcc)
            
            
            
            print("SGD training error:  ",sess.run(training_loss), "SGD val error: ", sess.run(valid_loss), "SGD test error:", sess.run(test_loss))  
            print("SGD training accuracy: ", sess.run(train_acc), "SGD val accuracy: ",  sess.run(valid_acc), "SGD test accuracy: ", sess.run(test_acc))
            
            
            #shuffle the data and target after every epoch    
            trainData, trainTarget = data_shuffle(trainData, trainTarget)
            validData, validTarget = data_shuffle(validData, validTarget)
            testData, testTarget = data_shuffle(testData, testTarget)
            
            
            
        sess.close()

        plotLab1(SGD_train_losses, SGD_valid_losses, SGD_test_losses, lr, train_accuracyList = SGD_train_accuracy, valid_accuracyList = SGD_valid_accuracy, test_accuracyList = SGD_test_accuracy, part = 3)
        
        return W, b, y_test_prime, y_test, optimizer, reg
    
def SGD_CE(trainData, trainTarget, validData, validTarget, testData, testTarget, W, b, lr, reg, epoch, mini_batch, opt = None, beta1_ = None, beta2_ = None, epsilon_ = None):
    
    
    #flattening the data
    trainData = np.reshape(trainData, (trainData.shape[0], -1))
    validData = np.reshape(validData, (validData.shape[0], -1))
    testData = np.reshape(testData, (testData.shape[0], -1))
    
    
    #intializing
    CE_train_losses = []
    CE_valid_losses = []
    CE_test_losses = []
    CE_train_accuracy = []
    CE_valid_accuracy = []
    CE_test_accuracy = []
    W = tf.Variable(W)
    b = tf.Variable(b)
    reg = tf.Variable(reg)
    # ~ Calculations but never used~
    #num_of_TrImages = trainData[0]
    #num_of_batches = num_of_TrImages / mini_batch
    
    
    x_training = trainData[0:(mini_batch-1)]
    y_training = trainTarget[0:(mini_batch -1)]
    y_training_prime = tf.matmul(tf.cast(x_training, tf.float32), W) #might need to add bias here????
    training_loss = tf.losses.sigmoid_cross_entropy(y_training, y_training_prime)
    train_acc = tf.math.divide(tf.math.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.math.greater(y_training_prime, tf.convert_to_tensor(0.5)), tf.int32), y_training), tf.int32)), y_training.shape[0])
    
    if opt is None:
        init = tf.global_variables_initializer()
        optimizer = tf.train.GradientDescentOptimizer(lr)
        minimize_opt = optimizer.minimize(training_loss)
    elif opt is "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate = lr, beta1 = beta1_, beta2 =beta2_, epsilon = epsilon_)
        minimize_opt = optimizer.minimize(training_loss)
        init = tf.global_variables_initializer()
        
    x_valid = validData[0:99]
    y_valid = validTarget[0:99]
    y_valid_prime = tf.matmul(tf.cast(x_valid, tf.float32), W)
    valid_loss = tf.losses.sigmoid_cross_entropy(y_valid, y_valid_prime)
    valid_acc = tf.math.divide(tf.math.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.math.greater(y_valid_prime, tf.convert_to_tensor(0.5)), tf.int32), y_valid), tf.int32)), y_valid.shape[0])
    
    
    x_test = testData[0:144]
    y_test = testTarget[0:144]
    y_test_prime = tf.matmul(tf.cast(x_test, tf.float32), W)
    test_loss = tf.losses.sigmoid_cross_entropy(y_test, y_test_prime)
    test_acc = tf.math.divide(tf.math.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.math.greater(y_test_prime, tf.convert_to_tensor(0.5)), tf.int32), y_test), tf.int32)), y_test.shape[0])
        
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(init)
        
        for i in range(epoch):
            sess.run(minimize_opt)
            
            CE_train_losses.append(sess.run(training_loss))
            CE_valid_losses.append(sess.run(valid_loss))
            CE_test_losses.append(sess.run(test_loss))

            
            trainAcc = sess.run(train_acc)
            validAcc = sess.run(valid_acc)
            testAcc = sess.run(test_acc)
            
            CE_train_accuracy.append(trainAcc)
            CE_valid_accuracy.append(validAcc)
            CE_test_accuracy.append(testAcc)
            
            
            
            print("CE training error:  ",sess.run(training_loss), "CE val error: ", sess.run(valid_loss), "CE test error:", sess.run(test_loss))  
            print("CE training accuracy: ", sess.run(train_acc), "CE val accuracy: ",  sess.run(valid_acc), "CE test accuracy: ", sess.run(test_acc))
            
            
            #shuffle the data and target after every epoch    
            trainData, trainTarget = data_shuffle(trainData, trainTarget)
            validData, validTarget = data_shuffle(validData, validTarget)
            testData, testTarget = data_shuffle(testData, testTarget)
            
        sess.close()
        
        plotLab1(CE_train_losses, CE_valid_losses, CE_test_losses, lr, train_accuracyList = CE_train_accuracy, valid_accuracyList = CE_valid_accuracy, test_accuracyList = CE_test_accuracy, part = 3)
    
        return W, b, y_test_prime, y_test, optimizer, reg
    
    

def plotLab(errorsList,accuracyList,learningRate, lab_part, plotAcc,plotLoss, _label, hyperparam):
    
    def plot_loss():
        plt.figure(1)
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
    calcClosedFormSolution(trainData,trainTarget)
        
def test_time_batch():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W_ = np.random.rand(784,1)
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
    
    buildGraph(lossType= "MSE")
# =============================================================================
#     W,b, errors, accuracies, valErr, testErr = grad_descent(W, b, trainData, trainTarget, validData, validTarget, testData, testTarget, 0.005, 5000, 0, 10**-7, "LOG")
#     plotLab(errors,accuracies,0.1,"2",False, True, "logistic regression", "Reg")
#     b_2 = 0
#     W_2,b_2, errors, accuracies, valErr, testErr = grad_descent(W_2, b_2, trainData, trainTarget, validData, validTarget, testData, testTarget, 0.005, 5000, 0, 10**-7, "LIN")
#     plotLab(errors,accuracies,0,'2', False, True,  "linear regression", "LIN VS LOG w Reg")
# =============================================================================
    
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