import numpy as np
import pandas as pds
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', parser='auto')
x = mnist.data
y = mnist.target

def init_data(l,t,x):
    W_0 = np.random.rand(l,x) - 0.5
    b_0 = np.random.rand(l,1) - 0.5
    W_1 = np.random.rand(t,l) - 0.5
    b_1 = np.random.rand(t,1) - 0.5
    return W_0,b_0,W_1,b_1


def ReLU(Z):
    return np.maximum(0,Z)

def dReLU(Z):
    return Z > 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(Z):
    e_x = np.exp(Z - np.max(Z))
    return e_x / e_x.sum(axis=0)


def forward_propagation(X,W_0,b_0,W_1,b_1):
    Z_0 = (W_0.dot(X.T)) + b_0 # l x n
    A_0 = sigmoid(Z_0)
    Z_1 = (W_1.dot(A_0)) + b_1 # t x n 
    A_1 = softmax(Z_1)
    return Z_0,A_0,Z_1,A_1



def back_propagation(X,Y,W_1,Z_0,A_0,Z_1,A_1):
    n = len(X)
    dZ_1 = (A_1 - Y)
    dW_1 = 1/n * (dZ_1.dot(A_0.T)) # t x t
    db_1 = 1/n * np.sum(dZ_1)
    dZ_0 = (W_1.T.dot(dZ_1)) * d_sigmoid(Z_0) # l x n
    dW_0 = 1/n * (dZ_0.dot(X)) 
    db_0 = 1/n * np.sum(dZ_0)
    return dW_0,db_0,dW_1,db_1


def update_params(dW_1,db_1,dW_0,db_0,W_0,W_1,b_0,b_1,rate):
    W_0 = W_0 - rate * dW_0
    W_1 = W_1 - rate * dW_1
    b_0 = b_0 - rate * db_0
    b_1 = b_1 - rate * db_1
    return W_0,W_1,b_0,b_1


def convert(Y,labels_total):
    ones_y = np.zeros((Y.size,labels_total))
    ones_y[np.arange(Y.size),Y] = 1
    return ones_y.T

# reminder : setting sample_update too low, might result in bad acc
def gradient_descent(X,Y,W_0,b_0,W_1,b_1,update='batch',sample_updates=10000,runs=1000,learning_rate=0.1):
    assert(sample_updates > 0 and runs >= 10 and learning_rate > 0)
    ones_y = convert(Y,10)
    arange_X = np.arange(0,len(X))
    dW_0 = 0
    dW_1 = 0
    db_0 = 0
    db_1 = 0
    for i in range(0,runs+1):
        Z_0,A_0,Z_1,A_1 = forward_propagation(X,W_0,b_0,W_1,b_1)
        if(update == "batch-stochastic"):
            np.random.shuffle(arange_X)
            dW_0,db_0,dW_1,db_1 = back_propagation(X[arange_X[0:sample_updates],:],ones_y[:,arange_X[0:sample_updates]],W_1,Z_0[:,arange_X[0:sample_updates]],
                             A_0[:,arange_X[0:sample_updates]],Z_1[:,arange_X[0:sample_updates]],A_1[:,arange_X[0:sample_updates]])
            W_0,W_1,b_0,b_1 = update_params(dW_1,db_1,dW_0,db_0,W_0,W_1,b_0,b_1,learning_rate) 
        else:
            dW_0,db_0,dW_1,db_1 = back_propagation(X,ones_y,W_1,Z_0,A_0,Z_1,A_1)
        
        W_0,W_1,b_0,b_1 = update_params(dW_1,db_1,dW_0,db_0,W_0,W_1,b_0,b_1,learning_rate)  
        if(i % 10 == 0):
            print("acc on run {} = {}%".format(i,acc(W_0,W_1,b_0,b_1,X,ones_y)))
    return W_0,W_1,b_0,b_1

def acc(W_0,W_1,b_0,b_1,X,Y):
    errors = 0        
    res = (W_1 @ ((W_0 @ X.T) + b_0)) + b_1
    for i in range(0,len(res[0])):
        indX = np.argmax(res[:,i])
        indY = np.where(Y[:,i] == 1)
        if(indX != indY):
            errors += 1
    return round((1 - (errors / len(res[0]))) * 100,2)    
        

def create_split_set(X,Y,trainSize,random=False):
    if(random):
        indexes = np.arange(0,Y.size)
        np.random.shuffle(indexes)
        X = X[indexes,:]
        Y = Y[indexes]
    devX   = X[0:500,:]
    devY   = Y[0:500]
    trainX = X[500:trainSize,:]
    trainY = Y[500:trainSize]
    testX  = X[trainSize:,:]
    testY  = Y[trainSize:]
    return trainX,trainY,testX,testY,devX,devY

size = int(0.8 * len(x))

trainX,trainY,testX,testY,devX,devY = create_split_set(x.to_numpy(),np.asarray(y.to_numpy(),dtype=int),size)

print(trainX)
print(trainY)

W_0,b_0,W_1,b_1 = init_data(10,10,784)

W_0,W_1,b_0,b_1 = gradient_descent(trainX,trainY,W_0,b_0,W_1,b_1,update='batch-stochastic') # output layer

print("\n\n")
print("check correctness on dev data:\n")

Z_0,A_0,Z_1,A_1 = forward_propagation(devX,W_0,b_0,W_1,b_1)
errors = 0
for i in range(0,len(devY)):
    res = np.where(A_1[:,i] == np.max(A_1[:,i]))[0][0]
    actual_label_i = devY[i]
    if actual_label_i != res:
        errors += 1
    print("sample {} : actual_label = {} --- predicted_label = {}\n".format(i,actual_label_i,res))

acc_dev = round((1 - (errors / devY.size)) * 100,2)
print("overall acc on dev training = {}%".format(acc_dev))    



