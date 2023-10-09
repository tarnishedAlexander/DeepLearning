import numpy as np
import copy
def initparm(layerDim):
    np.random.seed(4)
    parameters = {}
    L = len(layerDim)
    
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layerDim[l], layerDim[l-1])
        parameters["b" + str(l)] = np.zeros((layerDim(l),1))
    return parameters

def sigmoid(Z):
    cache = Z
    A = 1/(1+np.exp(-Z))
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A,cache

def dSigmoid(dA, cache):
    Z = cache 
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def dRelu(dA, cache):
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[dZ<=0] = 0
    return dZ
    

def forward(A,W,b):
    Z = np.dot(A,W) + b
    cache = (A,W,b)
    return Z, cache

def actForward(Aprev, W, b, act):
    Z, linearCache = forward(Aprev,W,b)
    if act == "sigmoid":
        A, actCache = sigmoid(Z)
    if act == "relu":
        A, actCache = relu(Z)

    cache = (linearCache,actCache)

    return A, cache

def deepForward(X, parameters):
    caches = []
    A = X
    L = len(parameters)
    for l in range(1,L):
        Aprev = A
        A, cache = actForward(Aprev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    AL, cache = actForward(Aprev, parameters["W" + str(l)], parameters["b" + str(l)], "sigmoid")
    caches.append(cache)
    return AL, caches

def computeCost(AL, Y):
    m = Y.shape[1]
    cost = np.squeeze(-np.sum(np.dot(Y, np.log(AL) + np.dot((1-Y), np.log(1-AL))))/m)
    return cost

def backward(dZ, cache):
    Aprev, W, b = cache
    m = Aprev.shape[1]
    dW = np.dot(dZ, Aprev)/m
    db = np.sum(dZ, axis = 1, keepdims=True)
    dAprev = np.dot(W.T, dZ)
    return dAprev, dW,db

def actBackward(dA, cache, act):
    linearCache, actCache = cache
    if act == "relu":
        dZ = dRelu(dA, actCache)
        dAprev, dW,db = backward(dZ, linearCache)
    if act == "sigmoid":
        dZ = dSigmoid(dA, actCache)
        dAprev, dW, db = backward(dZ, linearCache)

    return dAprev,dW,db

def deepBackward(AL, Y, cache):
    grads = {}
    L = len(cache)
    m = AL.shape[1]
    Y = Y.reshape(Y.shape)
    dAL = -(np.divide(Y,AL) - np.divide(1-Y, 1-AL))
    currentCache = cache[L-1]
    dAprevtemp, dWtemp,dbtemp = actBackward(dAL, currentCache, "sigmoid")
    grads["dA" + str(L-1)] = dAprevtemp
    grads["db" + str(L)] = dbtemp
    grads["dW" + str(L)] = dWtemp
    return grads

def updateParams(params, grads, learningRate):
    parameters = copy.deepcopy(params)
    L= len(parameters)//2
    for l in range(L):
        parameters["W" + str(l)] -= learningRate*grads["dW" + str(l+1)]
        parameters["b" + str(l)] -= learningRate*grads["db" + str(l+1)]
    return parameters

def predic(X,Y, parameters):
    m = X.shape[1]
    n = len(Y)
    p = np.zeros((1,m))
    proba,caches = deepForward(X, parameters)

    for l in range(0, proba.shape[1]):
        if proba[0,l] > 0.5:
            proba[0, l] = 1
        else:
            proba[0,l] = 0
    return p
def LLayermodel(X,Y, layerDims, learningRate = 0.0075, iter = 3000, printCost = False):
    np.random.seed(1)
    cost = []
    parameters = initparm(layerDims)
    for l in range(0,iter):
        AL, caches = deepForward(X,parameters)
        cost = computeCost(AL,Y)
        grads= deepBackward(AL, Y, caches)
        parameters= updateParams(parameters, grads, learningRate)
    return parameters,cost