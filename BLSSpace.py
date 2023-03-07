import numpy as np
from sklearn import preprocessing
from numpy import dtype, random, sqrt
from scipy import linalg as LA
import time
from util import *
import torch.nn as nn


# hidden feature extracted from EEG model input to bls for enhancemnt in broad space
class BLS:
    def __init__(self, N1, N2, N3, s, c, train_x, train_y,idx):
        self.weightOfEachFeatureWindow = []
        if N1*N2 >= N3:
            random.seed(67797325+idx)
            self.weightOfEnhancement = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
        else:
            random.seed(67797325+idx)
            self.weightOfEnhancement = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T
        self.outputWeight = []
        self.train_x = train_x
        self.train_y = train_y
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.s = s
        self.c = c

    def train(self,idx):
        # L=0
        train_x = preprocessing.scale(self.train_x, axis=1)
        FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
        OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], self.N2 * self.N1])
        self.distOfMaxAndMin = []
        self.minOfEachWindow = []
        time_start = time.time()
        # 生成特征映射结点
        for i in range(self.N2):
            random.seed(1999+i+idx)
            weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, self.N1) - 1
            FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow) # N*N1
            scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
            FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
            betaOfEachWindow = self.sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T  
            self.weightOfEachFeatureWindow.append(betaOfEachWindow)
            outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
            self.distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
            self.minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
            outputOfEachWindow = (outputOfEachWindow - self.minOfEachWindow[i]) / self.distOfMaxAndMin[i]
            OutputOfFeatureMappingLayer[:, self.N1 * i:self.N1 * (i + 1)] = outputOfEachWindow

        InputOfEnhanceLayerWithBias = np.hstack(
            [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
        tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, self.weightOfEnhancement)
        self.parameterOfShrink = self.s / np.max(tempOfOutputOfEnhanceLayer)
        OutputOfEnhanceLayer = self.tansig(tempOfOutputOfEnhanceLayer * self.parameterOfShrink)

        # 生成最终输入
        InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
        pinvOfInput = self.pinv(InputOfOutputLayer, self.c)
        self.outputWeight = np.dot(pinvOfInput, self.train_y)
        time_end = time.time()
        trainTime = time_end - time_start
        OutputOfTrain = np.dot(InputOfOutputLayer, self.outputWeight)
        trainAcc = self.show_accuracy(OutputOfTrain, self.train_y)
        return trainAcc, trainTime

    def predict(self, test_x, test_y=None):
        ymin = 0
        ymax = 1
        test_x = preprocessing.scale(test_x, axis=1)
        FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], self.N2 * self.N1])
        for i in range(self.N2):
            outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, self.weightOfEachFeatureWindow[i])
            OutputOfFeatureMappingLayerTest[:, self.N1 * i:self.N1 * (i + 1)] = (ymax - ymin) * (
                    outputOfEachWindowTest - self.minOfEachWindow[i]) / self.distOfMaxAndMin[i] - ymin

        InputOfEnhanceLayerWithBiasTest = np.hstack(
            [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])

        tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, self.weightOfEnhancement)
        OutputOfEnhanceLayerTest = self.tansig(tempOfOutputOfEnhanceLayerTest * self.parameterOfShrink)
        InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
        OutputOfTest = np.dot(InputOfOutputLayerTest, self.outputWeight)

        return OutputOfTest

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape((x.shape[0],1))

    def sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = A.T.dot(A)
        m = A.shape[1]
        n = b.shape[1]
        x1 = np.zeros([m, n])
        wk = x1
        ok = x1
        uk = x1
        L1 = np.mat(AA + np.eye(m)).I
        L2 = (L1.dot(A.T)).dot(b)
        for i in range(itrs):
            ck = L2 + np.dot(L1, (ok - uk))
            ok = self.shrinkage(ck + uk, lam)
            uk = uk + ck - ok
            wk = ok
        return wk

    def shrinkage(self, a, b):
        z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
        return z

    def tansig(self, x):
        return (2 / (1 + np.exp(-2 * x))) - 1

    def pinv(self, A, reg):
        return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)