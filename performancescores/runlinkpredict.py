from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.io import mmread,mminfo
from scipy.sparse import csr_matrix
import sys
import networkx as nx
from networkx.algorithms import community
import random, math
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
import warnings
warnings.filterwarnings("ignore")

def readEmbeddings(filename, nodes, dim):
    embfile = open(filename, "r")
    firstline = embfile.readline()
    N = int(firstline.strip().split()[0])
    X = [[0]*dim for i in range(nodes)]
    for line in embfile.readlines():
        tokens = line.strip().split()
        nodeid = int(tokens[0])-1
        x = []
        for j in range(1, len(tokens)):
            t = float(tokens[j])
            x.append(t)
        X[nodeid] = x
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

def readBinEmbeddings(filename, dim):
    embfile = open(filename, "r")
    D = np.fromfile(filename, np.float32)
    length = D.shape[0]
    embd_shape = [int(length / dim), dim]
    D = D.reshape(embd_shape)
    X = []
    for line in D:
        x = []
        for v in line:
            x.append(v)
        X.append(x)
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

def makeLinkPredictionData(graph, X, dist):
    Xd = []
    Yd = []
    count = 0
    for u in range(graph.number_of_nodes()):
        Nu = list(graph.neighbors(u))
        count += len(Nu)
        cn = 0
        totalns = 0
        for n in Nu:
            x = []
            if n > u:
                for d in range(len(X[0])):
                    if dist == "hadamard":
                        temp = X[u][d] * X[n][d]
                    elif dist == "l1":
                        temp = abs(X[u][d] - X[n][d])
                    elif dist == "l2":
                        temp = abs(X[u][d] - X[n][d]) * abs(X[u][d] - X[n][d])
                    elif dist == "average":
                        temp = (X[u][d] + X[n][d])/2
                    x.append(temp)
                Xd.append(x)
                Yd.append(1)
                totalns += 1
        tmpnn = []
        totalns += totalns
        if len(Nu) > graph.number_of_nodes() // 2:
            totalns = (graph.number_of_nodes() - len(Nu)) // 2
            print("Testing neighbors!")
        while cn < totalns:
            nn = random.randint(0, graph.number_of_nodes() - 1)
            if nn not in Nu and nn not in tmpnn:
                cn += 1
                x = []
                for d in range(len(X[0])):
                    if dist == "hadamard":
                        temp = X[u][d] * X[nn][d]
                    elif dist == "l1":
                        temp = abs(X[u][d] - X[nn][d])
                    elif dist == "l2":
                        temp = abs(X[u][d] - X[nn][d]) * abs(X[u][d] - X[nn][d])
                    elif dist == "average":
                        temp = (X[u][d] + X[nn][d])/2
                    x.append(temp)
                Xd.append(x)
                Yd.append(0)
                tmpnn.append(nn)
    Xd, Yd = np.array(Xd), np.array(Yd)
    indices = np.array(range(len(Yd)))
    np.random.shuffle(indices)
    Xd = Xd[indices]
    Yd = Yd[indices]
    #Xd = Xd[0:2*count,]
    #Yd = Xd[0:2*count,]
    print(len(Xd), len(Yd), count)
    return Xd, Yd

#print(sys.argv)
filename = sys.argv[1]
G = mmread(filename)
graph = nx.Graph(G)

#random.seed(1)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

if sys.argv[2] == "1":
    print("Running native...")
    X = readEmbeddings(sys.argv[3],  mminfo(filename)[0], int(sys.argv[4]))
else:
    X = readBinEmbeddings(sys.argv[3], int(sys.argv[4]))

truelabfile = sys.argv[5]

Xt, Yt = makeLinkPredictionData(graph, X, "hadamard")
ltrainfrac = [0.50]
for tf in ltrainfrac:
    CV = int(len(Yt) * tf)
    trainX = Xt[0:CV]
    testX = Xt[CV:]
    trainY = Yt[0:CV]
    testY = Yt[CV:]
    modelLR = LogisticRegression().fit(trainX, trainY)
    predictedY = modelLR.predict(testX)
    acc = accuracy_score(predictedY, testY)
    f1macro = f1_score(predictedY, testY, average='macro', labels=np.unique(predictedY))
    f1micro = f1_score(predictedY, testY, average='micro', labels=np.unique(predictedY))
    print("Link predictions(Hadamard):", tf, ":Accuracy:",acc, "F1-macro:", f1macro, "F1-micro:",f1micro)

'''
Xt, Yt = makeLinkPredictionData(graph, X, "l1")
ltrainfrac = [0.50]
for tf in ltrainfrac:
    CV = int(len(Yt) * tf)
    trainX = Xt[0:CV]
    testX = Xt[CV:]
    trainY = Yt[0:CV]
    testY = Yt[CV:]
    modelLR = LogisticRegression().fit(trainX, trainY)
    predictedY = modelLR.predict(testX)
    acc = accuracy_score(predictedY, testY)
    f1macro = f1_score(predictedY, testY, average='macro', labels=np.unique(predictedY))
    f1micro = f1_score(predictedY, testY, average='micro', labels=np.unique(predictedY))
    print("Link predictions(L1):", tf, ":Accuracy:",acc, "F1-macro:", f1macro, "F1-micro:",f1micro)

Xt, Yt = makeLinkPredictionData(graph, X, "l2")
ltrainfrac = [0.50]
for tf in ltrainfrac:
    CV = int(len(Yt) * tf)
    trainX = Xt[0:CV]
    testX = Xt[CV:]
    trainY = Yt[0:CV]
    testY = Yt[CV:]
    modelLR = LogisticRegression().fit(trainX, trainY)
    predictedY = modelLR.predict(testX)
    acc = accuracy_score(predictedY, testY)
    f1macro = f1_score(predictedY, testY, average='macro', labels=np.unique(predictedY))
    f1micro = f1_score(predictedY, testY, average='micro', labels=np.unique(predictedY))
    print("Link predictions(L2):", tf, ":Accuracy:",acc, "F1-macro:", f1macro, "F1-micro:",f1micro)

Xt, Yt = makeLinkPredictionData(graph, X, "average")
ltrainfrac = [0.50]
for tf in ltrainfrac:
    CV = int(len(Yt) * tf)
    trainX = Xt[0:CV]
    testX = Xt[CV:]
    trainY = Yt[0:CV]
    testY = Yt[CV:]
    modelLR = LogisticRegression().fit(trainX, trainY)
    predictedY = modelLR.predict(testX)
    acc = accuracy_score(predictedY, testY)
    f1macro = f1_score(predictedY, testY, average='macro', labels=np.unique(predictedY))
    f1micro = f1_score(predictedY, testY, average='micro', labels=np.unique(predictedY))
    print("Link predictions(Average):", tf, ":Accuracy:",acc, "F1-macro:", f1macro, "F1-micro:",f1micro)
'''
