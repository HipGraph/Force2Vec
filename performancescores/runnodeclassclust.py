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
#from networkx.algorithms.community import greedy_modularity_communities
import warnings
warnings.filterwarnings("ignore")

def readEmbeddingsROLX(filename, nodes, dim):
    embfile = open(filename, "r")
    firstline = embfile.readline()
    X = [[0]*dim for i in range(nodes)]
    vertex = 0
    for line in embfile.readlines():
        tokens = line.strip().split(",")
        x = []
        for j in range(0, len(tokens)):
            t = float(tokens[j])
            x.append(t)
        X[vertex] = x
        vertex += 1
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

def readEmbeddingsHOPE(filename, nodes, dim):
    embfile = open(filename, "r")
    X = [[0]*dim for i in range(nodes)]
    vertex = 0
    firstline = embfile.readline()
    for line in embfile.readlines():
        tokens = line.strip().split()
        x = []
        for j in range(0, len(tokens)):
            t = float(tokens[j])
            x.append(t)
        X[vertex] = x
        vertex += 1
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

def readEmbeddingsHARP(filename, nodes, dim):
    dX = np.load(filename)
    print("Size of X:", len(dX))
    return dX

def readEmbeddings(filename, nodes, dim):
    embfile = open(filename, "r")
    firstline = embfile.readline()
    N = int(firstline.strip().split()[0])
    X = [[0]*dim for i in range(nodes)]
    setrange = nodes
    count = 0
    for line in embfile.readlines():
        tokens = line.strip().split()
        nodeid = int(tokens[0])-1
        x = []
        for j in range(1, len(tokens)):
            t = float(tokens[j])
            x.append(t)
        if nodeid >= nodes:
            continue
        X[nodeid] = x
        count += 1
        if count > setrange:
            break
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
    count = 0
    setrange = 30000
    for line in D:
        x = []
        for v in line:
            x.append(v)
        X.append(x)
        count += 1
        if count > setrange:
            break
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

def makeLinkPredictionData(graph, X):
    Xd = []
    Yd = []
    count = 0
    setrange = 30000
    for u in range(graph.number_of_nodes()):
        Nu = list(graph.neighbors(u))
        count += len(Nu)
        cn = 0
        totalns = 0
        dd = 0
        if count > setrange:
            break
        for n in Nu:
            if n > setrange:
                continue
            x = []
            dd += 1
            #if dd > 3:
            #    break
            if n > u:
                for d in range(len(X[0])):
                    #temp =  abs(X[u][d] - X[n][d]) / (abs(X[u][d]) + abs(X[n][d]))
                    #temp =  math.sqrt(abs(X[u][d] - X[n][d]))
                    #temp = (X[u][d] + X[n][d])/2
                    temp = X[u][d] * X[n][d]
                    x.append(temp)
                Xd.append(x)
                Yd.append(1)
                totalns += 1
        tmpnn = []
        #totalns += totalns
        if len(Nu) > graph.number_of_nodes() // 2:
            totalns = (graph.number_of_nodes() - len(Nu)) // 2
            print("Testing neighbors!")
        while cn < totalns:
            #nn = random.randint(0, graph.number_of_nodes() - 1)
            nn = random.randint(0, setrange - 1)
            if nn not in Nu and nn not in tmpnn:
                cn += 1
                x = []
                for d in range(len(X[0])):
                    #temp = abs(X[u][d] - X[nn][d])/(abs(X[u][d]) + abs(X[nn][d]))
                    #temp = math.sqrt(abs(X[u][d] - X[nn][d]))
                    #temp = (X[u][d] + X[nn][d])/2
                    temp = X[u][d] * X[nn][d]
                    x.append(temp)
                Xd.append(x)
                Yd.append(0)
                tmpnn.append(nn)
    Xd, Yd = np.array(Xd), np.array(Yd)
    indices = np.array(range(len(Yd)))
    np.random.shuffle(indices)
    Xd = Xd[indices]
    Yd = Yd[indices]
    print(len(Xd), len(Yd), count)
    return Xd, Yd

from sklearn.multiclass import OneVsRestClassifier
#this class is defined similar to deepwalk paper.
class MyClass(OneVsRestClassifier):
    def prediction(self, X, nclasses):
        ps = np.asarray(super(MyClass, self).predict_proba(X))
        #print(ps)
        predlabels = []
        for i, k in enumerate(nclasses):
            ps_ = ps[i, :]
            labels = self.classes_[ps_.argsort()[-k:]].tolist()
            predlabels.append(labels)
        return predlabels

def makeNodeClassificationData(X, truthlabelsfile):
    Xd = [[] for i in range(len(X))]
    Yd = [[] for i in range(len(X))]
    distinctlabels = set()
    lfile = open(truthlabelsfile)
    for line in lfile.readlines():
        tokens = line.strip().split()
        node = int(tokens[0])-1
        label = int(tokens[1])
        #if label > 40:
        #    break
        Xd[node] = X[node]
        Yd[node].append(label)
        distinctlabels.add(label)
    lfile.close()
    Xd = [row for row in Xd if len(row)>0]
    Yd = [row for row in Yd if len(row)>0]
    return np.array(Xd), np.array(Yd), len(distinctlabels)

from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
def graphReconstruction(Gx, Xm):
    N = len(list(Gx.nodes))
    arange = range(0, N-1)
    V = 1000
    randvertices = random.sample(arange, V)
    totalcorrect = 0
    totalwrong = 0
    for i in randvertices:
        degree = len(list(Gx[i]))
        trueN = list(Gx[i])
        distances = []
        for j in range(N):
            if i == j:
                break
            dist = cosine_similarity([Xm[i]], [Xm[j]]).tolist()[0]
            distances.append([j, dist])
        distances.sort(key = lambda x: x[1], reverse=True)
        for j in range(degree):
            if i == j:
                break
            if distances[j][0] in trueN:
                totalcorrect += 1
            else:
                totalwrong += 1
        #print(totalcorrect)
    print("Graph Reconstruction Accuracy (",V,"):", (totalcorrect)/(totalcorrect+totalwrong))

#print(sys.argv)
print("Reading graph!")
filename = sys.argv[1]
G = mmread(filename)
graph = nx.Graph(G)
print("Done!")

#random.seed(1)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

if sys.argv[2] == "1":
    print("Running native...", mminfo(filename)[0])
    X = readEmbeddings(sys.argv[3],  mminfo(filename)[0], int(sys.argv[4]))
elif sys.argv[2] == "3":
    print("Running native...", mminfo(filename)[0])
    X = readEmbeddingsHOPE(sys.argv[3],  mminfo(filename)[0], int(sys.argv[4]))
elif sys.argv[2] == "4":
    X = readEmbeddingsROLX(sys.argv[3],  mminfo(filename)[0], int(sys.argv[4]))
elif sys.argv[2] == "5":
    X = readEmbeddingsHARP(sys.argv[3], mminfo(filename)[0], int(sys.argv[4]))
else:
    print("Running binnative...", mminfo(filename)[0])
    X = readBinEmbeddings(sys.argv[3], int(sys.argv[4]))

truelabfile = sys.argv[5]

print("Making prediction data!!")

Xt, Yt, labs = makeNodeClassificationData(X, truelabfile)
indices = np.array(range(len(Yt)))
assert(len(indices) == len(Yt))
onehotencode = MultiLabelBinarizer(range(labs))
#print(X)

###Graph Reconstruction
#graphReconstruction(graph, X)

'''
trainfrac = [0.15]
for tf in trainfrac:
    np.random.shuffle(indices)
    CV = int(len(Yt) * tf)
    trainX = Xt[indices[0:CV]]
    testX = Xt[indices[CV:]]
    trainY = Yt[indices[0:CV]]
    testY = Yt[indices[CV:]]
    trainY = onehotencode.fit_transform(trainY)
    testY = onehotencode.fit_transform(testY)
    #print(trainY)
    ncs = [1 for x in testY]
    print("\n\n")
    for lb in range(labs):
        trainYp = trainY[:, lb]
        testYp = testY[:, lb]
        modelLR = MyClass(LogisticRegression(random_state=0)).fit(trainX, trainYp)
        if sum(trainYp) == 0:
            continue
        predictedY = modelLR.prediction(testX, ncs)
        predictedY = np.squeeze(predictedY, axis=0)
        #print(predictedY)
        #print(testY)
        f1macro = f1_score(predictedY, testYp, average='macro')
        f1micro = f1_score(predictedY, testYp, average='micro')
        print("Label:", lb, "Multilabel-classification:", tf, "F1-macro:", f1macro, "F1-micro:",f1micro)
'''

trainfrac = [0.05, 0.10, 0.15, 0.20, 0.25]
#trainfrac = [0.01, 0.02, 0.03, 0.04, 0.05]
for tf in trainfrac:
    np.random.shuffle(indices)
    CV = int(len(Yt) * tf)
    #print(CV, indices[0:CV])
    trainX = Xt[indices[0:CV]]
    testX = Xt[indices[CV:]]
    trainY = Yt[indices[0:CV]]
    trainY = onehotencode.fit_transform(trainY)
    #print(trainY)
    testY = Yt[indices[CV:]]
    modelLR = MyClass(LogisticRegression(random_state=0)).fit(trainX, trainY)
    ncs = [len(x) for x in testY]
    predictedY = modelLR.prediction(testX, ncs)
    testY = onehotencode.fit_transform(testY)
    #print(predictedY)
    #print(testY)
    f1macro = f1_score(onehotencode.fit_transform(predictedY), testY, average='macro')
    f1micro = f1_score(onehotencode.fit_transform(predictedY), testY, average='micro')
    print("Multilabel-classification:", tf, "F1-macro:", f1macro, "F1-micro:",f1micro)

from sklearn.cluster import KMeans
import community as comm
nxG = nx.Graph(G)
bestModularity = 0
Xt = X
bestC = 2
NOC = 50
vectorMod = []
for cls in range(2, NOC):
    clusters = KMeans(n_clusters=cls, random_state=0).fit(Xt)
    predG = dict()
    for node in range(len(clusters.labels_)):
        predG[node] = clusters.labels_[node]
    modularity = comm.community_louvain.modularity(predG, nxG)
    vectorMod.append(modularity)
    if modularity > bestModularity:
        bestModularity = modularity
        bestC = cls
        #print("Sofar:", bestModularity, bestC)
print("Best Modularity:",bestModularity, "Clusters:", bestC)
print("All Modularities:",vectorMod)
'''
print("Making link prediction")
Xt, Yt = makeLinkPredictionData(graph, X)
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
    print("Link predictions:", tf, ":Accuracy:",acc, "F1-macro:", f1macro, "F1-micro:",f1micro)

print("Link prediction done!")

from sklearn.cluster import KMeans
import community as comm
nxG = nx.Graph(G)
bestModularity = 0
Xt = X
bestC = 2
NOC = 50
vectorMod = []
for cls in range(2, NOC):
    clusters = KMeans(n_clusters=cls, random_state=0).fit(Xt)
    predG = dict()
    for node in range(len(clusters.labels_)):
        predG[node] = clusters.labels_[node]
    modularity = comm.community_louvain.modularity(predG, nxG)
    vectorMod.append(modularity)
    if modularity > bestModularity:
        bestModularity = modularity
        bestC = cls
        #print("Sofar:", bestModularity, bestC)
print("Best Modularity:",bestModularity, "Clusters:", bestC)
print("All Modularities:",vectorMod)
'''
