import sys
import networkx as nx
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.io import mmread,mminfo
from scipy.sparse import csr_matrix
from networkx.algorithms import community
import random, math
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def readFile(filename, size=30000):
	infile = open(filename,"r")
	edge = []
	counter = 1
	maxn = 0
	print("Reading Input File ", filename)
	for line in infile.readlines():
		if line.startswith("%"):
			continue
		tokens = line.strip().split()
		x = int(tokens[0])-1
		y = int(tokens[1])-1
		if x >= size or y >= size:
			continue
		edge.append((x,y))
		maxn = max([maxn,x,y])
		#print(x,y)
		counter += 1
		if counter > size:
			break
	infile.close()
	return edge,maxn

def readEmbeddings(filename, nodes, dim):
	embfile = open(filename, "r")
	firstline = embfile.readline()
	N = int(firstline.strip().split()[0])
	X = [[0]*dim for i in range(nodes)]
	counter = 0
	for line in embfile.readlines():
		tokens = line.strip().split()
		nodeid = int(tokens[0])-1
		x = []
		for j in range(1, len(tokens)):
			t = float(tokens[j])
			x.append(t)
		X[nodeid] = x
		counter += 1
		if counter >= nodes:
			break
	embfile.close()
	print("Size of X:", len(X))
	return np.array(X)

def readBinEmbeddings(filename, nodes, dim):
	embfile = open(filename, "r")
	D = np.fromfile(filename, np.float32).reshape(39454746, dim)
	X = []
	count = 0
	for line in D:
		x = []
		for v in line:
			x.append(v)
		X.append(x)
		count += 1
		if count >= nodes:
			break
	embfile.close()
	print("Size of X:", len(X))
	return np.array(X)

def makeLinkPredictionData(graph, X):
	Xd = []
	Yd = []
	count = 0
	allNodes = list(graph.nodes)
	for u in allNodes:
		Nu = list(graph.neighbors(u))
		count += len(Nu)
		cn = 0
		totalns = 0
		for n in Nu:
			if n <= u:
				continue
			x = []
			for d in range(len(X[0])):
				x.append(X[u][d] * X[n][d])
			Xd.append(x)
			Yd.append(1)
			totalns += 1
		tmpnn = []
		totalns += totalns
		if len(Nu) > graph.number_of_nodes() // 2:
			totalns = (graph.number_of_nodes() - len(Nu)) // 2
			print("Testing neighbors!")

		while cn < totalns:
			nn = random.choice(allNodes)
			if nn not in Nu and nn not in tmpnn:
				cn += 1
				x = []
				for d in range(len(X[0])):
					x.append(X[u][d] * X[nn][d])
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

if __name__ == "__main__":
	fname = sys.argv[1]
	size = int(sys.argv[2])
	E,maxn = readFile(fname, size)
	G = nx.Graph()
	G.add_edges_from(E)
	if int(sys.argv[4]) == 1:
		X = readEmbeddings(sys.argv[3], maxn+1, 64)
	else:
		X = readBinEmbeddings(sys.argv[3], maxn+1, 64)

	print("Making link prediction data")
	Xt, Yt = makeLinkPredictionData(G, X)
	CV = int(len(Yt) * 0.5)
	trainX = Xt[0:CV]
	testX = Xt[CV:]
	trainY = Yt[0:CV]
	testY = Yt[CV:]
	print("Preparing model...")
	modelLR = LogisticRegression().fit(trainX, trainY)
	predictedY = modelLR.predict(testX)
	acc = accuracy_score(predictedY, testY)
	f1score = f1_score(predictedY, testY, average='macro', labels=np.unique(predictedY))
	print("Link Prediction: Accuracy:",acc, "F1-Score:", f1score)	
