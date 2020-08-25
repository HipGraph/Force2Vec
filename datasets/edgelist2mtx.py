import networkx as nx
import sys

if __name__ == "__main__":
	edgefile = sys.argv[1]
	outputfile = open(edgefile+".mtx", "w")
	print("Reading file ...")
	G = nx.read_edgelist(edgefile)
	print("Writing file ...")
	outputfile.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
	outputfile.write("%-------------------------------------------------------------------------------\n")
	outputfile.write(str(G.number_of_nodes()) + " " + str(G.number_of_nodes()) + " " + str(G.number_of_edges()) + "\n")
	for edges in G.edges:
		x = edges[0]
		y = edges[1]
		outputfile.write(str(x) + " " + str(y) + "\n")
	outputfile.close()
	print("Complete!")
