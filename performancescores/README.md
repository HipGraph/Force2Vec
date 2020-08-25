## Generate performance score for link prediction, multilabel classification and clustering

To run this experiment, type following command:
```
python datasets/experiments/runAllpredictions.py datasets/input/blogcatalog.mtx 1 datasets/output/blogcatalog.mtxF2V48D128IT1000.txt 128 datasets/input/blogcatalog.nodes.labels
```

Description: first argument -> input graph, second argument -> option of embedding type: 2 for verse as it uses bcsr format and 1 for others, third argument -> embedding file, fourth argument -> dimension of embedding, fifth argument -> file of true label of nodes.

This experiment will generate f1 (micro/macro) scores of link prediction and multilabel classification, and modularity score of clustering approach. #Clusters is set to 2-50 for K-means algorithm.
