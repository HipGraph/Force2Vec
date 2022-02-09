# Force2Vec
This is the official implementation for IEEE ICDM 2020 paper titled "Force2Vec: Parallel force-directed graph embedding".

[**PDF is available in arXiv**](https://arxiv.org/pdf/2009.10035.pdf)

## System Requirements
Users need to have the following softwares/tools installed in their PC/server. The source code was compiled and run successfully in both Linux and macOS.
```
GCC version >= 4.9
OpenMP version >= 4.5
```
Some helpful links for installation can be found at [GCC](https://gcc.gnu.org/install/), [OpenMP](https://clang-omp.github.io) and [Environment Setup](http://heather.cs.ucdavis.edu/~matloff/158/ToolsInstructions.html#compile_openmp).

If you would like to use python scripts to get performance scores and visualization, you need to have following python packages:
```
scikit-learn v0.22.2
networkx v2.3
scipy v1.3.1
numpy v1.16.2
matplotlib v3.1.1
python-louvain v0.13
```

## Compile Force2Vec
To compile Force2Vec, type the following command on terminal:
```
$ make clean
$ make
```
This will generate an executible file inside the bin folder.

If your machine supports AVX512 instruction set, then you can compile the code as follows:
```
$ make clean
$ make AVX512=true
```
On a Linux terminal, you can type `lscpu` to see supported flags in your machine. If you see avx512\* there, then your machine supports AVX512 instruction set.
Please note that you will need avx512f and avx512dq extension to run our AVX512 intrinsic codes. 

## Users: Run Force2Vec from Command Line

Input file must be in matrix market format ([check here for details about .mtx file](https://math.nist.gov/MatrixMarket/formats.html)). A lot of datasets can be found at [suitesparse website](https://sparse.tamu.edu). We provide a few input files inside the  datasets/input directory. To run Force2Vec, type the following command:
```
$ ./bin/Force2Vec -input ./datasets/input/cora.mtx -output ./datasets/output/ -iter 1200 -batch 256 -threads 32 -option 5
```
Here, `-input` is the full path of input file, `-output` is the directory where output/embedding file will be saved, `-iter` is the number of iterations, `-batch` is the size of minibatch which is 256 here, `-threads` is the maximum number of threads which is 32 and `-option` is the choice of algorithm to run which is 5 representing t-distribution  version for Force2Vec. All options are described below:
```
-input <string>, full path of input file (required).
-output <string>, directory where output file will be stored. (default: current directory)
-batch <int>, size of minibatch. (default:384)
-iter <int>, number of iteration. (default:1200)
-threads <int>, number of threads, default value is maximum available threads in the machine.
-dim <int>, size of embedding dimension. (default:128)
-nsamples <int>, number of negative samples. (default:5)
-lr <float>, learning rate of SGD. (default:0.02)
-bs <int>, chose 0 for s negative samples or 1 for batch * s negative samples. (default:0)
-option <int>, an integer among 1 to 11. (default:5)
        1 - for Force2Vec (O(n^2) version).
        2 - for BatchLayout (O(ns) version).
        3 - for LinLog (O(ns) version).
        4 - for ForceAtlas (O(ns) version).
        5 - for t-distribution + negative sampling (tForce2Vec).
        6 - for sigmoid + negative sampling (sForce2Vec).
        7 - for sigmoid + semi-random walk (rForce2Vec).
        8 - for option 5 with SIMD(AVX512).
        9 - for option 6 with SIMD(AVX512).
        10 - for option 7 with SIMD(AVX512).
	11 - for option 8 with load balancing.
```
First line of output file will contains the number of vertices (N) and the embedding dimension (D). The following N lines will contain vertex id and a D-dimensional embedding for corresponding vertex id.

N.B.: To reproduce the runtime results, we recommend to run the option 11 provided that you have AVX512 instruction set. For general purpose usages, run Force2Vec with any option from  5, 6, and 7. AVX512 currently supports 64 and 128 dimensional embeddings.

### Code Generator ###
We have a code generator in sample/kgen directory to generate various attractive and repulsive force kernels based on t-distribution and sigmoid, floating point precision, and dimension (d). By setting appropriate macro in simd.h, you can apply those generated kernels for different machines (currently supported AVX512, AVX2, AVX, SSE in X86, ASIMD, NEON in ARM64 and VSX in OpenPOWER). We have a script file in that directory as well which can be used to generate all kernels in a range of values of dimension. Please see README file in that directory for details.    

### Convert Edgelist Input File to Matrix Market Format ###
To increase the usability of Force2Vec for other file types, we have a python script that can convert edgelist input format to matrix market format. It uses `networkx` python package. You can convert an edgelist input file as follows:
```
$ python edgelist2mtx.py input/cora.edgelist
``` 
This will create a cora.edgelist.mtx file inside the input folder where cora.edgelist file is located.


## Generate Performance Scores of an Embedding ##

To generate the  quality scores of embeddings by performing prediction task like link prediction, multilabel classification and clustering, choose a file from the performancescores folder and run the python file with specific commands.

```
$ python -u performancescores/runnodeclassclust.py datasets/input/cora.mtx 1 datasets/output/cora.mtxF2V384D16IT1000.embd 16 datasets/input/cora.nodes.labels
```
Here, `runnodeclassclust.py` is the file for computing node classification and clustering measures, `1` is the option to read embedding file of Force2Vec (there are other options to read embeddings generated by other methods), `cora.mtxF2V384D16IT1000.embd` is the embedding file generated by Force2Vec, `16` is the dimension of the embedding, `cora.nodes.labels` is the file of ground truth node labels. The formate of the input graph and ground truth files must be the same as given inside the datasets/input folder. In the similar way, you can run link prediction task as follows (default: Hadamard):

```
$ python -u performancescores/runlinkpredict.py datasets/input/cora.mtx 1 datasets/output/cora.mtxF2V384D16IT1000.embd 16 datasets/input/cora.nodes.labels
```
For link prediction task, we recommend to use sigmoid with negative samples i.e., option 6.

## Generate 2D Visualizations of an Embedding ##
To generate 2D visualiation using t-SNE, run the following command which will generate a PDF file in the current directory:
```
$ python -u performancescores/runvisualization.py datasets/input/cora.mtx 1 datasets/output/cora.mtxF2V384D16IT1000.embd 16 datasets/input/cora.nodes.labels coragraph
```

## Citation
If you find this repository helpful, please cite our paper as follows:

Md. Khaledur Rahman, Majedul Haque Sujon and Ariful Azad, "Force2Vec: Parallel Force-Directed Graph Embedding", The 20th IEEE International Conference on Data Mining (IEEE ICDM 2020), November, 2020, Sorrento, Italy.
```
@inproceedings{rahman2020force2vec,
  title={Force2Vec: Parallel force-directed graph embedding},
  author={Rahman, Md Khaledur and Sujon, Majedul Haque and Azad, Ariful},
  booktitle={2020 IEEE International Conference on Data Mining (ICDM)},
  year={2020},
  organization={IEEE}
}
@article{rahman2022scalable,
  title={Scalable force-directed graph representation learning and visualization},
  author={Rahman, Md and Sujon, Majedul Haque and Azad, Ariful and others},
  journal={Knowledge and Information Systems},
  pages={1--27},
  year={2022},
  publisher={Springer}
}
```
## Contact 
This repository is maintained by Md. Khaledur Rahman. If you have questions, please ping me at `morahma@iu.edu` or create an issue.
