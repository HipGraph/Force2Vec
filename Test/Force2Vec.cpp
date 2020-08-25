#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <random>

#include "../sample/algorithms.h"

using namespace std;

/* #define INDEXTYPE int */ // already defined in algorithm.h 

void helpmessage(){
	printf("\n");
	printf("Usage of Force2Vec tool:\n");
	printf("-input <string>, full path of input file (required).\n");
	printf("-output <string>, directory where output file will be stored. (default: current directory)\n");
	printf("-batch <int>, size of minibatch. (default:384)\n");
	printf("-iter <int>, number of iteration. (default:1200)\n");
	printf("-threads <int>, number of threads, default value is maximum available threads in the machine.\n");
	printf("-dim <int>, size of embedding dimension. (default:128) \n");
	printf("-nsamples <int>, number of negative samples. (default:5) \n");
	printf("-lr <float>, learning rate of SGD. (default:0.02)\n");
	printf("-option <int>, an integer among 1 to 11. (default:5)\n");
	printf("	-option 1 - for Force2Vec (O(n^2) version).\n");
        printf("        -option 2 - for BatchLayout (O(n^2) version).\n");
        printf("        -option 3 - for LinLog (O(n^2) version).\n");
        printf("        -option 4 - for ForceAtlas (O(n^2) version).\n");
        printf("        -option 5 - for t-distribution + negative sampling (tForce2Vec).\n");
        printf("        -option 6 - for sigmoid + negative sampling (sForce2Vec).\n");
        printf("        -option 7 - for sigmoid + semi-random walk (rForce2Vec).\n");
        printf("        -option 8 - for option 5 with SIMD(AVX512). \n");
        printf("        -option 9 - for option 6 with SIMD(AVX512). \n");
        printf("        -option 10 - for option 7 with SIMD(AVX512).\n");
	printf("        -option 11 - for option 8 with load balancing..\n");
	printf("-h, show help message.\n");

}

void TestAlgorithms(int argc, char *argv[]){
	VALUETYPE gamma = 1.0, lr = 0.02;
	INDEXTYPE batchsize = 384, iterations = 1200, numberOfThreads = omp_get_max_threads(), dim = 128, option = 5, nsamples = 5;
	string inputfile = "", initfile = "", outputfile = "", algoname = "f2v", initname = "RAND";
	for(int p = 0; p < argc; p++){
		if(strcmp(argv[p], "-input") == 0){
			inputfile = argv[p+1];
		}
		else if(strcmp(argv[p], "-output") == 0){
			outputfile = argv[p+1];
		}
		else if(strcmp(argv[p], "-batch") == 0){
			batchsize = atoi(argv[p+1]);
		}
		else if(strcmp(argv[p], "-iter") == 0){
			iterations = atoi(argv[p+1]);
		}
		else if(strcmp(argv[p], "-threads") == 0){
			numberOfThreads = atoi(argv[p+1]);
		}
		else if(strcmp(argv[p], "-dim") == 0){
			dim = atoi(argv[p+1]);
		}
		else if(strcmp(argv[p], "-gamma") == 0){
			gamma = atof(argv[p+1]);
		}
		else if(strcmp(argv[p], "-option") == 0){
                        option = atoi(argv[p+1]);
			if(option == 1){
				algoname = "f2v";
			}
			else if(option == 2){
				algoname = "f2vbl";
			}else if(option == 3){
				algoname = "f2vll";
			}else if(option == 4){
				algoname = "f2vfa";
			}else if(option == 5){
				algoname = "tf2vns";
			}else if(option == 6){
				algoname = "sf2vns";
			}else if(option == 7){
				algoname = "rf2vnsrweff";
			}else if(option == 8){
				algoname = "tf2vns_avxz";
			}else if(option == 9){
                                algoname = "sf2vnsrw_avxz";
                        }else if(option == 10){
                                algoname = "rf2vnsrweff_avxz";
                        }else if(option == 11){
                                algoname = "tf2vnslb_avxz";
                        }
                }
		else if(strcmp(argv[p], "-lr") == 0){
			lr = atof(argv[p+1]);
		}
		else if(strcmp(argv[p], "-nsamples") == 0){
			nsamples = atoi(argv[p+1]);
		}
		else if(strcmp(argv[p], "-h") == 0){
			helpmessage();
			exit(1);
		}
	}
	if(inputfile.size() == 0){
		printf("Valid input file needed!...\n");
		exit(1);
	}
	CSR<INDEXTYPE, VALUETYPE> A_csr;
        SetInputMatricesAsCSR(A_csr, inputfile);
        A_csr.Sorted();
	vector<VALUETYPE> outputvec;
	algorithms algo = algorithms(A_csr, inputfile, outputfile, dim, gamma, batchsize);
	srand(1);
	A_csr.make_empty();
	cout << "Running: " << algoname << endl;
	if(option == 1){
		outputvec = algo.AlgoForce2Vec(iterations, numberOfThreads, batchsize);
		//outputvec = algo.AlgoForce2VecBR(iterations, numberOfThreads, batchsize);
	}else if(option == 2){
		outputvec = algo.AlgoForce2VecFR(iterations, numberOfThreads, batchsize);
	}else if(option == 3){
		outputvec = algo.AlgoForce2VecLL(iterations, numberOfThreads, batchsize);
	}else if(option == 4){
		outputvec = algo.AlgoForce2VecFA(iterations, numberOfThreads, batchsize);
	}else if(option == 5){
		outputvec = algo.AlgoForce2VecNS(iterations, numberOfThreads, batchsize, nsamples, lr);
	}else if(option == 6){
                outputvec = algo.AlgoForce2VecNSRW(iterations, numberOfThreads, batchsize, nsamples, lr);
        }else if(option == 7){
                outputvec = algo.AlgoForce2VecNSRWEFF(iterations, numberOfThreads, batchsize, nsamples, lr);
        }

	if(option > 7){
#ifdef AVX512
	if(option == 8){
                outputvec = algo.AlgoForce2VecNS_SREAL_D128_AVXZ(iterations, numberOfThreads, batchsize, nsamples, lr);
        }else if(option == 10){
		if(dim == 128){
                	outputvec = algo.AlgoForce2VecNSRWEFF_SREAL_D128_AVXZ(iterations, numberOfThreads, batchsize, nsamples, lr);
        	}else if(dim == 64){
			outputvec = algo.AlgoForce2VecNSRWEFF_SREAL_D64_AVXZ(iterations, numberOfThreads, batchsize, nsamples, lr);
		}
	}else if(option == 11){
		if(dim == 128){
                	outputvec = algo.AlgoForce2VecNSLB_SREAL_D128_AVXZ(iterations, numberOfThreads, batchsize, nsamples, lr);
        	}else if(dim == 64){
			outputvec =  algo.AlgoForce2VecNSLB_SREAL_D64_AVXZ(iterations, numberOfThreads, batchsize, nsamples, lr);
		}
	}
	else if(option == 9){
		if(dim == 64){
			outputvec = algo.AlgoForce2VecNSRWLB_SREAL_D64_AVXZ(iterations, numberOfThreads, batchsize, nsamples, lr);
		}else if(dim == 128){
                #ifdef SREAL 
                  outputvec = algo.AlgoForce2VecNSRW_SREAL_D128_AVXZ(iterations, numberOfThreads, batchsize, nsamples, lr);
                #else
                  cout << "Intrinsic code not implemented for double yet!" << endl;  
                  exit(1);
                #endif
		}else{
		  cout << "Intrinsic code not implemented for various dimensions!" << endl;
                  exit(1);
		}
	}
#else
printf("Force2Vec has not been compiled with AVX512...Choose an option from 5 to 8. \n");
exit(1);
#endif		
        }
	
	//A_csr.make_empty();
	string avgfile = "Results.txt";
        ofstream output;
       	output.open(avgfile, ofstream::app);
	output << "Algo:"<< algoname << "\tInit:" << initname << "\tIteration:";
       	output << iterations << "\tNumofthreads:" << numberOfThreads << "\tBatchSize:" << batchsize << "\tDimension:" << dim << "\tTime(sec.):";
        output << outputvec[0] << "\t";
	output << endl;
	output.close();
}
int main(int argc, char* argv[]){
	TestAlgorithms(argc, argv);
        return 0;
}
