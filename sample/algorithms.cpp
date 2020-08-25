#include "algorithms.h"
#include "simd.h"

uint64_t rseed[2];

VALUETYPE algorithms::scale(VALUETYPE v){
	if(v > MAXBOUND) return MAXBOUND;
        else if(v < -MAXBOUND) return -MAXBOUND;
        else return v;
}

//pseudo random number generator

//http://prng.di.unimi.it/xoroshiro128plusplus.c
static inline uint64_t rotl(const uint64_t x, INDEXTYPE k) {
	return (x << k) | (x >> (64 - k));
}

uint64_t nextrand(void) {
	const uint64_t s0 = rseed[0];
	uint64_t s1 = rseed[1];
	const uint64_t result = s1 + s0;

	s1 ^= s0;
	rseed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	rseed[1] = rotl(s1, 36); // c

	return result;
}

//http://prng.di.unimi.it/#shootout

static inline VALUETYPE drand() {
  const union{uint64_t i;VALUETYPE d;} u = {UINT64_C(0x3FF) << 52 | nextrand() >> 12};
  return u.d - 1.0;
}

void algorithms::randInit(){
	for(INDEXTYPE i = 0; i < graph.rows; i++){
        	for(INDEXTYPE d = 0; d < DIM; d++){
                 	//nCoordinates[i*DIM + d] = drand() - 0.5;//-1.0 + 2.0 * rand()/(RAND_MAX+1.0);
			nCoordinates[i*DIM + d] = rand()/(RAND_MAX+1.0);
		}
       	}		
}

void algorithms::randInitF(){
        for(INDEXTYPE i = 0; i < graph.rows; i++){
                for(INDEXTYPE d = 0; d < DIM; d++){
                        nCoordinates[i*DIM + d] = -1.0 + 2.0 * rand()/(RAND_MAX+1.0);
                }
        }
}

vector<VALUETYPE> algorithms::AlgoForce2VecFA(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
        INDEXTYPE LOOP = 0;
        VALUETYPE STEP = 1.0, EPS = 0.001, start, end, ENERGY, ENERGY0;
        vector<VALUETYPE> result;
        vector<INDEXTYPE> indices;
        for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
        omp_set_num_threads(NUMOFTHREADS);
        start = omp_get_wtime();
        randInitF();
	INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
	VALUETYPE *prevCoordinates;
	prevCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
	for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
		INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                	prevCoordinates[IDIM + d] = 0;
              	}
      	}
        while(LOOP < ITERATIONS){
		ENERGY0 = ENERGY;
                ENERGY = 0;
                for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
			INDEXTYPE baseindex = b * BATCHSIZE * DIM;
			#pragma omp parallel for schedule(static)
			for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                VALUETYPE forceDiff[DIM];
                                INDEXTYPE iindex = indices[i]*DIM; // i within batch, j can be far away
				#pragma forceinline
				#pragma omp simd
				for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        VALUETYPE attrc = 0;
					INDEXTYPE colidj = graph.colids[j];
					INDEXTYPE jindex = colidj*DIM;
					for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = nCoordinates[jindex + d]-nCoordinates[iindex + d];
                                                attrc += forceDiff[d] * forceDiff[d];
                                        }
					attrc = sqrt(attrc) + 1.0 / attrc;
					for(INDEXTYPE d = 0; d < this->DIM; d++){
						prevCoordinates[iindex - baseindex + d] += attrc * forceDiff[d];
						forceDiff[d] = 0;
					}
				}
				for(INDEXTYPE j = 0; j < i; j += 1){
                                        VALUETYPE repuls = 0;
					INDEXTYPE jindex = j * DIM;
                                       	for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        	forceDiff[d] = this->nCoordinates[jindex + d] - this->nCoordinates[iindex + d];
                                              	repuls += forceDiff[d] * forceDiff[d];
                                     	}
					repuls = 1.0 / repuls;
                                	for(INDEXTYPE d = 0; d < this->DIM; d++){
                                              	prevCoordinates[iindex - baseindex + d] -= repuls * forceDiff[d];
                                      	}
				}
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        VALUETYPE repuls = 0;
					INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[jindex + d] - this->nCoordinates[iindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
					repuls = 1.0 / repuls;
                                	for(INDEXTYPE d = 0; d < this->DIM; d++){
                                              	prevCoordinates[iindex - baseindex + d] -= repuls * forceDiff[d];
                                        }
				}
                        }
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) break;
				VALUETYPE factor = 0;
				INDEXTYPE iindex = indices[i] * DIM;
				for(INDEXTYPE d = 0; d < this->DIM; d++){
					factor += prevCoordinates[iindex - baseindex + d] * prevCoordinates[iindex - baseindex + d];
				}
				ENERGY += factor;
				factor = (1.0 * STEP) / sqrt(factor);
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        this->nCoordinates[iindex + d] += factor * prevCoordinates[iindex - baseindex + d];
                                	prevCoordinates[iindex - baseindex + d] = 0;
				}
                        }
                }
                STEP = STEP * 0.999;
                LOOP++;
        }
        end = omp_get_wtime();
        cout << "Force2VecFA Parallel Wall time required:" << end - start << endl;
        result.push_back(end - start);
        writeToFile("F2VFA"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP));
	return result;
}

vector<VALUETYPE> algorithms::AlgoForce2VecFR(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
        INDEXTYPE LOOP = 0;
        VALUETYPE STEP = 1.0, EPS = 0.001, start, end, ENERGY, ENERGY0;
        vector<VALUETYPE> result;
        vector<INDEXTYPE> indices;
        for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
        omp_set_num_threads(NUMOFTHREADS);
        start = omp_get_wtime();
        randInitF();
	INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
	VALUETYPE *prevCoordinates;
        prevCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
        for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
		INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                        prevCoordinates[IDIM + d] = 0;
                }
        }
        while(LOOP < ITERATIONS){
		ENERGY0 = ENERGY;
                ENERGY = 0;
                for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                        INDEXTYPE baseindex = b * BATCHSIZE * DIM;
                        #pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                VALUETYPE forceDiff[DIM];
                                INDEXTYPE iindex = indices[i]*DIM; // i within batch, j can be far away
                                #pragma forceinline
                                #pragma omp simd
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        VALUETYPE attrc = 0;
                                        INDEXTYPE colidj = graph.colids[j];
                                        INDEXTYPE jindex = colidj*DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = nCoordinates[jindex + d]-nCoordinates[iindex + d];
                                                attrc += forceDiff[d] * forceDiff[d];
                                        }
                                        attrc = attrc + 1.0 / attrc;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[iindex - baseindex + d] += attrc * forceDiff[d];
                                                forceDiff[d] = 0;
                                        }
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[jindex + d] - this->nCoordinates[iindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
                                        repuls = 1.0 / repuls;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[iindex - baseindex + d] -= repuls * forceDiff[d];
                                        }
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[jindex + d] - this->nCoordinates[iindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
                                        repuls = 1.0 / repuls;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[iindex - baseindex + d] -= repuls * forceDiff[d];
                                        }
                                }
                        }
                	for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) break;
                                VALUETYPE factor = 0;
                                INDEXTYPE iindex = indices[i] * DIM;
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        factor += prevCoordinates[iindex - baseindex + d] * prevCoordinates[iindex - baseindex + d];
                                }
                                ENERGY += factor;
                                factor = (1.0 * STEP) / sqrt(factor);
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        this->nCoordinates[iindex + d] += factor * prevCoordinates[iindex - baseindex + d];
                                        prevCoordinates[iindex - baseindex + d] = 0;
                                }
                        }
		}
                STEP = STEP * 0.999;
                LOOP++;
        }
        end = omp_get_wtime();
        cout << "Force2VecBL Parallel Wall time required:" << end - start << endl;
        result.push_back(end - start);
        writeToFile("F2VBL"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP));
        return result;
}
				
vector<VALUETYPE> algorithms::AlgoForce2VecLL(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
        INDEXTYPE LOOP = 0;
        VALUETYPE STEP = 1.0, EPS = 0.001, start, end, ENERGY, ENERGY0;
        vector<VALUETYPE> result;
        vector<INDEXTYPE> indices;
        for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
        omp_set_num_threads(NUMOFTHREADS);
        start = omp_get_wtime();
        randInitF();
        INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
	VALUETYPE *prevCoordinates;
        prevCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
        for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
		INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                        prevCoordinates[IDIM + d] = 0;
                }
        }
	while(LOOP < ITERATIONS){
                ENERGY0 = ENERGY;
                ENERGY = 0;
		for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                        INDEXTYPE baseindex = b * BATCHSIZE * DIM;
                        #pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                VALUETYPE forceDiff[DIM];
                                INDEXTYPE iindex = indices[i]*DIM; // i within batch, j can be far away
                                #pragma forceinline
                                #pragma omp simd
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        VALUETYPE attrc = 0;
                                        INDEXTYPE colidj = graph.colids[j];
                                        INDEXTYPE jindex = colidj*DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = nCoordinates[jindex + d]-nCoordinates[iindex + d];
                                                attrc += forceDiff[d] * forceDiff[d];
                                        }
                                        attrc = log2(1 + sqrt(attrc));
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[iindex - baseindex + d] += attrc * forceDiff[d];
                                                forceDiff[d] = 0;
                                        }
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[jindex + d] - this->nCoordinates[iindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
                                        repuls = 1.0 / repuls;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[iindex - baseindex + d] -= repuls * forceDiff[d];
                                        }
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[jindex + d] - this->nCoordinates[iindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
                                        repuls = 1.0 / repuls;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[iindex - baseindex + d] -= repuls * forceDiff[d];
                                        }
                                }
                        }
			for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) break;
                                VALUETYPE factor = 0;
                                INDEXTYPE iindex = indices[i] * DIM;
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        factor += prevCoordinates[iindex - baseindex + d] * prevCoordinates[iindex - baseindex + d];
                                }
                                ENERGY += factor;
                                factor = (1.0 * STEP) / sqrt(factor);
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        this->nCoordinates[iindex + d] += factor * prevCoordinates[iindex - baseindex + d];
                                        prevCoordinates[iindex - baseindex + d] = 0;
                                }
                        }
		}
                STEP = STEP * 0.999;
                LOOP++;
        }
        end = omp_get_wtime();
        cout << "Force2VecLL Parallel Wall time required:" << end - start << endl;
        result.push_back(end - start);
        writeToFile("F2VLL"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP));
        return result;
}

vector<VALUETYPE> algorithms::AlgoForce2Vec(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
	INDEXTYPE LOOP = 0;
        VALUETYPE STEP = 1.0, EPS = 0.000001;
	VALUETYPE start, end;
	double loglike0, loglike;
        vector<VALUETYPE> result;
        vector<INDEXTYPE> indices;
	INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
	VALUETYPE *prevCoordinates;
	prevCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
        for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
        omp_set_num_threads(NUMOFTHREADS);
        start = omp_get_wtime();
        randInitF();
	for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
		INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                	prevCoordinates[IDIM + d] = 0;
              	}
      	}
	loglike0 = loglike = numeric_limits<double>::max();
        while(LOOP < ITERATIONS){
		loglike0 = loglike;
		loglike = 0;
                for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
			INDEXTYPE baseindex = b * BATCHSIZE * DIM;
                	#pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
				if(i >= graph.rows) continue;
                        	VALUETYPE forceDiff[DIM];
				INDEXTYPE iindex = indices[i]*DIM; // i within batch, j can be far away
				#pragma forceinline
				#pragma omp simd
				//cout << "Vertex i:" << i << ", accessing:" << iindex << ", batchi:" << iindex - baseindex << ":";
				for(INDEXTYPE j = graph.rowptr[indices[i]]; j < graph.rowptr[indices[i]+1]; j += 1){
                                	VALUETYPE attrc = 0;
					INDEXTYPE colidj = graph.colids[j];
					INDEXTYPE jindex = colidj*DIM;
					
					for(INDEXTYPE d = 0; d < this->DIM; d++){
						forceDiff[d] = nCoordinates[iindex + d] - nCoordinates[jindex + d];
                                               	attrc += forceDiff[d] * forceDiff[d];
                                     	}
					loglike += log(1 + attrc);
					VALUETYPE d1 = -2.0 / (1.0 + attrc);
					VALUETYPE d2 = 2.0 / ((attrc) * (1.0 + attrc));
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        	forceDiff[d] = scale(forceDiff[d] * d1) - scale(forceDiff[d] * d2);
                                               	prevCoordinates[iindex - baseindex + d] += STEP * forceDiff[d];
					}
				}
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                	VALUETYPE repuls = 0;
					INDEXTYPE jindex = j * DIM;
                                       	for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        	forceDiff[d] = this->nCoordinates[iindex + d] - this->nCoordinates[jindex + d];
                                              	repuls += forceDiff[d] * forceDiff[d];
                                     	}
					VALUETYPE d1 = 2.0 / ((repuls) * (1.0 + repuls));
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        	forceDiff[d] = scale(forceDiff[d] * d1);
                                              	prevCoordinates[iindex - baseindex + d] += STEP * forceDiff[d];
                                      	}
					loglike -= log(EPS + repuls) - log(1.0 + repuls);
                           	}
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                	VALUETYPE repuls = 0;
					INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        	forceDiff[d] = this->nCoordinates[iindex + d] - this->nCoordinates[jindex + d];
                                               	repuls += forceDiff[d] * forceDiff[d];
                                       	}
					loglike -= log(EPS + repuls) - log(1.0 + repuls);
					VALUETYPE d1 = 2.0 / ((repuls) * (1.0 + repuls));
                                       	for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        	forceDiff[d] = scale(forceDiff[d] * d1);
                                              	prevCoordinates[iindex - baseindex + d] += STEP * forceDiff[d];
                                     	}
                           	}
                  	}
			#pragma omp parallel for schedule(static)
			for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                		if(i >= graph.rows) continue;
				INDEXTYPE iindex = indices[i] * DIM;
				#pragma omp simd
				for(INDEXTYPE d = 0; d < this->DIM; d++){
                        		this->nCoordinates[iindex + d] += prevCoordinates[iindex - baseindex + d];
					prevCoordinates[iindex - baseindex + d] = 0;
                      		}
             		}
    		}
                //STEP = 1.0 - (LOOP * 1.0) / ITERATIONS;
                STEP = STEP * 0.999;
		LOOP++;
		//cout << "Iteration:" << LOOP << " :LOGLIKELIHOOD:" << loglike << endl;
	}
        end = omp_get_wtime();
     	cout << "Force2Vec Parallel Wall time required:" << end - start << " seconds" << endl;
        result.push_back(end - start);
       	writeToFile("F2V"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP));
	return result;
}

vector<VALUETYPE> algorithms::AlgoForce2VecBR(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
        INDEXTYPE LOOP = 0;
        VALUETYPE STEP = 1.0, EPS = 0;
        VALUETYPE start, end;
        vector<VALUETYPE> result;
        vector<INDEXTYPE> indices;
        INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
        VALUETYPE *prevCoordinates;
        prevCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
        for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
        omp_set_num_threads(NUMOFTHREADS);
        start = omp_get_wtime();
        randInitF();
        for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
                INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                        prevCoordinates[IDIM + d] = 0;
                }
        }
        while(LOOP < ITERATIONS){
                random_shuffle(indices.begin(), indices.end());
                for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                        INDEXTYPE baseindex = b * BATCHSIZE * DIM;
                        #pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                VALUETYPE forceDiff[DIM];
                                INDEXTYPE iindex = indices[i]*DIM; // i within batch, j can be far away
                                INDEXTYPE bindex = i*DIM;
				#pragma forceinline
                                #pragma omp simd
				for(INDEXTYPE j = graph.rowptr[indices[i]]; j < graph.rowptr[indices[i]+1]; j += 1){
                                        VALUETYPE attrc = 0;
                                        INDEXTYPE colidj = graph.colids[j];
                                        INDEXTYPE jindex = colidj*DIM;

                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = nCoordinates[iindex + d] - nCoordinates[jindex + d];
                                                attrc += forceDiff[d] * forceDiff[d];
                                        }
                                        VALUETYPE d1 = -2.0 / (1.0 + attrc);
                                        VALUETYPE d2 = 2.0 / ((attrc) * (1.0 + attrc));
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = scale(forceDiff[d] * d1) - scale(forceDiff[d] * d2);
                                                prevCoordinates[bindex - baseindex + d] += STEP * forceDiff[d];
                                        }
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[iindex + d] - this->nCoordinates[jindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
                                        VALUETYPE d1 = 2.0 / ((repuls) * (1.0 + repuls));
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = scale(forceDiff[d] * d1);
                                                prevCoordinates[bindex - baseindex + d] += STEP * forceDiff[d];
                                        }
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[iindex + d] - this->nCoordinates[jindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
                                        VALUETYPE d1 = 2.0 / ((repuls) * (1.0 + repuls));
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = scale(forceDiff[d] * d1);
                                                prevCoordinates[bindex - baseindex + d] += STEP * forceDiff[d];
                                        }
                                }
                        }
                        #pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                INDEXTYPE iindex = indices[i] * DIM;
				INDEXTYPE bindex = i * DIM;
                                #pragma omp simd
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        this->nCoordinates[iindex + d] += prevCoordinates[bindex - baseindex + d];
                                        prevCoordinates[bindex - baseindex + d] = 0;
                                }
                        }
                }
		STEP = 1.0 - (LOOP * 1.0) / ITERATIONS;
                LOOP++;
        }
        end = omp_get_wtime();
        cout << "Force2VecBR Parallel Wall time required:" << end - start << " seconds" << endl;
        result.push_back(end - start);
        writeToFile("F2VBR"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP));
        return result;
}
	

INDEXTYPE randIndex(INDEXTYPE max_num, INDEXTYPE min_num){
	const INDEXTYPE newIndex = (rand() % (max_num - min_num)) + min_num;
	return newIndex;
}
vector<VALUETYPE> algorithms::AlgoForce2VecNS(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr){
        INDEXTYPE LOOP = 0;
        VALUETYPE STEP = lr, EPS = 0.00001;
        double start, end;
	double loglike0, loglike;
        vector<VALUETYPE> result;
        vector<INDEXTYPE> indices;
	vector<INDEXTYPE> nsamples;
	VALUETYPE *samples;
	samples = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
        for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
	for(INDEXTYPE i = 0; i < graph.rows; i++) nsamples.push_back(i);
        omp_set_num_threads(NUMOFTHREADS);
        start = omp_get_wtime();
        randInitF();
        INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
        VALUETYPE *prevCoordinates;
        prevCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
	for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
		INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                        prevCoordinates[IDIM + d] = 0;
                }
        }
	loglike0 = loglike = numeric_limits<double>::max();
        while(LOOP < ITERATIONS){
		loglike0 = loglike;
		loglike = 0;
		//random_shuffle(indices.begin(), indices.end());
		//random_shuffle(nsamples.begin(), nsamples.end());
                for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                        INDEXTYPE baseindex = b * BATCHSIZE * DIM;
			//cout << "batch:"<< b <<":";
			for(INDEXTYPE s = 0; s < ns; s++){
				int rindex = randIndex(graph.rows-1, 0);
				//int rindex = nsamples[minv + s];
				int randombaseindex = rindex * DIM;
				int sindex = s * DIM;
				for(INDEXTYPE d = 0; d < DIM; d++){
					samples[sindex + d] = nCoordinates[randombaseindex + d];
				}
				//cout << rindex << ",";
			}
			//cout << endl;
                        #pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                VALUETYPE forceDiff[DIM];
                                //INDEXTYPE iindex = indices[i]*DIM; // i within batch, j can be far away
                                INDEXTYPE iindex = i*DIM;
				INDEXTYPE bindex = i * DIM;
				INDEXTYPE degree = graph.rowptr[i+1] - graph.rowptr[i];
				#pragma forceinline
                                #pragma omp simd
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        VALUETYPE attrc = 0;
                                        INDEXTYPE colidj = graph.colids[j];
                                        INDEXTYPE jindex = colidj*DIM;

                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = nCoordinates[iindex + d] - nCoordinates[jindex + d];
                                                attrc += forceDiff[d] * forceDiff[d];
                                        }
					loglike += log(1 + attrc);
                                        VALUETYPE d1 = -2.0 / (1.0 + attrc);
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = scale(forceDiff[d] * d1);
                                                prevCoordinates[bindex - baseindex + d] += (STEP) * forceDiff[d];
                                        }
                                }
                                for(INDEXTYPE j = 0; j < ns; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = this->nCoordinates[iindex + d] - samples[jindex + d];
                                                repuls += forceDiff[d] * forceDiff[d];
                                        }
					loglike -= log(0.000001+repuls) - log(1+repuls);
					VALUETYPE d1 = 2.0 / ((repuls) * (1.0 + repuls));
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                forceDiff[d] = scale(forceDiff[d] * d1);
                                                prevCoordinates[bindex - baseindex + d] += (STEP) * forceDiff[d];
                                        }
                                }
                        }
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) break;
                                //INDEXTYPE iindex = indices[i] * DIM;
                                INDEXTYPE iindex = i * DIM;
				INDEXTYPE bindex = i * DIM;
				#pragma omp simd
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        this->nCoordinates[iindex + d] += prevCoordinates[bindex - baseindex + d];
					prevCoordinates[bindex - baseindex + d] = 0;
                                }
                        }
                }
                //STEP = 1.0 - (LOOP * 1.0) / ITERATIONS;
                //STEP = 0.999 * STEP;
		LOOP++;
		//if(LOOP % 8 == 0)
		//	cout << "Iteration:" << LOOP << " :LOGLIKELIHOOD: " << loglike << endl;
        }
        end = omp_get_wtime();
        cout << "Force2Vec Parallel Wall time required:" << end - start << " seconds" << endl;
        result.push_back(end - start);
        writeToFile("F2VNS"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        return result;
}

VALUETYPE *sm_table;

void init_SM_TABLE(){
        VALUETYPE x;
        sm_table = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[SM_TABLE_SIZE])));
        for(INDEXTYPE i = 0; i < SM_TABLE_SIZE; i++){
                x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND;
                sm_table[i] = 1.0 / (1 + exp(-x));
        }
}

VALUETYPE fast_SM(VALUETYPE v){
        if(v > SM_BOUND) return 1.0;
        else if(v < - SM_BOUND) return 0.0;
        return sm_table[(int)((v + SM_BOUND) * SM_RESOLUTION)];
}

VALUETYPE coscale(VALUETYPE v){
	if(v > 1.0f) return 1.0f;
        else if(v < - 1.0f) return -1.0f;
        return v;
}

vector<VALUETYPE> algorithms::AlgoForce2VecNSRW(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr){
        INDEXTYPE LOOP = 0;
        VALUETYPE STEP = lr, EPS = 0.00001;
        double start, end;
        double loglike0, loglike;
        vector<VALUETYPE> result;
        vector<INDEXTYPE> indices;
        vector<INDEXTYPE> nsamples;
        VALUETYPE *samples;
        samples = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
        unsigned long long x = time(nullptr);
  	for (int i = 0; i < 2; i++) {
    		unsigned long long z = x += UINT64_C(0x9E3779B97F4A7C15);
    		z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    		z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    		rseed[i] = z ^ z >> 31;
  	}
	//for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
        //for(INDEXTYPE i = 0; i < graph.rows; i++) nsamples.push_back(i);
        init_SM_TABLE();
	omp_set_num_threads(NUMOFTHREADS);
        start = omp_get_wtime();
        randInit();
        INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
        VALUETYPE *prevCoordinates;
	int maxdegree = -1;
        prevCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
        loglike0 = loglike = numeric_limits<double>::max();
        while(LOOP < ITERATIONS){
                loglike0 = loglike;
                loglike = 0;
		for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                        INDEXTYPE baseindex = b * BATCHSIZE * DIM;
			#pragma omp simd
			for(INDEXTYPE s = 0; s < ns; s++){
                                int minv = 0;//b * BATCHSIZE;
                                int maxv = min((b+1) * BATCHSIZE, graph.rows-1);
                                //int maxv = graph.rows-1;
				int rindex = randIndex(maxv, minv);
				int randombaseindex = rindex * DIM;
                                int sindex = s * DIM;
                                for(INDEXTYPE d = 0; d < DIM; d++){
                                        samples[sindex + d] = nCoordinates[randombaseindex + d];
                                }
			}
			INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
			for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
				if(i >= graph.rows) continue;
                		INDEXTYPE IDIM = i*DIM;
                		#pragma omp simd
				for(INDEXTYPE d = 0; d < this->DIM; d++){
                        		prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
                		}
        		}
			//printf("Batch:%d, baseindex=%d\n", b, baseindex);	
			#pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                VALUETYPE forceDiff[DIM];
				INDEXTYPE iindex = i * DIM;
                                INDEXTYPE bindex = i * DIM;
				INDEXTYPE wlength = 0;
				INDEXTYPE windex = i;
				

				/*while(wlength < 5){
					INDEXTYPE j;
                                        if(graph.rowptr[windex+1] - graph.rowptr[windex] > 2)
                                                j = randIndex(graph.rowptr[windex+1]-1, graph.rowptr[windex]);
                                        else if(graph.rowptr[windex+1] - graph.rowptr[windex] == 2)
                                                j = graph.rowptr[windex];
                                */
				VALUETYPE degi = 1.0;
				//if(graph.rowptr[i+1] - graph.rowptr[i] > 150)
				degi = 1.0 / (graph.rowptr[i+1] - graph.rowptr[i] + 1);
				//printf("Max degree of %d:%d\n", i, graph.rowptr[i+1] - graph.rowptr[i]);
				for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                	//cout << "Here:" << LOOP << ", max:" << graph.rowptr[windex+1]-1 << ",min" << graph.rowptr[windex] << endl;
					VALUETYPE attrc = 0;
                                        INDEXTYPE colidj = graph.colids[j];
                                        INDEXTYPE jindex = colidj*DIM;
 
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                attrc += nCoordinates[iindex + d] * nCoordinates[jindex + d];
                                                //attrc += forceDiff[d] * forceDiff[d];
                                        }
                                        VALUETYPE d1 = fast_SM(attrc);
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[bindex - baseindex + d] += STEP * degi * (1.0 - d1) * nCoordinates[jindex + d];
                                        	//prevCoordinates[bindex - baseindex + d] = coscale(prevCoordinates[bindex - baseindex + d]);
					}
#if 0
    					//printf("Loop = %d, Node = %d, Nei: %d\n", LOOP, i, colidj);
    					VALUETYPE degj = 1.0;
					//if(graph.rowptr[j+1] - graph.rowptr[j] > 150)
					degj = 1.0 / (graph.rowptr[colidj+1] - graph.rowptr[colidj] + 1);
	                               	wlength = 0;
					//while(wlength < 5 && wlength < graph.rowptr[colidj+1]-1-graph.rowptr[colidj]){
					//	INDEXTYPE k = randIndex(graph.rowptr[colidj+1]-1, graph.rowptr[colidj]);
					
					for(INDEXTYPE k = graph.rowptr[colidj]; k < graph.rowptr[colidj+1]; k += 1){
						attrc = 0;
                                                INDEXTYPE colidk = graph.colids[k];
                                                INDEXTYPE kindex = colidk*DIM;
                                                
                                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                        attrc += nCoordinates[iindex + d] * nCoordinates[kindex + d];
                                                }
                                                d1 = fast_SM(attrc);
                                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                        prevCoordinates[bindex - baseindex + d] += STEP * degi * degj * (1.0 - d1) * nCoordinates[kindex + d];
							//prevCoordinates[bindex - baseindex + d] = coscale(prevCoordinates[bindex - baseindex + d]);
						}
						wlength++;
                                        }
#endif
	
					wlength++;
					windex = colidj;
                                }
                                for(INDEXTYPE j = 0; j < ns; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
						repuls += this->nCoordinates[iindex + d] * samples[jindex + d];
						//repuls += forceDiff[d] * forceDiff[d];
                                        }
                                        VALUETYPE d1 = fast_SM(repuls);
					
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[bindex - baseindex + d] -= (STEP) * d1 * samples[jindex + d];
						//prevCoordinates[bindex - baseindex + d] = coscale(prevCoordinates[bindex - baseindex + d]);
                                        }
                                }
                        }
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
				INDEXTYPE iindex = i * DIM;
                                INDEXTYPE bindex = i * DIM;
                                #pragma omp simd
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        this->nCoordinates[iindex + d] = prevCoordinates[bindex - baseindex + d];
                                }
                        }
                }
		//cout << "Iteration:" << LOOP << endl; // << ",maxdeg:" << maxdegree<< endl;
		//STEP = 1.0 - (LOOP * 1.0) / ITERATIONS;		
		LOOP++;
	}
        end = omp_get_wtime();
        cout << "Force2Vec Parallel Wall time required:" << end - start << " seconds" << endl;
        result.push_back((float)(end - start));
        writeToFile("F2VWNS"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        return result;
}

vector<VALUETYPE> algorithms::AlgoForce2VecNSRWEFF(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE ns, VALUETYPE lr){
        INDEXTYPE LOOP = 0;
        VALUETYPE STEP = lr, EPS = 0.00001;
        double start, end;
        double loglike0, loglike;
        vector<VALUETYPE> result;
        vector<INDEXTYPE> indices;
        vector<INDEXTYPE> nsamples;
        VALUETYPE *samples;
	INDEXTYPE *walksamples;
	INDEXTYPE WALKLENGTH = 5;
	samples = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
	walksamples = static_cast<INDEXTYPE *> (::operator new (sizeof(INDEXTYPE[WALKLENGTH * graph.rows])));
        unsigned long long x = time(nullptr);
        for (int i = 0; i < 2; i++) {
                unsigned long long z = x += UINT64_C(0x9E3779B97F4A7C15);
                z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
                z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
                rseed[i] = z ^ z >> 31;
        }
	init_SM_TABLE();
        omp_set_num_threads(NUMOFTHREADS);
        start = omp_get_wtime();
        randInit();
        INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
        VALUETYPE *prevCoordinates;
        int maxdegree = -1;
        prevCoordinates = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
        loglike0 = loglike = numeric_limits<double>::max();
	while(LOOP < ITERATIONS){
                //break;
		loglike0 = loglike;
                loglike = 0;
		//#pragma omp parallel for schedule(static)
		for(INDEXTYPE i = 0; i < graph.rows; i++){
                	INDEXTYPE wlength = 0;
                	INDEXTYPE windex = i;
			while(wlength < WALKLENGTH){
                        	INDEXTYPE j=windex;
				if(graph.rowptr[windex+1] - graph.rowptr[windex] > 2){
                                	j = randIndex(graph.rowptr[windex+1]-1, graph.rowptr[windex]);
					//cout << "rand->j:" << j << endl;
				}
                        	else if(graph.rowptr[windex+1] - graph.rowptr[windex] == 2){
                                	j = graph.rowptr[windex];
					//cout << "fixed->j:" << j << endl;
				}
				//j = (j != -1)? j:i;
				//printf("(j = %d, %d)", j, graph.colids[j]);
                        	walksamples[i*WALKLENGTH + wlength] = graph.colids[j];
                        	wlength++;
                        	windex = graph.colids[j];
                	}
			//printf("\n");
			
        	}
		//printf("Walk generation: walk: %d\n", walksamples[0]);
                for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                        INDEXTYPE baseindex = b * BATCHSIZE * DIM;
                        #pragma omp simd
                        for(INDEXTYPE s = 0; s < ns; s++){
                                int minv = 0;//b * BATCHSIZE;
                                int maxv = min((b+1) * BATCHSIZE, graph.rows-1);
                                int rindex = randIndex(maxv, minv);
                                int randombaseindex = rindex * DIM;
                                int sindex = s * DIM;
                                for(INDEXTYPE d = 0; d < DIM; d++){
                                        samples[sindex + d] = nCoordinates[randombaseindex + d];
                                }
                        }
                        INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                INDEXTYPE IDIM = i*DIM;
                                #pragma omp simd
				for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
                                }
                        }
                        #pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
				//printf("Check..., nCoordinates %d degree: %d\n", i, graph.rowptr[i+1] - graph.rowptr[i]);
                          	//if(i >= 1225) 
				//print();
				VALUETYPE forceDiff[DIM];
                           
				INDEXTYPE iindex = i * DIM;
                                INDEXTYPE bindex = i * DIM;
                                INDEXTYPE wlength = 0;
                                INDEXTYPE windex = i;
                                for(INDEXTYPE j = 0; j < WALKLENGTH; j++){
                                	VALUETYPE degi = 1.0;
					VALUETYPE attrc = 0;
                                        INDEXTYPE colidj = walksamples[i * WALKLENGTH + j];
                                        INDEXTYPE jindex = colidj*DIM;
					degi = 1.0 / (graph.rowptr[i+1] - graph.rowptr[i] + 1);
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                attrc += nCoordinates[iindex + d] * nCoordinates[jindex + d];
					}
					VALUETYPE d1 = fast_SM(attrc);
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[bindex - baseindex + d] += STEP * degi * (1.0 - d1) * nCoordinates[jindex + d];
					}
					//if(i>=1025)
					//for(int d = 0; d<DIM; d++){
					//	printf("%d ", prevCoordinates[bindex - baseindex + d]);
					//}
                                }
                                for(INDEXTYPE j = 0; j < ns; j += 1){
                                        VALUETYPE repuls = 0;
                                        INDEXTYPE jindex = j * DIM;
                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                repuls += this->nCoordinates[iindex + d] * samples[jindex + d];
					}
                                        VALUETYPE d1 = fast_SM(repuls);

                                        for(INDEXTYPE d = 0; d < this->DIM; d++){
                                                prevCoordinates[bindex - baseindex + d] -= (STEP) * d1 * samples[jindex + d];
					}
                                }
                        }
                        for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                if(i >= graph.rows) continue;
                                INDEXTYPE iindex = i * DIM;
                                INDEXTYPE bindex = i * DIM;
                                #pragma omp simd
                                for(INDEXTYPE d = 0; d < this->DIM; d++){
                                        this->nCoordinates[iindex + d] = prevCoordinates[bindex - baseindex + d];
                                }
                        }
                }
		//cout << "Iteration:" << LOOP << ",sample:" << this->nCoordinates[0] << endl;
		LOOP++;
        }
        end = omp_get_wtime();
        cout << "Force2VecWNSEFF Parallel Wall time required:" << end - start << " seconds" << endl;
        result.push_back((float)(end - start));
        writeToFile("F2VWNSF"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        return result;
}



/*
 * Intrinsic implementation with register blocking:
 *    SREAL -> VALUETYPE = float 
 *    D128  -> D=128
 *    AVXZ  -> AVX512 architecture
 * We will generate all combination using a generator later 
 */

/*
 * void PrintVector(char*name, VTYPE v)
{
   int i;
#ifdef DREAL 
   double *fptr = (double*) &v;
#else
   float *fptr = (float*) &v;
#endif
   fprintf(stdout, "vector-%s: < ", name);
   for (i=0; i < VLEN; i++)
      fprintf(stdout, "%lf, ", fptr[i]);
   fprintf(stdout, ">\n");
}
*/
#ifdef AVX512

vector<VALUETYPE> algorithms::AlgoForce2VecNS_SREAL_D128_AVXZ
(
 INDEXTYPE ITERATIONS, 
 INDEXTYPE NUMOFTHREADS, 
 INDEXTYPE BATCHSIZE, 
 INDEXTYPE ns, 
 VALUETYPE lr
 )
{
   INDEXTYPE LOOP = 0;
   VALUETYPE STEP = lr, EPS = 0.00001;
   double start, end;
   double loglike0, loglike;
   
   vector<VALUETYPE> result;
   vector<INDEXTYPE> indices;
   vector<INDEXTYPE> nsamples;
   VTYPE VMAXBOUND, VMINBOUND, VEPS, VSTEP;   
   VALUETYPE *samples;
   samples = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
   
   omp_set_num_threads(NUMOFTHREADS);
   start = omp_get_wtime();
   randInitF();
   INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
   VALUETYPE *prevCoordinates;
   int maxdegree = -1;
        
   prevCoordinates = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));

   for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
		INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                        prevCoordinates[IDIM + d] = 0;
                }
   }

   BCL_vset1(VMAXBOUND, MAXBOUND); 
   BCL_vset1(VMINBOUND, -MAXBOUND); 

   loglike0 = loglike = numeric_limits<double>::max();
   while(LOOP < ITERATIONS)
   {
      loglike0 = loglike;
      loglike = 0;
      for(INDEXTYPE b = 0; b < graph.rows / BATCHSIZE; b += 1)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
	    int maxv = graph.rows-1;
            int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
         INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
          #pragma omp parallel for schedule(dynamic)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            VALUETYPE *X = nCoordinates; 
            VALUETYPE *Y = prevCoordinates; 
            VALUETYPE *S = samples; 

            INDEXTYPE iindex = i * DIM;
            VALUETYPE *Xi = X + iindex; 
            INDEXTYPE bindex = i * DIM;
            VALUETYPE degi = 1.0;

            VTYPE Vy0, Vy1, Vy2, Vy3, Vy4, Vy5, Vy6, Vy7; 
            VTYPE Vxi0, Vxi1, Vxi2, Vxi3, Vxi4, Vxi5, Vxi6, Vxi7; 
            BCL_vldu(Vy0, Y + bindex - baseindex +VLEN*0);
            BCL_vldu(Vy1, Y + bindex - baseindex +VLEN*1);
            BCL_vldu(Vy2, Y + bindex - baseindex +VLEN*2);
            BCL_vldu(Vy3, Y + bindex - baseindex +VLEN*3);
            BCL_vldu(Vy4, Y + bindex - baseindex +VLEN*4);
            BCL_vldu(Vy5, Y + bindex - baseindex +VLEN*5);
            BCL_vldu(Vy6, Y + bindex - baseindex +VLEN*6);
            BCL_vldu(Vy7, Y + bindex - baseindex +VLEN*7);
            
            BCL_vldu(Vxi0, Xi + VLEN*0);
            BCL_vldu(Vxi1, Xi + VLEN*1);
            BCL_vldu(Vxi2, Xi + VLEN*2);
            BCL_vldu(Vxi3, Xi + VLEN*3);
            BCL_vldu(Vxi4, Xi + VLEN*4);
            BCL_vldu(Vxi5, Xi + VLEN*5);
            BCL_vldu(Vxi6, Xi + VLEN*6);
            BCL_vldu(Vxi7, Xi + VLEN*7);
	
	    VTYPE Vt, Vd0, Vd1, Vd2, Vd3, Vd4, Vd5, Vd6, Vd7;

            for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
               VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = X + jindex; 
        
               VTYPE Vtemp, Vatt0, Vatt1, Vatt2, Vatt3;
               VTYPE Vxj0, Vxj1, Vxj2, Vxj3, Vxj4, Vxj5, Vxj6, Vxj7; 

               BCL_vldu(Vxj0, Xj + VLEN*0);
               BCL_vldu(Vxj1, Xj + VLEN*1);
               BCL_vldu(Vxj2, Xj + VLEN*2);
               BCL_vldu(Vxj3, Xj + VLEN*3);
               BCL_vldu(Vxj4, Xj + VLEN*4);
               BCL_vldu(Vxj5, Xj + VLEN*5);
               BCL_vldu(Vxj6, Xj + VLEN*6);
               BCL_vldu(Vxj7, Xj + VLEN*7);
               
	       BCL_vzero(Vatt0); 
	       BCL_vzero(Vatt1);
	       BCL_vzero(Vatt2);
	       BCL_vzero(Vatt3);
         
               BCL_vsub(Vd0, Vxi0, Vxj0);
               BCL_vsub(Vd1, Vxi1, Vxj1);
               BCL_vsub(Vd2, Vxi2, Vxj2);
               BCL_vsub(Vd3, Vxi3, Vxj3);
               BCL_vsub(Vd4, Vxi4, Vxj4);
               BCL_vsub(Vd5, Vxi5, Vxj5);
               BCL_vsub(Vd6, Vxi6, Vxj6);
               BCL_vsub(Vd7, Vxi7, Vxj7);

	       BCL_vmac(Vatt0, Vd0, Vd0);
               BCL_vmac(Vatt1, Vd1, Vd1);
               BCL_vmac(Vatt2, Vd2, Vd2);
               BCL_vmac(Vatt3, Vd3, Vd3);
               BCL_vmac(Vatt0, Vd4, Vd4);
               BCL_vmac(Vatt1, Vd5, Vd5);
               BCL_vmac(Vatt2, Vd6, Vd6);
               BCL_vmac(Vatt3, Vd7, Vd7);

               BCL_vadd(Vatt0, Vatt0, Vatt1);
               BCL_vadd(Vatt2, Vatt2, Vatt3);
               BCL_vadd(Vatt0, Vatt0, Vatt2);
         
               BCL_vrsum1(attrc, Vatt0);

	       BCL_vset1(Vatt0, attrc); // a = a
	       BCL_vset1(Vt, 1.0f); // t = 1
	       BCL_vadd(Vatt0, Vatt0, Vt); // a = t + a

	       BCL_vrcp(Vatt0, Vatt0); // a = 1/a
	       BCL_vset1(Vt, -2.0f); // t = -2
	       BCL_vmul(Vatt0, Vatt0, Vt); // a = -2 * a
               
               BCL_vmul(Vd0, Vatt0, Vd0); 
	       BCL_vmul(Vd1, Vatt0, Vd1);  
	       BCL_vmul(Vd2, Vatt0, Vd2);  
 	       BCL_vmul(Vd3, Vatt0, Vd3);  
 	       BCL_vmul(Vd4, Vatt0, Vd4);  
               BCL_vmul(Vd5, Vatt0, Vd5);
               BCL_vmul(Vd6, Vatt0, Vd6);
               BCL_vmul(Vd7, Vatt0, Vd7);               

	       BCL_vmax(Vd0, Vd0, VMINBOUND);
	       BCL_vmax(Vd1, Vd1, VMINBOUND); 
	       BCL_vmax(Vd2, Vd2, VMINBOUND);  
               BCL_vmax(Vd3, Vd3, VMINBOUND);
               BCL_vmax(Vd4, Vd4, VMINBOUND);  
               BCL_vmax(Vd5, Vd5, VMINBOUND);
               BCL_vmax(Vd6, Vd6, VMINBOUND);
               BCL_vmax(Vd7, Vd7, VMINBOUND);

	       BCL_vmin(Vd0, Vd0, VMAXBOUND);
               BCL_vmin(Vd1, Vd1, VMAXBOUND);
               BCL_vmin(Vd2, Vd2, VMAXBOUND);
               BCL_vmin(Vd3, Vd3, VMAXBOUND);
               BCL_vmin(Vd4, Vd4, VMAXBOUND);
               BCL_vmin(Vd5, Vd5, VMAXBOUND);
               BCL_vmin(Vd6, Vd6, VMAXBOUND);
               BCL_vmin(Vd7, Vd7, VMAXBOUND);
	
	       BCL_vset1(Vt, STEP);
	     
               BCL_vmac(Vy0, Vd0, Vt); 
               BCL_vmac(Vy1, Vd1, Vt); 
               BCL_vmac(Vy2, Vd2, Vt); 
               BCL_vmac(Vy3, Vd3, Vt); 
               BCL_vmac(Vy4, Vd4, Vt); 
               BCL_vmac(Vy5, Vd5, Vt); 
               BCL_vmac(Vy6, Vd6, Vt); 
               BCL_vmac(Vy7, Vd7, Vt); 
            }
	
	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               VTYPE Vs0, Vs1, Vs2, Vs3, Vs4, Vs5, Vs6, Vs7;
               VTYPE Vd1;
               VTYPE Vrep0, Vrep1, Vrep2, Vrep3;  
      
               BCL_vldu(Vs0, S + jindex + VLEN*0);
               BCL_vldu(Vs1, S + jindex + VLEN*1);
               BCL_vldu(Vs2, S + jindex + VLEN*2);
               BCL_vldu(Vs3, S + jindex + VLEN*3);
               BCL_vldu(Vs4, S + jindex + VLEN*4);
               BCL_vldu(Vs5, S + jindex + VLEN*5);
               BCL_vldu(Vs6, S + jindex + VLEN*6);
               BCL_vldu(Vs7, S + jindex + VLEN*7);

	       BCL_vzero(Vrep0); 
               BCL_vzero(Vrep1); 
               BCL_vzero(Vrep2); 
               BCL_vzero(Vrep3); 

	       BCL_vsub(Vd0, Vxi0, Vs0);
               BCL_vsub(Vd1, Vxi1, Vs1);
               BCL_vsub(Vd2, Vxi2, Vs2);
               BCL_vsub(Vd3, Vxi3, Vs3);
               BCL_vsub(Vd4, Vxi4, Vs4);
               BCL_vsub(Vd5, Vxi5, Vs5);
               BCL_vsub(Vd6, Vxi6, Vs6);
               BCL_vsub(Vd7, Vxi7, Vs7);

               BCL_vmac(Vrep0, Vd0, Vd0);
               BCL_vmac(Vrep1, Vd1, Vd1);
               BCL_vmac(Vrep2, Vd2, Vd2);
               BCL_vmac(Vrep3, Vd3, Vd3);
               BCL_vmac(Vrep0, Vd4, Vd4);
               BCL_vmac(Vrep1, Vd5, Vd5);
               BCL_vmac(Vrep2, Vd6, Vd6);
               BCL_vmac(Vrep3, Vd7, Vd7);

               BCL_vadd(Vrep0, Vrep0, Vrep1);
               BCL_vadd(Vrep2, Vrep2, Vrep3);
               BCL_vadd(Vrep0, Vrep0, Vrep2);

               BCL_vrsum1(repuls, Vrep0);
	       
	       BCL_vset1(Vrep0, repuls); // a = a
               BCL_vset1(Vt, 1.0f); // t = 1
               BCL_vadd(Vt, Vrep0, Vt); // t = t + a
	       BCL_vmul(Vrep0, Vrep0, Vt); // a = t * a

               BCL_vrcp(Vrep0, Vrep0); // a = 1/a
               BCL_vset1(Vt, 2.0f); // t = 2
               BCL_vmul(Vrep0, Vrep0, Vt); // a = 2 * a

               BCL_vmul(Vd0, Vrep0, Vd0);
               BCL_vmul(Vd1, Vrep0, Vd1);
               BCL_vmul(Vd2, Vrep0, Vd2);
               BCL_vmul(Vd3, Vrep0, Vd3);
               BCL_vmul(Vd4, Vrep0, Vd4);
               BCL_vmul(Vd5, Vrep0, Vd5);
               BCL_vmul(Vd6, Vrep0, Vd6);
               BCL_vmul(Vd7, Vrep0, Vd7);

               BCL_vmax(Vd0, Vd0, VMINBOUND);
               BCL_vmax(Vd1, Vd1, VMINBOUND);
               BCL_vmax(Vd2, Vd2, VMINBOUND);
               BCL_vmax(Vd3, Vd3, VMINBOUND);
               BCL_vmax(Vd4, Vd4, VMINBOUND);
               BCL_vmax(Vd5, Vd5, VMINBOUND);
               BCL_vmax(Vd6, Vd6, VMINBOUND);
               BCL_vmax(Vd7, Vd7, VMINBOUND);

               BCL_vmin(Vd0, Vd0, VMAXBOUND);
               BCL_vmin(Vd1, Vd1, VMAXBOUND);
               BCL_vmin(Vd2, Vd2, VMAXBOUND);
               BCL_vmin(Vd3, Vd3, VMAXBOUND);
               BCL_vmin(Vd4, Vd4, VMAXBOUND);
               BCL_vmin(Vd5, Vd5, VMAXBOUND);
               BCL_vmin(Vd6, Vd6, VMAXBOUND);
               BCL_vmin(Vd7, Vd7, VMAXBOUND);

               BCL_vset1(Vt, STEP);
	     
               BCL_vmac(Vy0, Vd0, Vt);
               BCL_vmac(Vy1, Vd1, Vt);
               BCL_vmac(Vy2, Vd2, Vt);
               BCL_vmac(Vy3, Vd3, Vt);
               BCL_vmac(Vy4, Vd4, Vt);
               BCL_vmac(Vy5, Vd5, Vt);
               BCL_vmac(Vy6, Vd6, Vt);
               BCL_vmac(Vy7, Vd7, Vt);


            }
	    BCL_vstu(Y + bindex - baseindex +VLEN*0, Vy0);
            BCL_vstu(Y + bindex - baseindex +VLEN*1, Vy1);
            BCL_vstu(Y + bindex - baseindex +VLEN*2, Vy2);
            BCL_vstu(Y + bindex - baseindex +VLEN*3, Vy3);
            BCL_vstu(Y + bindex - baseindex +VLEN*4, Vy4);
            BCL_vstu(Y + bindex - baseindex +VLEN*5, Vy5);
            BCL_vstu(Y + bindex - baseindex +VLEN*6, Vy6);
            BCL_vstu(Y + bindex - baseindex +VLEN*7, Vy7);
         }
         
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] += 
                  prevCoordinates[bindex - baseindex + d];
		prevCoordinates[bindex - baseindex + d] = 0;
            }
         }
      }
	INDEXTYPE b = (graph.rows/BATCHSIZE); // can be skipped 
	if (graph.rows > b * BATCHSIZE)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;
	    int maxv = graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
	 INDEXTYPE INITPREVSIZE = graph.rows;
	 #pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    VALUETYPE forceDiff[DIM];
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
	    INDEXTYPE wlength = 0;
	    INDEXTYPE windex = i;
            VALUETYPE degi = 1.0;
	
            VALUETYPE *Xi = nCoordinates + iindex; 
  	    for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
	       VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = nCoordinates + jindex; 
	       VALUETYPE forceDiff[DIM];
 
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
			forceDiff[d] = Xj[d] - Xi[d];
			attrc += forceDiff[d] * forceDiff[d];
		}
	       VALUETYPE d1 = -2.0 / (1.0 + attrc);
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
			forceDiff[d] = scale(forceDiff[d] * d1);
			prevCoordinates[bindex - baseindex + d] += STEP * forceDiff[d];
		 }
            }
	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
	           forceDiff[d] = this->nCoordinates[iindex + d] - samples[jindex + d];
		  repuls += forceDiff[d] * forceDiff[d];
               }
               VALUETYPE d1 = 2.0 / ((repuls) * (1.0 + repuls));
					
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
		 	forceDiff[d] = scale(forceDiff[d] * d1);
                  	prevCoordinates[bindex - baseindex + d] +=  (STEP) * forceDiff[d];
               }
            }
         }
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] += prevCoordinates[bindex - baseindex + d];
            	prevCoordinates[bindex - baseindex + d] = 0;
	    }
         }
      }


      LOOP++;
   }
	
   end = omp_get_wtime();
   cout << "optForce2VecNS_AVXZ Parallel Wall time required:" << end - start << " seconds" << endl;
   
   result.push_back((float)(end - start));
   writeToFile("F2VNS_AVXZ"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        
   return result;
		
}

vector<VALUETYPE> algorithms::AlgoForce2VecNSRW_SREAL_D128_AVXZ
(
 INDEXTYPE ITERATIONS, 
 INDEXTYPE NUMOFTHREADS, 
 INDEXTYPE BATCHSIZE, 
 INDEXTYPE ns, 
 VALUETYPE lr
 )
{
   INDEXTYPE LOOP = 0;
   VALUETYPE STEP = lr, EPS = 0.00001;
   double start, end;
   double loglike0, loglike;
   
   vector<VALUETYPE> result;
   vector<INDEXTYPE> indices;
   vector<INDEXTYPE> nsamples;
   
   VALUETYPE *samples;
   samples = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
   
   unsigned long long x = time(nullptr);
   for (int i = 0; i < 2; i++) 
   {
      unsigned long long z = x += UINT64_C(0x9E3779B97F4A7C15);
      z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
      z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
      rseed[i] = z ^ z >> 31;
   }
	
   init_SM_TABLE();
   omp_set_num_threads(NUMOFTHREADS);
   start = omp_get_wtime();
   randInit();
   INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
   VALUETYPE *prevCoordinates;
   int maxdegree = -1;
        
   prevCoordinates = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
   
   loglike0 = loglike = numeric_limits<double>::max();
   
   while(LOOP < ITERATIONS)
   {
      loglike0 = loglike;
      loglike = 0;
      
      // considering cleanup seperately...  
      //for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1)
      // handle iteration which is multiple of BATCHSIZE 
      for(INDEXTYPE b = 0; b < graph.rows / BATCHSIZE; b += 1)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
            //int maxv = min((b+1) * BATCHSIZE, graph.rows-1);
            int maxv = (b+1) * BATCHSIZE;//graph.rows-1;
            //int maxv = graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
	 //INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
	 INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            INDEXTYPE IDIM = i*DIM;
            #pragma omp simd
	    for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
            }
         }
	 //printf("Batch:%d, baseindex=%d\n", b, baseindex);	
	 #pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            // renaming some pointers 
            VALUETYPE *X = nCoordinates; 
            VALUETYPE *Y = prevCoordinates; 
            VALUETYPE *S = samples; 

            INDEXTYPE iindex = i * DIM;
            VALUETYPE *Xi = X + iindex; 
            INDEXTYPE bindex = i * DIM;
            VALUETYPE degi = 1.0;
	
            degi = 1.0 / (graph.rowptr[i+1] - graph.rowptr[i] + 1);

            VTYPE Vy0, Vy1, Vy2, Vy3, Vy4, Vy5, Vy6, Vy7, Vy8; 
            VTYPE Vxi0, Vxi1, Vxi2, Vxi3, Vxi4, Vxi5, Vxi6, Vxi7, Vxi8; 
      
            // load Y 
            // FIXME: make sure to malloc aligned memory ... use vld then!!! 
            BCL_vldu(Vy0, Y + bindex - baseindex +VLEN*0);
            BCL_vldu(Vy1, Y + bindex - baseindex +VLEN*1);
            BCL_vldu(Vy2, Y + bindex - baseindex +VLEN*2);
            BCL_vldu(Vy3, Y + bindex - baseindex +VLEN*3);
            BCL_vldu(Vy4, Y + bindex - baseindex +VLEN*4);
            BCL_vldu(Vy5, Y + bindex - baseindex +VLEN*5);
            BCL_vldu(Vy6, Y + bindex - baseindex +VLEN*6);
            BCL_vldu(Vy7, Y + bindex - baseindex +VLEN*7);

            // load Xi and Y 
            BCL_vldu(Vxi0, Xi + VLEN*0);
            BCL_vldu(Vxi1, Xi + VLEN*1);
            BCL_vldu(Vxi2, Xi + VLEN*2);
            BCL_vldu(Vxi3, Xi + VLEN*3);
            BCL_vldu(Vxi4, Xi + VLEN*4);
            BCL_vldu(Vxi5, Xi + VLEN*5);
            BCL_vldu(Vxi6, Xi + VLEN*6);
            BCL_vldu(Vxi7, Xi + VLEN*7);
/*
 *          attractive force calc: spase-dense computation 
 */
            for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
               VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = X + jindex; 
        
               VTYPE Vd1;
               VTYPE Vatt0, Vatt1, Vatt2, Vatt3;  
               VTYPE Vxj0, Vxj1, Vxj2, Vxj3, Vxj4, Vxj5, Vxj6, Vxj7; 
               
               // load Xj 
               BCL_vldu(Vxj0, Xj + VLEN*0);
               BCL_vldu(Vxj1, Xj + VLEN*1);
               BCL_vldu(Vxj2, Xj + VLEN*2);
               BCL_vldu(Vxj3, Xj + VLEN*3);
               BCL_vldu(Vxj4, Xj + VLEN*4);
               BCL_vldu(Vxj5, Xj + VLEN*5);
               BCL_vldu(Vxj6, Xj + VLEN*6);
               BCL_vldu(Vxj7, Xj + VLEN*7);
               #if 0 
                  for(INDEXTYPE d = 0; d < this->DIM; d++)
                  {
                     attrc += Xi[d] * Xj[d];
                  }
               #endif
               BCL_vzero(Vatt0); 
               BCL_vzero(Vatt1); 
               BCL_vzero(Vatt2); 
               BCL_vzero(Vatt3); 
         
               BCL_vmac(Vatt0, Vxi0, Vxj0);
               BCL_vmac(Vatt1, Vxi1, Vxj1);
               BCL_vmac(Vatt2, Vxi2, Vxj2);
               BCL_vmac(Vatt3, Vxi3, Vxj3);
         
               BCL_vmac(Vatt0, Vxi4, Vxj4);
               BCL_vmac(Vatt1, Vxi5, Vxj5);
               BCL_vmac(Vatt2, Vxi6, Vxj6);
               BCL_vmac(Vatt3, Vxi7, Vxj7);
      
               BCL_vadd(Vatt0, Vatt0, Vatt1);
               BCL_vadd(Vatt2, Vatt2, Vatt3);
               BCL_vadd(Vatt0, Vatt0, Vatt2);
         
               BCL_vrsum1(attrc, Vatt0);
               //PrintVector("Vatt0", Vatt0);
	       //cout << attrc << endl;
               // will optimize the fast_SM function later  
               VALUETYPE d1 = fast_SM(attrc);
               d1 = STEP * degi * (1.0 - d1); 
               BCL_vset1(Vd1, d1);
               
               #if 0
               // 
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
                  Y[bindex - baseindex + d] += STEP * degi 
                                       * (1.0 - d1) * Xj[d];
	       }
               #endif
               BCL_vmac(Vy0, Vd1, Vxj0); 
               BCL_vmac(Vy1, Vd1, Vxj1); 
               BCL_vmac(Vy2, Vd1, Vxj2); 
               BCL_vmac(Vy3, Vd1, Vxj3); 
               BCL_vmac(Vy4, Vd1, Vxj4); 
               BCL_vmac(Vy5, Vd1, Vxj5); 
               BCL_vmac(Vy6, Vd1, Vxj6); 
               BCL_vmac(Vy7, Vd1, Vxj7); 
            }
/*
 *          repulsive force 
 */
            for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               VTYPE Vs0, Vs1, Vs2, Vs3, Vs4, Vs5, Vs6, Vs7;
               VTYPE Vd1;
               VTYPE Vrep0, Vrep1, Vrep2, Vrep3;  
      
               BCL_vldu(Vs0, S + jindex + VLEN*0);
               BCL_vldu(Vs1, S + jindex + VLEN*1);
               BCL_vldu(Vs2, S + jindex + VLEN*2);
               BCL_vldu(Vs3, S + jindex + VLEN*3);
               BCL_vldu(Vs4, S + jindex + VLEN*4);
               BCL_vldu(Vs5, S + jindex + VLEN*5);
               BCL_vldu(Vs6, S + jindex + VLEN*6);
               BCL_vldu(Vs7, S + jindex + VLEN*7);
               
               #if 0
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
	          repuls += X[iindex + d] * S[jindex + d];
               }
               #endif
               BCL_vzero(Vrep0); 
               BCL_vzero(Vrep1); 
               BCL_vzero(Vrep2); 
               BCL_vzero(Vrep3); 
         
               BCL_vmac(Vrep0, Vxi0, Vs0);
               BCL_vmac(Vrep1, Vxi1, Vs1);
               BCL_vmac(Vrep2, Vxi2, Vs2);
               BCL_vmac(Vrep3, Vxi3, Vs3);
         
               BCL_vmac(Vrep0, Vxi4, Vs4);
               BCL_vmac(Vrep1, Vxi5, Vs5);
               BCL_vmac(Vrep2, Vxi6, Vs6);
               BCL_vmac(Vrep3, Vxi7, Vs7);
      
               BCL_vadd(Vrep0, Vrep0, Vrep1);
               BCL_vadd(Vrep2, Vrep2, Vrep3);
               BCL_vadd(Vrep0, Vrep0, Vrep2);
         
               BCL_vrsum1(repuls, Vrep0); 

               VALUETYPE d1 = fast_SM(repuls);
               d1 = -1.0 * STEP * d1; // Y = Y - d * S  => Y += (-d) * S;
               BCL_vset1(Vd1, d1);

               #if 0	
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
                  Y[bindex - baseindex + d] -= 
                     (STEP) * d1 * S[jindex + d];
               }
               #endif
               BCL_vmac(Vy0, Vd1, Vs0); // Vd1 = [-d1, -d1, ... ]
               BCL_vmac(Vy1, Vd1, Vs1); 
               BCL_vmac(Vy2, Vd1, Vs2); 
               BCL_vmac(Vy3, Vd1, Vs3); 
               BCL_vmac(Vy4, Vd1, Vs4); 
               BCL_vmac(Vy5, Vd1, Vs5); 
               BCL_vmac(Vy6, Vd1, Vs6); 
               BCL_vmac(Vy7, Vd1, Vs7); 
            }
/*
 *          Store Y, FIXME: use aligned malloc  
 */
            BCL_vstu(Y + bindex - baseindex +VLEN*0, Vy0);
            BCL_vstu(Y + bindex - baseindex +VLEN*1, Vy1);
            BCL_vstu(Y + bindex - baseindex +VLEN*2, Vy2);
            BCL_vstu(Y + bindex - baseindex +VLEN*3, Vy3);
            BCL_vstu(Y + bindex - baseindex +VLEN*4, Vy4);
            BCL_vstu(Y + bindex - baseindex +VLEN*5, Vy5);
            BCL_vstu(Y + bindex - baseindex +VLEN*6, Vy6);
            BCL_vstu(Y + bindex - baseindex +VLEN*7, Vy7);
         }
         
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] = 
                  prevCoordinates[bindex - baseindex + d];
            }
         }
      }
/*
 *    Handle cleanup hare: no need to optimize 
 */
      INDEXTYPE b = (graph.rows/BATCHSIZE); // can be skipped 
      if (graph.rows > b * BATCHSIZE)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
            //int maxv = min((b+1) * BATCHSIZE, graph.rows-1);
            int maxv = graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
	 //INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
	 INDEXTYPE INITPREVSIZE = graph.rows;
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    if(i >= graph.rows) continue;
            INDEXTYPE IDIM = i*DIM;
            #pragma omp simd
	    for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
            }
         }
	 #pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue; /* get rid of it !!! */
            VALUETYPE forceDiff[DIM];
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
	    INDEXTYPE wlength = 0;
	    INDEXTYPE windex = i;
            VALUETYPE degi = 1.0;
	
            VALUETYPE *Xi = nCoordinates + iindex; 

            //if(graph.rowptr[i+1] - graph.rowptr[i] > 150)
	    degi = 1.0 / (graph.rowptr[i+1] - graph.rowptr[i] + 1);
	    //printf("Max degree of %d:%d\n", i, graph.rowptr[i+1] - graph.rowptr[i]);
/*
 *          attractive force calc: spase-dense computation 
 */
	    for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
	       VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = nCoordinates + jindex; 
 
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
                //attrc += nCoordinates[iindex + d] * nCoordinates[jindex + d];
                  attrc += Xi[d] * Xj[d];
                  //attrc += forceDiff[d] * forceDiff[d];
               }
               // table access: size of table = SM_TABLE_SIZE
               VALUETYPE d1 = fast_SM(attrc);
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
                  //prevCoordinates[bindex - baseindex + d] += STEP * degi 
                  //                     * (1.0 - d1) * nCoordinates[jindex + d];
                  prevCoordinates[bindex - baseindex + d] += STEP * degi 
                                       * (1.0 - d1) * Xj[d];
	       }
	       wlength++;
	       windex = colidj;
            }
/*
 *          repulsive force 
 */
            for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
	          repuls += this->nCoordinates[iindex + d] * samples[jindex + d];
               }
               VALUETYPE d1 = fast_SM(repuls);
					
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
                  prevCoordinates[bindex - baseindex + d] -= 
                     (STEP) * d1 * samples[jindex + d];
               }
            }
         }
         
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] = 
                  prevCoordinates[bindex - baseindex + d];
            }
         }
      }


      LOOP++;
   }
   end = omp_get_wtime();
   cout << "optForce2VecNSRW_AVXZ Parallel Wall time required:" << end - start << " seconds" << endl;
   
   result.push_back((float)(end - start));
   writeToFile("F2VWNS_AVXZ"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        
   return result;
}
									
vector<VALUETYPE> algorithms::AlgoForce2VecNSRWEFF_SREAL_D128_AVXZ
(
 INDEXTYPE ITERATIONS, 
 INDEXTYPE NUMOFTHREADS, 
 INDEXTYPE BATCHSIZE, 
 INDEXTYPE ns, 
 VALUETYPE lr
 )
{
   INDEXTYPE LOOP = 0;
   VALUETYPE STEP = lr, EPS = 0.00001;
   double start, end;
   double loglike0, loglike;
   
   vector<VALUETYPE> result;
   vector<INDEXTYPE> indices;
   vector<INDEXTYPE> nsamples;
   
   VALUETYPE *samples;
   INDEXTYPE *walksamples; 
   INDEXTYPE WALKLENGTH = 5;
   samples = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
   walksamples = static_cast<INDEXTYPE *> (::operator new (sizeof(INDEXTYPE[WALKLENGTH * graph.rows])));
   unsigned long long x = time(nullptr);
   for (int i = 0; i < 2; i++) 
   {
      unsigned long long z = x += UINT64_C(0x9E3779B97F4A7C15);
      z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
      z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
      rseed[i] = z ^ z >> 31;
   }
	
   init_SM_TABLE();
   omp_set_num_threads(NUMOFTHREADS);
   start = omp_get_wtime();
   randInit();
   INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
   VALUETYPE *prevCoordinates;
   int maxdegree = -1;
        
   prevCoordinates = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
   
   loglike0 = loglike = numeric_limits<double>::max();
   
   while(LOOP < ITERATIONS)
   {
      loglike0 = loglike;
      loglike = 0;

      for(INDEXTYPE i = 0; i < graph.rows; i++){
      	 INDEXTYPE wlength = 0;
         INDEXTYPE windex = i;
         while(wlength < WALKLENGTH){
         	INDEXTYPE j = windex;
                if(graph.rowptr[windex+1] - graph.rowptr[windex] > 2)
                	j = randIndex(graph.rowptr[windex+1]-1, graph.rowptr[windex]);
                else if(graph.rowptr[windex+1] - graph.rowptr[windex] == 2)
                	j = graph.rowptr[windex];
              	walksamples[i*WALKLENGTH + wlength] = graph.colids[j];
               	wlength++;
                windex = graph.colids[j];
          }
        }	

	for(INDEXTYPE b = 0; b < graph.rows / BATCHSIZE; b += 1)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
            int maxv = min((b+1) * BATCHSIZE, graph.rows-1);//graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
  	 INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            INDEXTYPE IDIM = i*DIM;
            #pragma omp simd
	    for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
            }
         }
 	 #pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
		VALUETYPE *X = nCoordinates; 
            VALUETYPE *Y = prevCoordinates; 
            VALUETYPE *S = samples; 

            INDEXTYPE iindex = i * DIM;
            VALUETYPE *Xi = X + iindex; 
            INDEXTYPE bindex = i * DIM;
            VALUETYPE degi = 1.0;
	
            VTYPE Vy0, Vy1, Vy2, Vy3, Vy4, Vy5, Vy6, Vy7, Vy8; 
            VTYPE Vxi0, Vxi1, Vxi2, Vxi3, Vxi4, Vxi5, Vxi6, Vxi7, Vxi8; 
	    
	    BCL_vldu(Vy0, Y + bindex - baseindex +VLEN*0);
            BCL_vldu(Vy1, Y + bindex - baseindex +VLEN*1);
            BCL_vldu(Vy2, Y + bindex - baseindex +VLEN*2);
            BCL_vldu(Vy3, Y + bindex - baseindex +VLEN*3);
            BCL_vldu(Vy4, Y + bindex - baseindex +VLEN*4);
            BCL_vldu(Vy5, Y + bindex - baseindex +VLEN*5);
            BCL_vldu(Vy6, Y + bindex - baseindex +VLEN*6);
            BCL_vldu(Vy7, Y + bindex - baseindex +VLEN*7);

   	    BCL_vldu(Vxi0, Xi + VLEN*0);
            BCL_vldu(Vxi1, Xi + VLEN*1);
            BCL_vldu(Vxi2, Xi + VLEN*2);
            BCL_vldu(Vxi3, Xi + VLEN*3);
            BCL_vldu(Vxi4, Xi + VLEN*4);
            BCL_vldu(Vxi5, Xi + VLEN*5);
            BCL_vldu(Vxi6, Xi + VLEN*6);
            BCL_vldu(Vxi7, Xi + VLEN*7);

	    for(INDEXTYPE j = 0; j < WALKLENGTH; j += 1)
            {
               VALUETYPE attrc = 0;
               INDEXTYPE colidj = walksamples[i * WALKLENGTH + j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = X + jindex; 
        
               VTYPE Vd1;
               VTYPE Vatt0, Vatt1, Vatt2, Vatt3;  
               VTYPE Vxj0, Vxj1, Vxj2, Vxj3, Vxj4, Vxj5, Vxj6, Vxj7;

		BCL_vldu(Vxj0, Xj + VLEN*0);
               BCL_vldu(Vxj1, Xj + VLEN*1);
               BCL_vldu(Vxj2, Xj + VLEN*2);
               BCL_vldu(Vxj3, Xj + VLEN*3);
               BCL_vldu(Vxj4, Xj + VLEN*4);
               BCL_vldu(Vxj5, Xj + VLEN*5);
               BCL_vldu(Vxj6, Xj + VLEN*6);
               BCL_vldu(Vxj7, Xj + VLEN*7);

		BCL_vzero(Vatt0); 
               BCL_vzero(Vatt1); 
               BCL_vzero(Vatt2); 
               BCL_vzero(Vatt3); 
         
               BCL_vmac(Vatt0, Vxi0, Vxj0);
               BCL_vmac(Vatt1, Vxi1, Vxj1);
               BCL_vmac(Vatt2, Vxi2, Vxj2);
               BCL_vmac(Vatt3, Vxi3, Vxj3);
         
               BCL_vmac(Vatt0, Vxi4, Vxj4);
               BCL_vmac(Vatt1, Vxi5, Vxj5);
               BCL_vmac(Vatt2, Vxi6, Vxj6);
               BCL_vmac(Vatt3, Vxi7, Vxj7);
      
               BCL_vadd(Vatt0, Vatt0, Vatt1);
               BCL_vadd(Vatt2, Vatt2, Vatt3);
               BCL_vadd(Vatt0, Vatt0, Vatt2);
         
               BCL_vrsum1(attrc, Vatt0);

		VALUETYPE d1 = fast_SM(attrc);
               d1 = STEP * degi * (1.0 - d1); 
               BCL_vset1(Vd1, d1);

		BCL_vmac(Vy0, Vd1, Vxj0); 
               BCL_vmac(Vy1, Vd1, Vxj1); 
               BCL_vmac(Vy2, Vd1, Vxj2); 
               BCL_vmac(Vy3, Vd1, Vxj3); 
               BCL_vmac(Vy4, Vd1, Vxj4); 
               BCL_vmac(Vy5, Vd1, Vxj5); 
               BCL_vmac(Vy6, Vd1, Vxj6); 
               BCL_vmac(Vy7, Vd1, Vxj7); 
            }

	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               VTYPE Vs0, Vs1, Vs2, Vs3, Vs4, Vs5, Vs6, Vs7;
               VTYPE Vd1;
               VTYPE Vrep0, Vrep1, Vrep2, Vrep3;  
      
               BCL_vldu(Vs0, S + jindex + VLEN*0);
               BCL_vldu(Vs1, S + jindex + VLEN*1);
               BCL_vldu(Vs2, S + jindex + VLEN*2);
               BCL_vldu(Vs3, S + jindex + VLEN*3);
               BCL_vldu(Vs4, S + jindex + VLEN*4);
               BCL_vldu(Vs5, S + jindex + VLEN*5);
               BCL_vldu(Vs6, S + jindex + VLEN*6);
               BCL_vldu(Vs7, S + jindex + VLEN*7);

		BCL_vzero(Vrep0); 
               BCL_vzero(Vrep1); 
               BCL_vzero(Vrep2); 
               BCL_vzero(Vrep3); 
         
               BCL_vmac(Vrep0, Vxi0, Vs0);
               BCL_vmac(Vrep1, Vxi1, Vs1);
               BCL_vmac(Vrep2, Vxi2, Vs2);
               BCL_vmac(Vrep3, Vxi3, Vs3);
         
               BCL_vmac(Vrep0, Vxi4, Vs4);
               BCL_vmac(Vrep1, Vxi5, Vs5);
               BCL_vmac(Vrep2, Vxi6, Vs6);
               BCL_vmac(Vrep3, Vxi7, Vs7);
      
               BCL_vadd(Vrep0, Vrep0, Vrep1);
               BCL_vadd(Vrep2, Vrep2, Vrep3);
               BCL_vadd(Vrep0, Vrep0, Vrep2);
         
               BCL_vrsum1(repuls, Vrep0); 

               VALUETYPE d1 = fast_SM(repuls);
               d1 = -1.0 * STEP * d1; // Y = Y - d * S  => Y += (-d) * S;
               BCL_vset1(Vd1, d1);

		BCL_vmac(Vy0, Vd1, Vs0); // Vd1 = [-d1, -d1, ... ]
               BCL_vmac(Vy1, Vd1, Vs1); 
               BCL_vmac(Vy2, Vd1, Vs2); 
               BCL_vmac(Vy3, Vd1, Vs3); 
               BCL_vmac(Vy4, Vd1, Vs4); 
               BCL_vmac(Vy5, Vd1, Vs5); 
               BCL_vmac(Vy6, Vd1, Vs6); 
               BCL_vmac(Vy7, Vd1, Vs7); 
            }

	    BCL_vstu(Y + bindex - baseindex +VLEN*0, Vy0);
            BCL_vstu(Y + bindex - baseindex +VLEN*1, Vy1);
            BCL_vstu(Y + bindex - baseindex +VLEN*2, Vy2);
            BCL_vstu(Y + bindex - baseindex +VLEN*3, Vy3);
            BCL_vstu(Y + bindex - baseindex +VLEN*4, Vy4);
            BCL_vstu(Y + bindex - baseindex +VLEN*5, Vy5);
            BCL_vstu(Y + bindex - baseindex +VLEN*6, Vy6);
            BCL_vstu(Y + bindex - baseindex +VLEN*7, Vy7);
         }
         
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] = 
                  prevCoordinates[bindex - baseindex + d];
            }
         }
      }
	INDEXTYPE b = (graph.rows/BATCHSIZE); // can be skipped 
      if (graph.rows > b * BATCHSIZE)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
	    int maxv = graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
	 INDEXTYPE INITPREVSIZE = graph.rows;
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    if(i >= graph.rows) continue;
            INDEXTYPE IDIM = i*DIM;
            #pragma omp simd
	    for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
            }
         }
	 #pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue; /* get rid of it !!! */
            VALUETYPE forceDiff[DIM];
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
	    INDEXTYPE wlength = 0;
	    INDEXTYPE windex = i;
            VALUETYPE degi = 1.0;
	
            VALUETYPE *Xi = nCoordinates + iindex; 

	    for(INDEXTYPE j = 0; j < WALKLENGTH; j += 1)
            {
	       VALUETYPE attrc = 0;
               INDEXTYPE colidj =  walksamples[i * WALKLENGTH + j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = nCoordinates + jindex; 
 
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {

			attrc += Xi[d] * Xj[d];
		}

		VALUETYPE d1 = fast_SM(attrc);
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
			prevCoordinates[bindex - baseindex + d] += STEP * degi * (1.0 - d1) * Xj[d];
	       }
	       wlength++;
	       windex = colidj;
            }
	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
	          repuls += this->nCoordinates[iindex + d] * samples[jindex + d];
               }
               VALUETYPE d1 = fast_SM(repuls);
					
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
                  prevCoordinates[bindex - baseindex + d] -= 
                     (STEP) * d1 * samples[jindex + d];
               }
            }
         }
         
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] = 
                  prevCoordinates[bindex - baseindex + d];
            }
         }
      }


      LOOP++;
   }
   end = omp_get_wtime();
   cout << "optForce2VecNSRWEFF_AVXZ Parallel Wall time required:" << end - start << " seconds" << endl;
   
   result.push_back((float)(end - start));
   writeToFile("F2VWEFFNS_AVXZ"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        
   return result;
}

vector<VALUETYPE> algorithms::AlgoForce2VecNSLB_SREAL_D128_AVXZ
(
 INDEXTYPE ITERATIONS, 
 INDEXTYPE NUMOFTHREADS, 
 INDEXTYPE BATCHSIZE, 
 INDEXTYPE ns, 
 VALUETYPE lr
 )
{
   INDEXTYPE LOOP = 0;
   VALUETYPE STEP = lr, EPS = 0.00001;
   double start, end;
   double loglike0, loglike;
   
   vector<VALUETYPE> result;
   vector<INDEXTYPE> indices;
   vector<INDEXTYPE> nsamples;
   VTYPE VMAXBOUND, VMINBOUND, VEPS, VSTEP;   
   VALUETYPE *samples;
   samples = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
   
   omp_set_num_threads(NUMOFTHREADS);
   start = omp_get_wtime();
   randInitF();
   INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
   VALUETYPE *prevCoordinates;
   int maxdegree = -1;
        
   prevCoordinates = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));

   for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
		INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                        prevCoordinates[IDIM + d] = 0;
                }
   }

   BCL_vset1(VMAXBOUND, MAXBOUND); 
   BCL_vset1(VMINBOUND, -MAXBOUND); 

   INDEXTYPE *ThRowId;
   //ThRowId = static_cast<INDEXTYPE *> (::operator new (sizeof(INDEXTYPE[NUMOFTHREADS+1])));
   ThRowId = (INDEXTYPE *) malloc(sizeof(INDEXTYPE)*(NUMOFTHREADS+1));
   loglike0 = loglike = numeric_limits<double>::max();
   while(LOOP < ITERATIONS)
   {
      loglike0 = loglike;
      loglike = 0;
      for(INDEXTYPE b = 0; b < graph.rows / BATCHSIZE; b += 1)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
	    int maxv = graph.rows-1;
            int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
         INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
	 
	/*..........load balancing(prefix sum)..................*/
   	 INDEXTYPE RowPerThd; 
   	 INDEXTYPE i;
   	 INDEXTYPE nnzB = 0;

	 for (i=b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
      		nnzB += (graph.rowptr[i+1] - graph.rowptr[i]); 
	 RowPerThd = nnzB / NUMOFTHREADS; 
	 ThRowId[0] = b * BATCHSIZE;
	 INDEXTYPE curRow = 0;
	 INDEXTYPE cumRow = 0;
	 INDEXTYPE tt=1;
      	 for(i=b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
      	 {
		INDEXTYPE deg = graph.rowptr[i+1] - graph.rowptr[i];
		cumRow += deg;
		curRow += deg;
		if(curRow > RowPerThd){
			ThRowId[tt] = i;
			curRow = 0;
			RowPerThd = (nnzB - cumRow) / (NUMOFTHREADS - tt);
			tt += 1;
		}
      	 }
	 ThRowId[tt] = (b + 1) * BATCHSIZE;
	 assert(cumRow == nnzB);
	 for(i= tt+1; i< NUMOFTHREADS+1; i += 1){
                ThRowId[i] = 0;
         }
	 /*..............load balancing.............*/
	 /*for(INDEXTYPE i = 0; i < NUMOFTHREADS; i += 1){
		printf("%d --- %d:%d\n", i, ThRowId[i+1]-ThRowId[i],ThRowId[i]);
	 }
	 printf("%d ### %d,%d\n", NUMOFTHREADS, ThRowId[NUMOFTHREADS],ThRowId[NUMOFTHREADS-1]);
	 printf("OK\n");
	 //exit(0);
	 */ 
         #pragma omp parallel
	 {
	 INDEXTYPE id = omp_get_thread_num();
         for(INDEXTYPE i = ThRowId[id]; i < ThRowId[id+1]; i += 1)
         {
            VALUETYPE *X = nCoordinates; 
            VALUETYPE *Y = prevCoordinates; 
            VALUETYPE *S = samples; 

            INDEXTYPE iindex = i * DIM;
            VALUETYPE *Xi = X + iindex; 
            INDEXTYPE bindex = i * DIM;
            VALUETYPE degi = 1.0;

            VTYPE Vy0, Vy1, Vy2, Vy3, Vy4, Vy5, Vy6, Vy7; 
            VTYPE Vxi0, Vxi1, Vxi2, Vxi3, Vxi4, Vxi5, Vxi6, Vxi7; 
            BCL_vldu(Vy0, Y + bindex - baseindex +VLEN*0);
            BCL_vldu(Vy1, Y + bindex - baseindex +VLEN*1);
            BCL_vldu(Vy2, Y + bindex - baseindex +VLEN*2);
            BCL_vldu(Vy3, Y + bindex - baseindex +VLEN*3);
            BCL_vldu(Vy4, Y + bindex - baseindex +VLEN*4);
            BCL_vldu(Vy5, Y + bindex - baseindex +VLEN*5);
            BCL_vldu(Vy6, Y + bindex - baseindex +VLEN*6);
            BCL_vldu(Vy7, Y + bindex - baseindex +VLEN*7);
            
            BCL_vldu(Vxi0, Xi + VLEN*0);
            BCL_vldu(Vxi1, Xi + VLEN*1);
            BCL_vldu(Vxi2, Xi + VLEN*2);
            BCL_vldu(Vxi3, Xi + VLEN*3);
            BCL_vldu(Vxi4, Xi + VLEN*4);
            BCL_vldu(Vxi5, Xi + VLEN*5);
            BCL_vldu(Vxi6, Xi + VLEN*6);
            BCL_vldu(Vxi7, Xi + VLEN*7);
	
	    VTYPE Vt, Vd0, Vd1, Vd2, Vd3, Vd4, Vd5, Vd6, Vd7;

            for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
               VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = X + jindex; 
        
               VTYPE Vtemp, Vatt0, Vatt1, Vatt2, Vatt3;
               VTYPE Vxj0, Vxj1, Vxj2, Vxj3, Vxj4, Vxj5, Vxj6, Vxj7; 

               BCL_vldu(Vxj0, Xj + VLEN*0);
               BCL_vldu(Vxj1, Xj + VLEN*1);
               BCL_vldu(Vxj2, Xj + VLEN*2);
               BCL_vldu(Vxj3, Xj + VLEN*3);
               BCL_vldu(Vxj4, Xj + VLEN*4);
               BCL_vldu(Vxj5, Xj + VLEN*5);
               BCL_vldu(Vxj6, Xj + VLEN*6);
               BCL_vldu(Vxj7, Xj + VLEN*7);
               
	       BCL_vzero(Vatt0); 
	       BCL_vzero(Vatt1);
	       BCL_vzero(Vatt2);
	       BCL_vzero(Vatt3);
         
               BCL_vsub(Vd0, Vxi0, Vxj0);
               BCL_vsub(Vd1, Vxi1, Vxj1);
               BCL_vsub(Vd2, Vxi2, Vxj2);
               BCL_vsub(Vd3, Vxi3, Vxj3);
               BCL_vsub(Vd4, Vxi4, Vxj4);
               BCL_vsub(Vd5, Vxi5, Vxj5);
               BCL_vsub(Vd6, Vxi6, Vxj6);
               BCL_vsub(Vd7, Vxi7, Vxj7);

	       BCL_vmac(Vatt0, Vd0, Vd0);
               BCL_vmac(Vatt1, Vd1, Vd1);
               BCL_vmac(Vatt2, Vd2, Vd2);
               BCL_vmac(Vatt3, Vd3, Vd3);
               BCL_vmac(Vatt0, Vd4, Vd4);
               BCL_vmac(Vatt1, Vd5, Vd5);
               BCL_vmac(Vatt2, Vd6, Vd6);
               BCL_vmac(Vatt3, Vd7, Vd7);

               BCL_vadd(Vatt0, Vatt0, Vatt1);
               BCL_vadd(Vatt2, Vatt2, Vatt3);
               BCL_vadd(Vatt0, Vatt0, Vatt2);
         
               BCL_vrsum1(attrc, Vatt0);

	       BCL_vset1(Vatt0, attrc); // a = a
	       BCL_vset1(Vt, 1.0f); // t = 1
	       BCL_vadd(Vatt0, Vatt0, Vt); // a = t + a

	       BCL_vrcp(Vatt0, Vatt0); // a = 1/a
	       BCL_vset1(Vt, -2.0f); // t = -2
	       BCL_vmul(Vatt0, Vatt0, Vt); // a = -2 * a
               
               BCL_vmul(Vd0, Vatt0, Vd0); 
	       BCL_vmul(Vd1, Vatt0, Vd1);  
	       BCL_vmul(Vd2, Vatt0, Vd2);  
 	       BCL_vmul(Vd3, Vatt0, Vd3);  
 	       BCL_vmul(Vd4, Vatt0, Vd4);  
               BCL_vmul(Vd5, Vatt0, Vd5);
               BCL_vmul(Vd6, Vatt0, Vd6);
               BCL_vmul(Vd7, Vatt0, Vd7);               

	       BCL_vmax(Vd0, Vd0, VMINBOUND);
	       BCL_vmax(Vd1, Vd1, VMINBOUND); 
	       BCL_vmax(Vd2, Vd2, VMINBOUND);  
               BCL_vmax(Vd3, Vd3, VMINBOUND);
               BCL_vmax(Vd4, Vd4, VMINBOUND);  
               BCL_vmax(Vd5, Vd5, VMINBOUND);
               BCL_vmax(Vd6, Vd6, VMINBOUND);
               BCL_vmax(Vd7, Vd7, VMINBOUND);

	       BCL_vmin(Vd0, Vd0, VMAXBOUND);
               BCL_vmin(Vd1, Vd1, VMAXBOUND);
               BCL_vmin(Vd2, Vd2, VMAXBOUND);
               BCL_vmin(Vd3, Vd3, VMAXBOUND);
               BCL_vmin(Vd4, Vd4, VMAXBOUND);
               BCL_vmin(Vd5, Vd5, VMAXBOUND);
               BCL_vmin(Vd6, Vd6, VMAXBOUND);
               BCL_vmin(Vd7, Vd7, VMAXBOUND);
	
	       BCL_vset1(Vt, STEP);
	     
               BCL_vmac(Vy0, Vd0, Vt); 
               BCL_vmac(Vy1, Vd1, Vt); 
               BCL_vmac(Vy2, Vd2, Vt); 
               BCL_vmac(Vy3, Vd3, Vt); 
               BCL_vmac(Vy4, Vd4, Vt); 
               BCL_vmac(Vy5, Vd5, Vt); 
               BCL_vmac(Vy6, Vd6, Vt); 
               BCL_vmac(Vy7, Vd7, Vt); 
            }
	
	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               VTYPE Vs0, Vs1, Vs2, Vs3, Vs4, Vs5, Vs6, Vs7;
               VTYPE Vd1;
               VTYPE Vrep0, Vrep1, Vrep2, Vrep3;  
      
               BCL_vldu(Vs0, S + jindex + VLEN*0);
               BCL_vldu(Vs1, S + jindex + VLEN*1);
               BCL_vldu(Vs2, S + jindex + VLEN*2);
               BCL_vldu(Vs3, S + jindex + VLEN*3);
               BCL_vldu(Vs4, S + jindex + VLEN*4);
               BCL_vldu(Vs5, S + jindex + VLEN*5);
               BCL_vldu(Vs6, S + jindex + VLEN*6);
               BCL_vldu(Vs7, S + jindex + VLEN*7);

	       BCL_vzero(Vrep0); 
               BCL_vzero(Vrep1); 
               BCL_vzero(Vrep2); 
               BCL_vzero(Vrep3); 

	       BCL_vsub(Vd0, Vxi0, Vs0);
               BCL_vsub(Vd1, Vxi1, Vs1);
               BCL_vsub(Vd2, Vxi2, Vs2);
               BCL_vsub(Vd3, Vxi3, Vs3);
               BCL_vsub(Vd4, Vxi4, Vs4);
               BCL_vsub(Vd5, Vxi5, Vs5);
               BCL_vsub(Vd6, Vxi6, Vs6);
               BCL_vsub(Vd7, Vxi7, Vs7);

               BCL_vmac(Vrep0, Vd0, Vd0);
               BCL_vmac(Vrep1, Vd1, Vd1);
               BCL_vmac(Vrep2, Vd2, Vd2);
               BCL_vmac(Vrep3, Vd3, Vd3);
               BCL_vmac(Vrep0, Vd4, Vd4);
               BCL_vmac(Vrep1, Vd5, Vd5);
               BCL_vmac(Vrep2, Vd6, Vd6);
               BCL_vmac(Vrep3, Vd7, Vd7);

               BCL_vadd(Vrep0, Vrep0, Vrep1);
               BCL_vadd(Vrep2, Vrep2, Vrep3);
               BCL_vadd(Vrep0, Vrep0, Vrep2);

               BCL_vrsum1(repuls, Vrep0);
	       
	       BCL_vset1(Vrep0, repuls); // a = a
               BCL_vset1(Vt, 1.0f); // t = 1
               BCL_vadd(Vt, Vrep0, Vt); // t = t + a
	       BCL_vmul(Vrep0, Vrep0, Vt); // a = t * a

               BCL_vrcp(Vrep0, Vrep0); // a = 1/a
               BCL_vset1(Vt, 2.0f); // t = 2
               BCL_vmul(Vrep0, Vrep0, Vt); // a = 2 * a

               BCL_vmul(Vd0, Vrep0, Vd0);
               BCL_vmul(Vd1, Vrep0, Vd1);
               BCL_vmul(Vd2, Vrep0, Vd2);
               BCL_vmul(Vd3, Vrep0, Vd3);
               BCL_vmul(Vd4, Vrep0, Vd4);
               BCL_vmul(Vd5, Vrep0, Vd5);
               BCL_vmul(Vd6, Vrep0, Vd6);
               BCL_vmul(Vd7, Vrep0, Vd7);

               BCL_vmax(Vd0, Vd0, VMINBOUND);
               BCL_vmax(Vd1, Vd1, VMINBOUND);
               BCL_vmax(Vd2, Vd2, VMINBOUND);
               BCL_vmax(Vd3, Vd3, VMINBOUND);
               BCL_vmax(Vd4, Vd4, VMINBOUND);
               BCL_vmax(Vd5, Vd5, VMINBOUND);
               BCL_vmax(Vd6, Vd6, VMINBOUND);
               BCL_vmax(Vd7, Vd7, VMINBOUND);

               BCL_vmin(Vd0, Vd0, VMAXBOUND);
               BCL_vmin(Vd1, Vd1, VMAXBOUND);
               BCL_vmin(Vd2, Vd2, VMAXBOUND);
               BCL_vmin(Vd3, Vd3, VMAXBOUND);
               BCL_vmin(Vd4, Vd4, VMAXBOUND);
               BCL_vmin(Vd5, Vd5, VMAXBOUND);
               BCL_vmin(Vd6, Vd6, VMAXBOUND);
               BCL_vmin(Vd7, Vd7, VMAXBOUND);

               BCL_vset1(Vt, STEP);
	     
               BCL_vmac(Vy0, Vd0, Vt);
               BCL_vmac(Vy1, Vd1, Vt);
               BCL_vmac(Vy2, Vd2, Vt);
               BCL_vmac(Vy3, Vd3, Vt);
               BCL_vmac(Vy4, Vd4, Vt);
               BCL_vmac(Vy5, Vd5, Vt);
               BCL_vmac(Vy6, Vd6, Vt);
               BCL_vmac(Vy7, Vd7, Vt);


            }
	    BCL_vstu(Y + bindex - baseindex +VLEN*0, Vy0);
            BCL_vstu(Y + bindex - baseindex +VLEN*1, Vy1);
            BCL_vstu(Y + bindex - baseindex +VLEN*2, Vy2);
            BCL_vstu(Y + bindex - baseindex +VLEN*3, Vy3);
            BCL_vstu(Y + bindex - baseindex +VLEN*4, Vy4);
            BCL_vstu(Y + bindex - baseindex +VLEN*5, Vy5);
            BCL_vstu(Y + bindex - baseindex +VLEN*6, Vy6);
            BCL_vstu(Y + bindex - baseindex +VLEN*7, Vy7);
         }
         }  
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] += 
                  prevCoordinates[bindex - baseindex + d];
		prevCoordinates[bindex - baseindex + d] = 0;
            }
         }
      }
	INDEXTYPE b = (graph.rows/BATCHSIZE); // can be skipped 
	if (graph.rows > b * BATCHSIZE)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;
	    int maxv = graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
	 INDEXTYPE INITPREVSIZE = graph.rows;
	 #pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    VALUETYPE forceDiff[DIM];
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
	    INDEXTYPE wlength = 0;
	    INDEXTYPE windex = i;
            VALUETYPE degi = 1.0;
	
            VALUETYPE *Xi = nCoordinates + iindex; 
  	    for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
	       VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = nCoordinates + jindex; 
	       VALUETYPE forceDiff[DIM];
 
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
			forceDiff[d] = Xj[d] - Xi[d];
			attrc += forceDiff[d] * forceDiff[d];
		}
	       VALUETYPE d1 = -2.0 / (1.0 + attrc);
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
			forceDiff[d] = scale(forceDiff[d] * d1);
			prevCoordinates[bindex - baseindex + d] += STEP * forceDiff[d];
		 }
            }
	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
	           forceDiff[d] = this->nCoordinates[iindex + d] - samples[jindex + d];
		  repuls += forceDiff[d] * forceDiff[d];
               }
               VALUETYPE d1 = 2.0 / ((repuls) * (1.0 + repuls));
					
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
		 	forceDiff[d] = scale(forceDiff[d] * d1);
                  	prevCoordinates[bindex - baseindex + d] +=  (STEP) * forceDiff[d];
               }
            }
         }
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] += prevCoordinates[bindex - baseindex + d];
            	prevCoordinates[bindex - baseindex + d] = 0;
	    }
         }
      }
      //printf("Iteration = %d\n", LOOP);

      LOOP++;
   }
	
   end = omp_get_wtime();
   cout << "optForce2VecNSLB_AVXZ Parallel Wall time required:" << end - start << " seconds" << endl;
   
   result.push_back((float)(end - start));
   writeToFile("F2VNSLB_AVXZ"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        
   return result;
		
}

vector<VALUETYPE> algorithms::AlgoForce2VecNSRWLB_SREAL_D64_AVXZ
(
 INDEXTYPE ITERATIONS, 
 INDEXTYPE NUMOFTHREADS, 
 INDEXTYPE BATCHSIZE, 
 INDEXTYPE ns, 
 VALUETYPE lr
 )
{
   INDEXTYPE LOOP = 0;
   VALUETYPE STEP = lr, EPS = 0.00001;
   double start, end;
   double loglike0, loglike;
   
   vector<VALUETYPE> result;
   vector<INDEXTYPE> indices;
   vector<INDEXTYPE> nsamples;
   
   VALUETYPE *samples;
   samples = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
   
   unsigned long long x = time(nullptr);
   for (int i = 0; i < 2; i++) 
   {
      unsigned long long z = x += UINT64_C(0x9E3779B97F4A7C15);
      z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
      z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
      rseed[i] = z ^ z >> 31;
   }
	
   init_SM_TABLE();
   omp_set_num_threads(NUMOFTHREADS);
   start = omp_get_wtime();
   randInit();
   INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
   VALUETYPE *prevCoordinates;
   int maxdegree = -1;
        
   prevCoordinates = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
   
   loglike0 = loglike = numeric_limits<double>::max();
   INDEXTYPE *ThRowId;
   ThRowId = (INDEXTYPE *) malloc(sizeof(INDEXTYPE)*(NUMOFTHREADS+1));
   while(LOOP < ITERATIONS)
   {
      loglike0 = loglike;
      loglike = 0;	
      for(INDEXTYPE b = 0; b < graph.rows / BATCHSIZE; b += 1)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
	    int minv = 0;
	    int maxv = (b+1) * BATCHSIZE;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
	INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
	INDEXTYPE RowPerThd; 
   	 INDEXTYPE i;
   	 INDEXTYPE nnzB = 0;
	for (i=b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
      		nnzB += (graph.rowptr[i+1] - graph.rowptr[i]); 
	 RowPerThd = nnzB / NUMOFTHREADS; 
	 ThRowId[0] = b * BATCHSIZE;
	 INDEXTYPE curRow = 0;
	 INDEXTYPE cumRow = 0;
	 INDEXTYPE tt=1;
      	 for(i=b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
      	 {
		INDEXTYPE deg = graph.rowptr[i+1] - graph.rowptr[i];
		cumRow += deg;
		curRow += deg;
		if(curRow > RowPerThd){
			ThRowId[tt] = i;
			curRow = 0;
			RowPerThd = (nnzB - cumRow) / (NUMOFTHREADS - tt);
			tt += 1;
		}
      	 }
	 ThRowId[tt] = (b + 1) * BATCHSIZE;
	 assert(cumRow == nnzB);
	 for(i= tt+1; i< NUMOFTHREADS+1; i += 1){
                ThRowId[i] = 0;
         } 
	for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            INDEXTYPE IDIM = i*DIM;
            #pragma omp simd
	    for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
            }
         }
	
	#pragma omp parallel
	{
	INDEXTYPE id = omp_get_thread_num();
         for(INDEXTYPE i = ThRowId[id]; i < ThRowId[id+1]; i += 1)
         //for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    VALUETYPE *X = nCoordinates; 
            VALUETYPE *Y = prevCoordinates; 
            VALUETYPE *S = samples; 

            INDEXTYPE iindex = i * DIM;
            VALUETYPE *Xi = X + iindex; 
            INDEXTYPE bindex = i * DIM;
            VALUETYPE degi = 1.0;
	
            degi = 1.0 / (graph.rowptr[i+1] - graph.rowptr[i] + 1);

            VTYPE Vy0, Vy1, Vy2, Vy3; 
            VTYPE Vxi0, Vxi1, Vxi2, Vxi3; 
	    BCL_vldu(Vy0, Y + bindex - baseindex +VLEN*0);
            BCL_vldu(Vy1, Y + bindex - baseindex +VLEN*1);
            BCL_vldu(Vy2, Y + bindex - baseindex +VLEN*2);
            BCL_vldu(Vy3, Y + bindex - baseindex +VLEN*3);
            //BCL_vldu(Vy4, Y + bindex - baseindex +VLEN*4);
            //BCL_vldu(Vy5, Y + bindex - baseindex +VLEN*5);
            //BCL_vldu(Vy6, Y + bindex - baseindex +VLEN*6);
            //BCL_vldu(Vy7, Y + bindex - baseindex +VLEN*7);

	    BCL_vldu(Vxi0, Xi + VLEN*0);
            BCL_vldu(Vxi1, Xi + VLEN*1);
            BCL_vldu(Vxi2, Xi + VLEN*2);
            BCL_vldu(Vxi3, Xi + VLEN*3);
            //BCL_vldu(Vxi4, Xi + VLEN*4);
            //BCL_vldu(Vxi5, Xi + VLEN*5);
            //BCL_vldu(Vxi6, Xi + VLEN*6);
            //BCL_vldu(Vxi7, Xi + VLEN*7);

	    for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
               VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = X + jindex; 
        
               VTYPE Vd1;
               VTYPE Vatt0, Vatt1, Vatt2, Vatt3;  
               VTYPE Vxj0, Vxj1, Vxj2, Vxj3; 
	       BCL_vldu(Vxj0, Xj + VLEN*0);
               BCL_vldu(Vxj1, Xj + VLEN*1);
               BCL_vldu(Vxj2, Xj + VLEN*2);
               BCL_vldu(Vxj3, Xj + VLEN*3);
               //BCL_vldu(Vxj4, Xj + VLEN*4);
               //BCL_vldu(Vxj5, Xj + VLEN*5);
               //BCL_vldu(Vxj6, Xj + VLEN*6);
               //BCL_vldu(Vxj7, Xj + VLEN*7);

	       BCL_vzero(Vatt0); 
               BCL_vzero(Vatt1); 
               BCL_vzero(Vatt2); 
               BCL_vzero(Vatt3); 
         
               BCL_vmac(Vatt0, Vxi0, Vxj0);
               BCL_vmac(Vatt1, Vxi1, Vxj1);
               BCL_vmac(Vatt2, Vxi2, Vxj2);
               BCL_vmac(Vatt3, Vxi3, Vxj3);
         
               //BCL_vmac(Vatt0, Vxi4, Vxj4);
               //BCL_vmac(Vatt1, Vxi5, Vxj5);
               //BCL_vmac(Vatt2, Vxi6, Vxj6);
               //BCL_vmac(Vatt3, Vxi7, Vxj7);
      
               BCL_vadd(Vatt0, Vatt0, Vatt1);
               BCL_vadd(Vatt2, Vatt2, Vatt3);
               BCL_vadd(Vatt0, Vatt0, Vatt2);
         
               BCL_vrsum1(attrc, Vatt0);

	       VALUETYPE d1 = fast_SM(attrc);
               d1 = STEP * degi * (1.0 - d1); 
               BCL_vset1(Vd1, d1);

	       BCL_vmac(Vy0, Vd1, Vxj0); 
               BCL_vmac(Vy1, Vd1, Vxj1); 
               BCL_vmac(Vy2, Vd1, Vxj2); 
               BCL_vmac(Vy3, Vd1, Vxj3); 
               //BCL_vmac(Vy4, Vd1, Vxj4); 
               //BCL_vmac(Vy5, Vd1, Vxj5); 
               //BCL_vmac(Vy6, Vd1, Vxj6); 
               //BCL_vmac(Vy7, Vd1, Vxj7); 
            }

	   for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               VTYPE Vs0, Vs1, Vs2, Vs3;
               VTYPE Vd1;
               VTYPE Vrep0, Vrep1, Vrep2, Vrep3;  
      
               BCL_vldu(Vs0, S + jindex + VLEN*0);
               BCL_vldu(Vs1, S + jindex + VLEN*1);
               BCL_vldu(Vs2, S + jindex + VLEN*2);
               BCL_vldu(Vs3, S + jindex + VLEN*3);
               //BCL_vldu(Vs4, S + jindex + VLEN*4);
               //BCL_vldu(Vs5, S + jindex + VLEN*5);
               //BCL_vldu(Vs6, S + jindex + VLEN*6);
               //BCL_vldu(Vs7, S + jindex + VLEN*7);

	       BCL_vzero(Vrep0); 
               BCL_vzero(Vrep1); 
               BCL_vzero(Vrep2); 
               BCL_vzero(Vrep3); 
         
               BCL_vmac(Vrep0, Vxi0, Vs0);
               BCL_vmac(Vrep1, Vxi1, Vs1);
               BCL_vmac(Vrep2, Vxi2, Vs2);
               BCL_vmac(Vrep3, Vxi3, Vs3);
         
               //BCL_vmac(Vrep0, Vxi4, Vs4);
               //BCL_vmac(Vrep1, Vxi5, Vs5);
               //BCL_vmac(Vrep2, Vxi6, Vs6);
               //BCL_vmac(Vrep3, Vxi7, Vs7);
      
               BCL_vadd(Vrep0, Vrep0, Vrep1);
               BCL_vadd(Vrep2, Vrep2, Vrep3);
               BCL_vadd(Vrep0, Vrep0, Vrep2);
         
               BCL_vrsum1(repuls, Vrep0); 

               VALUETYPE d1 = fast_SM(repuls);
               d1 = -1.0 * STEP * d1; // Y = Y - d * S  => Y += (-d) * S;
               BCL_vset1(Vd1, d1);
	       BCL_vmac(Vy0, Vd1, Vs0); // Vd1 = [-d1, -d1, ... ]
               BCL_vmac(Vy1, Vd1, Vs1); 
               BCL_vmac(Vy2, Vd1, Vs2); 
               BCL_vmac(Vy3, Vd1, Vs3); 
               //BCL_vmac(Vy4, Vd1, Vs4); 
               //BCL_vmac(Vy5, Vd1, Vs5); 
               //BCL_vmac(Vy6, Vd1, Vs6); 
               //BCL_vmac(Vy7, Vd1, Vs7); 
            }
	    BCL_vstu(Y + bindex - baseindex +VLEN*0, Vy0);
            BCL_vstu(Y + bindex - baseindex +VLEN*1, Vy1);
            BCL_vstu(Y + bindex - baseindex +VLEN*2, Vy2);
            BCL_vstu(Y + bindex - baseindex +VLEN*3, Vy3);
            //BCL_vstu(Y + bindex - baseindex +VLEN*4, Vy4);
            //BCL_vstu(Y + bindex - baseindex +VLEN*5, Vy5);
            //BCL_vstu(Y + bindex - baseindex +VLEN*6, Vy6);
            //BCL_vstu(Y + bindex - baseindex +VLEN*7, Vy7);
         }
         }
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] = 
                  prevCoordinates[bindex - baseindex + d];
            }
         }
      }
      INDEXTYPE b = (graph.rows/BATCHSIZE); // can be skipped 
      if (graph.rows > b * BATCHSIZE)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
	    int maxv = graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
	INDEXTYPE INITPREVSIZE = graph.rows;
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    if(i >= graph.rows) continue;
            INDEXTYPE IDIM = i*DIM;
            #pragma omp simd
	    for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
            }
         }
	#pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue; /* get rid of it !!! */
            VALUETYPE forceDiff[DIM];
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
	    INDEXTYPE wlength = 0;
	    INDEXTYPE windex = i;
            VALUETYPE degi = 1.0;
	
            VALUETYPE *Xi = nCoordinates + iindex; 
		degi = 1.0 / (graph.rowptr[i+1] - graph.rowptr[i] + 1);

		for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
	       VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = nCoordinates + jindex; 
 
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {

			attrc += Xi[d] * Xj[d];
		}
		VALUETYPE d1 = fast_SM(attrc);
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
			prevCoordinates[bindex - baseindex + d] += STEP * degi * (1.0 - d1) * Xj[d];
	       }
	       wlength++;
	       windex = colidj;
            }

		for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
	          repuls += this->nCoordinates[iindex + d] * samples[jindex + d];
               }
               VALUETYPE d1 = fast_SM(repuls);
					
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
                  prevCoordinates[bindex - baseindex + d] -= 
                     (STEP) * d1 * samples[jindex + d];
               }
            }
         }
	for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] = 
                  prevCoordinates[bindex - baseindex + d];
            }
         }
      }


      LOOP++;
   }
   end = omp_get_wtime();
   cout << "optForce2VecNSRWLB_AVXZ64 Parallel Wall time required:" << end - start << " seconds" << endl;
   
   result.push_back((float)(end - start));
   writeToFile("F2VWNSLB64_AVXZ"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        
   return result;
}





vector<VALUETYPE> algorithms::AlgoForce2VecNSLB_SREAL_D64_AVXZ
(
 INDEXTYPE ITERATIONS, 
 INDEXTYPE NUMOFTHREADS, 
 INDEXTYPE BATCHSIZE, 
 INDEXTYPE ns, 
 VALUETYPE lr
 )
{
   INDEXTYPE LOOP = 0;
   VALUETYPE STEP = lr, EPS = 0.00001;
   double start, end;
   double loglike0, loglike;
   
   vector<VALUETYPE> result;
   vector<INDEXTYPE> indices;
   vector<INDEXTYPE> nsamples;
   VTYPE VMAXBOUND, VMINBOUND, VEPS, VSTEP;   
   VALUETYPE *samples;
   samples = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
   
   omp_set_num_threads(NUMOFTHREADS);
   start = omp_get_wtime();
   randInitF();
   INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
   VALUETYPE *prevCoordinates;
   int maxdegree = -1;
        
   prevCoordinates = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));

   for(INDEXTYPE i = 0; i < NUMSIZE; i += 1){
		INDEXTYPE IDIM = i*DIM;
                for(INDEXTYPE d = 0; d < this->DIM; d++){
                        prevCoordinates[IDIM + d] = 0;
                }
   }

   BCL_vset1(VMAXBOUND, MAXBOUND); 
   BCL_vset1(VMINBOUND, -MAXBOUND); 

   INDEXTYPE *ThRowId;

ThRowId = (INDEXTYPE *) malloc(sizeof(INDEXTYPE)*(NUMOFTHREADS+1));
   loglike0 = loglike = numeric_limits<double>::max();
   while(LOOP < ITERATIONS)
   {
      loglike0 = loglike;
      loglike = 0;
      for(INDEXTYPE b = 0; b < graph.rows / BATCHSIZE; b += 1)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
	    int maxv = graph.rows-1;
            int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
         INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
	 
	/*..........load balancing(prefix sum)..................*/
   	 INDEXTYPE RowPerThd; 
   	 INDEXTYPE i;
   	 INDEXTYPE nnzB = 0;

	 for (i=b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
      		nnzB += (graph.rowptr[i+1] - graph.rowptr[i]); 
	 RowPerThd = nnzB / NUMOFTHREADS; 
	 ThRowId[0] = b * BATCHSIZE;
	 INDEXTYPE curRow = 0;
	 INDEXTYPE cumRow = 0;
	 INDEXTYPE tt=1;
      	 for(i=b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
      	 {
		INDEXTYPE deg = graph.rowptr[i+1] - graph.rowptr[i];
		cumRow += deg;
		curRow += deg;
		if(curRow > RowPerThd){
			ThRowId[tt] = i;
			curRow = 0;
			RowPerThd = (nnzB - cumRow) / (NUMOFTHREADS - tt);
			tt += 1;
		}
      	 }
	 ThRowId[tt] = (b + 1) * BATCHSIZE;
	 assert(cumRow == nnzB);
	 for(i= tt+1; i< NUMOFTHREADS+1; i += 1){
                ThRowId[i] = 0;
         }
 #pragma omp parallel
	 {
	 INDEXTYPE id = omp_get_thread_num();
         for(INDEXTYPE i = ThRowId[id]; i < ThRowId[id+1]; i += 1)
         {
            VALUETYPE *X = nCoordinates; 
            VALUETYPE *Y = prevCoordinates; 
            VALUETYPE *S = samples; 

            INDEXTYPE iindex = i * DIM;
            VALUETYPE *Xi = X + iindex; 
            INDEXTYPE bindex = i * DIM;
            VALUETYPE degi = 1.0;

            VTYPE Vy0, Vy1, Vy2, Vy3, Vy4, Vy5, Vy6, Vy7; 
            VTYPE Vxi0, Vxi1, Vxi2, Vxi3, Vxi4, Vxi5, Vxi6, Vxi7; 
            BCL_vldu(Vy0, Y + bindex - baseindex +VLEN*0);
            BCL_vldu(Vy1, Y + bindex - baseindex +VLEN*1);
            BCL_vldu(Vy2, Y + bindex - baseindex +VLEN*2);
            BCL_vldu(Vy3, Y + bindex - baseindex +VLEN*3);
            //BCL_vldu(Vy4, Y + bindex - baseindex +VLEN*4);
            //BCL_vldu(Vy5, Y + bindex - baseindex +VLEN*5);
            //BCL_vldu(Vy6, Y + bindex - baseindex +VLEN*6);
            //BCL_vldu(Vy7, Y + bindex - baseindex +VLEN*7);
            
            BCL_vldu(Vxi0, Xi + VLEN*0);
            BCL_vldu(Vxi1, Xi + VLEN*1);
            BCL_vldu(Vxi2, Xi + VLEN*2);
            BCL_vldu(Vxi3, Xi + VLEN*3);
            //BCL_vldu(Vxi4, Xi + VLEN*4);
            //BCL_vldu(Vxi5, Xi + VLEN*5);
            //BCL_vldu(Vxi6, Xi + VLEN*6);
            //BCL_vldu(Vxi7, Xi + VLEN*7);
	
	    VTYPE Vt, Vd0, Vd1, Vd2, Vd3, Vd4, Vd5, Vd6, Vd7;

            for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
               VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = X + jindex; 
        
               VTYPE Vtemp, Vatt0, Vatt1, Vatt2, Vatt3;
               VTYPE Vxj0, Vxj1, Vxj2, Vxj3, Vxj4, Vxj5, Vxj6, Vxj7; 

               BCL_vldu(Vxj0, Xj + VLEN*0);
               BCL_vldu(Vxj1, Xj + VLEN*1);
               BCL_vldu(Vxj2, Xj + VLEN*2);
               BCL_vldu(Vxj3, Xj + VLEN*3);
               //BCL_vldu(Vxj4, Xj + VLEN*4);
               //BCL_vldu(Vxj5, Xj + VLEN*5);
               //BCL_vldu(Vxj6, Xj + VLEN*6);
               //BCL_vldu(Vxj7, Xj + VLEN*7);
               
	       BCL_vzero(Vatt0); 
	       BCL_vzero(Vatt1);
	       BCL_vzero(Vatt2);
	       BCL_vzero(Vatt3);
         
               BCL_vsub(Vd0, Vxi0, Vxj0);
               BCL_vsub(Vd1, Vxi1, Vxj1);
               BCL_vsub(Vd2, Vxi2, Vxj2);
               BCL_vsub(Vd3, Vxi3, Vxj3);
               //BCL_vsub(Vd4, Vxi4, Vxj4);
               //BCL_vsub(Vd5, Vxi5, Vxj5);
               //BCL_vsub(Vd6, Vxi6, Vxj6);
               //BCL_vsub(Vd7, Vxi7, Vxj7);

	       BCL_vmac(Vatt0, Vd0, Vd0);
               BCL_vmac(Vatt1, Vd1, Vd1);
               BCL_vmac(Vatt2, Vd2, Vd2);
               BCL_vmac(Vatt3, Vd3, Vd3);
               //BCL_vmac(Vatt0, Vd4, Vd4);
               //BCL_vmac(Vatt1, Vd5, Vd5);
               //BCL_vmac(Vatt2, Vd6, Vd6);
               //BCL_vmac(Vatt3, Vd7, Vd7);

               BCL_vadd(Vatt0, Vatt0, Vatt1);
               BCL_vadd(Vatt2, Vatt2, Vatt3);
               BCL_vadd(Vatt0, Vatt0, Vatt2);
         
               BCL_vrsum1(attrc, Vatt0);

	       BCL_vset1(Vatt0, attrc); // a = a
	       BCL_vset1(Vt, 1.0f); // t = 1
	       BCL_vadd(Vatt0, Vatt0, Vt); // a = t + a

	       BCL_vrcp(Vatt0, Vatt0); // a = 1/a
	       BCL_vset1(Vt, -2.0f); // t = -2
	       BCL_vmul(Vatt0, Vatt0, Vt); // a = -2 * a
               
               BCL_vmul(Vd0, Vatt0, Vd0); 
	       BCL_vmul(Vd1, Vatt0, Vd1);  
	       BCL_vmul(Vd2, Vatt0, Vd2);  
 	       BCL_vmul(Vd3, Vatt0, Vd3);  
 	       //BCL_vmul(Vd4, Vatt0, Vd4);  
               //BCL_vmul(Vd5, Vatt0, Vd5);
               //BCL_vmul(Vd6, Vatt0, Vd6);
               //BCL_vmul(Vd7, Vatt0, Vd7);               

	       BCL_vmax(Vd0, Vd0, VMINBOUND);
	       BCL_vmax(Vd1, Vd1, VMINBOUND); 
	       BCL_vmax(Vd2, Vd2, VMINBOUND);  
               BCL_vmax(Vd3, Vd3, VMINBOUND);
               //BCL_vmax(Vd4, Vd4, VMINBOUND);  
               //BCL_vmax(Vd5, Vd5, VMINBOUND);
               //BCL_vmax(Vd6, Vd6, VMINBOUND);
               //BCL_vmax(Vd7, Vd7, VMINBOUND);

	       BCL_vmin(Vd0, Vd0, VMAXBOUND);
               BCL_vmin(Vd1, Vd1, VMAXBOUND);
               BCL_vmin(Vd2, Vd2, VMAXBOUND);
               BCL_vmin(Vd3, Vd3, VMAXBOUND);
               //BCL_vmin(Vd4, Vd4, VMAXBOUND);
               //BCL_vmin(Vd5, Vd5, VMAXBOUND);
               //BCL_vmin(Vd6, Vd6, VMAXBOUND);
               //BCL_vmin(Vd7, Vd7, VMAXBOUND);
	
	       BCL_vset1(Vt, STEP);
	     
               BCL_vmac(Vy0, Vd0, Vt); 
               BCL_vmac(Vy1, Vd1, Vt); 
               BCL_vmac(Vy2, Vd2, Vt); 
               BCL_vmac(Vy3, Vd3, Vt); 
               //BCL_vmac(Vy4, Vd4, Vt); 
               //BCL_vmac(Vy5, Vd5, Vt); 
               //BCL_vmac(Vy6, Vd6, Vt); 
               //BCL_vmac(Vy7, Vd7, Vt); 
            }
	
	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               VTYPE Vs0, Vs1, Vs2, Vs3, Vs4, Vs5, Vs6, Vs7;
               VTYPE Vd1;
               VTYPE Vrep0, Vrep1, Vrep2, Vrep3;  
      
               BCL_vldu(Vs0, S + jindex + VLEN*0);
               BCL_vldu(Vs1, S + jindex + VLEN*1);
               BCL_vldu(Vs2, S + jindex + VLEN*2);
               BCL_vldu(Vs3, S + jindex + VLEN*3);
               //BCL_vldu(Vs4, S + jindex + VLEN*4);
               //BCL_vldu(Vs5, S + jindex + VLEN*5);
               //BCL_vldu(Vs6, S + jindex + VLEN*6);
               //BCL_vldu(Vs7, S + jindex + VLEN*7);

	       BCL_vzero(Vrep0); 
               BCL_vzero(Vrep1); 
               BCL_vzero(Vrep2); 
               BCL_vzero(Vrep3); 

	       BCL_vsub(Vd0, Vxi0, Vs0);
               BCL_vsub(Vd1, Vxi1, Vs1);
               BCL_vsub(Vd2, Vxi2, Vs2);
               BCL_vsub(Vd3, Vxi3, Vs3);
               //BCL_vsub(Vd4, Vxi4, Vs4);
               //BCL_vsub(Vd5, Vxi5, Vs5);
               //BCL_vsub(Vd6, Vxi6, Vs6);
               //BCL_vsub(Vd7, Vxi7, Vs7);

               BCL_vmac(Vrep0, Vd0, Vd0);
               BCL_vmac(Vrep1, Vd1, Vd1);
               BCL_vmac(Vrep2, Vd2, Vd2);
               BCL_vmac(Vrep3, Vd3, Vd3);
               //BCL_vmac(Vrep0, Vd4, Vd4);
               //BCL_vmac(Vrep1, Vd5, Vd5);
               //BCL_vmac(Vrep2, Vd6, Vd6);
               //BCL_vmac(Vrep3, Vd7, Vd7);

               BCL_vadd(Vrep0, Vrep0, Vrep1);
               BCL_vadd(Vrep2, Vrep2, Vrep3);
               BCL_vadd(Vrep0, Vrep0, Vrep2);

               BCL_vrsum1(repuls, Vrep0);
	       
	       BCL_vset1(Vrep0, repuls); // a = a
               BCL_vset1(Vt, 1.0f); // t = 1
               BCL_vadd(Vt, Vrep0, Vt); // t = t + a
	       BCL_vmul(Vrep0, Vrep0, Vt); // a = t * a

               BCL_vrcp(Vrep0, Vrep0); // a = 1/a
               BCL_vset1(Vt, 2.0f); // t = 2
               BCL_vmul(Vrep0, Vrep0, Vt); // a = 2 * a

               BCL_vmul(Vd0, Vrep0, Vd0);
               BCL_vmul(Vd1, Vrep0, Vd1);
               BCL_vmul(Vd2, Vrep0, Vd2);
               BCL_vmul(Vd3, Vrep0, Vd3);
               //BCL_vmul(Vd4, Vrep0, Vd4);
               //BCL_vmul(Vd5, Vrep0, Vd5);
               //BCL_vmul(Vd6, Vrep0, Vd6);
               //BCL_vmul(Vd7, Vrep0, Vd7);

               BCL_vmax(Vd0, Vd0, VMINBOUND);
               BCL_vmax(Vd1, Vd1, VMINBOUND);
               BCL_vmax(Vd2, Vd2, VMINBOUND);
               BCL_vmax(Vd3, Vd3, VMINBOUND);
               //BCL_vmax(Vd4, Vd4, VMINBOUND);
               //BCL_vmax(Vd5, Vd5, VMINBOUND);
               //BCL_vmax(Vd6, Vd6, VMINBOUND);
               //BCL_vmax(Vd7, Vd7, VMINBOUND);

               BCL_vmin(Vd0, Vd0, VMAXBOUND);
               BCL_vmin(Vd1, Vd1, VMAXBOUND);
               BCL_vmin(Vd2, Vd2, VMAXBOUND);
               BCL_vmin(Vd3, Vd3, VMAXBOUND);
               //BCL_vmin(Vd4, Vd4, VMAXBOUND);
               //BCL_vmin(Vd5, Vd5, VMAXBOUND);
               //BCL_vmin(Vd6, Vd6, VMAXBOUND);
               //BCL_vmin(Vd7, Vd7, VMAXBOUND);

               BCL_vset1(Vt, STEP);
	     
               BCL_vmac(Vy0, Vd0, Vt);
               BCL_vmac(Vy1, Vd1, Vt);
               BCL_vmac(Vy2, Vd2, Vt);
               BCL_vmac(Vy3, Vd3, Vt);
               //BCL_vmac(Vy4, Vd4, Vt);
               //BCL_vmac(Vy5, Vd5, Vt);
               //BCL_vmac(Vy6, Vd6, Vt);
               //BCL_vmac(Vy7, Vd7, Vt);


            }
	    BCL_vstu(Y + bindex - baseindex +VLEN*0, Vy0);
            BCL_vstu(Y + bindex - baseindex +VLEN*1, Vy1);
            BCL_vstu(Y + bindex - baseindex +VLEN*2, Vy2);
            BCL_vstu(Y + bindex - baseindex +VLEN*3, Vy3);
            //BCL_vstu(Y + bindex - baseindex +VLEN*4, Vy4);
            //BCL_vstu(Y + bindex - baseindex +VLEN*5, Vy5);
            //BCL_vstu(Y + bindex - baseindex +VLEN*6, Vy6);
            //BCL_vstu(Y + bindex - baseindex +VLEN*7, Vy7);
         }
         }  
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] += 
                  prevCoordinates[bindex - baseindex + d];
		prevCoordinates[bindex - baseindex + d] = 0;
            }
         }
      }
	INDEXTYPE b = (graph.rows/BATCHSIZE); // can be skipped 
	if (graph.rows > b * BATCHSIZE)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;
	    int maxv = graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
	 INDEXTYPE INITPREVSIZE = graph.rows;
	 #pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    VALUETYPE forceDiff[DIM];
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
	    INDEXTYPE wlength = 0;
	    INDEXTYPE windex = i;
            VALUETYPE degi = 1.0;
	
            VALUETYPE *Xi = nCoordinates + iindex; 
  	    for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
            {
	       VALUETYPE attrc = 0;
               INDEXTYPE colidj = graph.colids[j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = nCoordinates + jindex; 
	       VALUETYPE forceDiff[DIM];
 
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
			forceDiff[d] = Xj[d] - Xi[d];
			attrc += forceDiff[d] * forceDiff[d];
		}
	       VALUETYPE d1 = -2.0 / (1.0 + attrc);
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
			forceDiff[d] = scale(forceDiff[d] * d1);
			prevCoordinates[bindex - baseindex + d] += STEP * forceDiff[d];
		 }
            }
	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
	           forceDiff[d] = this->nCoordinates[iindex + d] - samples[jindex + d];
		  repuls += forceDiff[d] * forceDiff[d];
               }
               VALUETYPE d1 = 2.0 / ((repuls) * (1.0 + repuls));
					
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
		 	forceDiff[d] = scale(forceDiff[d] * d1);
                  	prevCoordinates[bindex - baseindex + d] +=  (STEP) * forceDiff[d];
               }
            }
         }
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] += prevCoordinates[bindex - baseindex + d];
            	prevCoordinates[bindex - baseindex + d] = 0;
	    }
         }
      }

    LOOP++;
   }
	
   end = omp_get_wtime();
   cout << "optForce2VecNSLB_AVXZ64 Parallel Wall time required:" << end - start << " seconds" << endl;
   
   result.push_back((float)(end - start));
   writeToFile("F2VNSLB_AVXZ64"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        
   return result;
		
}




vector<VALUETYPE> algorithms::AlgoForce2VecNSRWEFF_SREAL_D64_AVXZ
(
 INDEXTYPE ITERATIONS, 
 INDEXTYPE NUMOFTHREADS, 
 INDEXTYPE BATCHSIZE, 
 INDEXTYPE ns, 
 VALUETYPE lr
 )
{
   INDEXTYPE LOOP = 0;
   VALUETYPE STEP = lr, EPS = 0.00001;
   double start, end;
   double loglike0, loglike;
   
   vector<VALUETYPE> result;
   vector<INDEXTYPE> indices;
   vector<INDEXTYPE> nsamples;
   
   VALUETYPE *samples;
   INDEXTYPE *walksamples; 
   INDEXTYPE WALKLENGTH = 5;
   samples = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[DIM * ns])));
   walksamples = static_cast<INDEXTYPE *> (::operator new (sizeof(INDEXTYPE[WALKLENGTH * graph.rows])));
   unsigned long long x = time(nullptr);
   for (int i = 0; i < 2; i++) 
   {
      unsigned long long z = x += UINT64_C(0x9E3779B97F4A7C15);
      z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
      z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
      rseed[i] = z ^ z >> 31;
   }
	
   init_SM_TABLE();
   omp_set_num_threads(NUMOFTHREADS);
   start = omp_get_wtime();
   randInit();
   INDEXTYPE NUMSIZE = min(BATCHSIZE, graph.rows);
   VALUETYPE *prevCoordinates;
   int maxdegree = -1;
        
   prevCoordinates = 
      static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[NUMSIZE * DIM])));
   
   loglike0 = loglike = numeric_limits<double>::max();
   
   while(LOOP < ITERATIONS)
   {
      loglike0 = loglike;
      loglike = 0;

      for(INDEXTYPE i = 0; i < graph.rows; i++){
      	 INDEXTYPE wlength = 0;
         INDEXTYPE windex = i;
         while(wlength < WALKLENGTH){
         	INDEXTYPE j = windex;
                if(graph.rowptr[windex+1] - graph.rowptr[windex] > 2)
                	j = randIndex(graph.rowptr[windex+1]-1, graph.rowptr[windex]);
                else if(graph.rowptr[windex+1] - graph.rowptr[windex] == 2)
                	j = graph.rowptr[windex];
              	walksamples[i*WALKLENGTH + wlength] = graph.colids[j];
               	wlength++;
                windex = graph.colids[j];
          }
        }	

	for(INDEXTYPE b = 0; b < graph.rows / BATCHSIZE; b += 1)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
            int maxv = min((b+1) * BATCHSIZE, graph.rows-1);//graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
  	 INDEXTYPE INITPREVSIZE = min(b * BATCHSIZE, graph.rows);
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            INDEXTYPE IDIM = i*DIM;
            #pragma omp simd
	    for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
            }
         }
 	 #pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
		VALUETYPE *X = nCoordinates; 
            VALUETYPE *Y = prevCoordinates; 
            VALUETYPE *S = samples; 

            INDEXTYPE iindex = i * DIM;
            VALUETYPE *Xi = X + iindex; 
            INDEXTYPE bindex = i * DIM;
            VALUETYPE degi = 1.0;
	
            VTYPE Vy0, Vy1, Vy2, Vy3, Vy4, Vy5, Vy6, Vy7, Vy8; 
            VTYPE Vxi0, Vxi1, Vxi2, Vxi3, Vxi4, Vxi5, Vxi6, Vxi7, Vxi8; 
	    
	    BCL_vldu(Vy0, Y + bindex - baseindex +VLEN*0);
            BCL_vldu(Vy1, Y + bindex - baseindex +VLEN*1);
            BCL_vldu(Vy2, Y + bindex - baseindex +VLEN*2);
            BCL_vldu(Vy3, Y + bindex - baseindex +VLEN*3);
            //BCL_vldu(Vy4, Y + bindex - baseindex +VLEN*4);
            //BCL_vldu(Vy5, Y + bindex - baseindex +VLEN*5);
            //BCL_vldu(Vy6, Y + bindex - baseindex +VLEN*6);
            //BCL_vldu(Vy7, Y + bindex - baseindex +VLEN*7);

   	    BCL_vldu(Vxi0, Xi + VLEN*0);
            BCL_vldu(Vxi1, Xi + VLEN*1);
            BCL_vldu(Vxi2, Xi + VLEN*2);
            BCL_vldu(Vxi3, Xi + VLEN*3);
            //BCL_vldu(Vxi4, Xi + VLEN*4);
            //BCL_vldu(Vxi5, Xi + VLEN*5);
            //BCL_vldu(Vxi6, Xi + VLEN*6);
            //BCL_vldu(Vxi7, Xi + VLEN*7);

	    for(INDEXTYPE j = 0; j < WALKLENGTH; j += 1)
            {
               VALUETYPE attrc = 0;
               INDEXTYPE colidj = walksamples[i * WALKLENGTH + j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = X + jindex; 
        
               VTYPE Vd1;
               VTYPE Vatt0, Vatt1, Vatt2, Vatt3;  
               VTYPE Vxj0, Vxj1, Vxj2, Vxj3, Vxj4, Vxj5, Vxj6, Vxj7;

		BCL_vldu(Vxj0, Xj + VLEN*0);
               BCL_vldu(Vxj1, Xj + VLEN*1);
               BCL_vldu(Vxj2, Xj + VLEN*2);
               BCL_vldu(Vxj3, Xj + VLEN*3);
               //BCL_vldu(Vxj4, Xj + VLEN*4);
               //BCL_vldu(Vxj5, Xj + VLEN*5);
               //BCL_vldu(Vxj6, Xj + VLEN*6);
               //BCL_vldu(Vxj7, Xj + VLEN*7);

		BCL_vzero(Vatt0); 
               BCL_vzero(Vatt1); 
               BCL_vzero(Vatt2); 
               BCL_vzero(Vatt3); 
         
               BCL_vmac(Vatt0, Vxi0, Vxj0);
               BCL_vmac(Vatt1, Vxi1, Vxj1);
               BCL_vmac(Vatt2, Vxi2, Vxj2);
               BCL_vmac(Vatt3, Vxi3, Vxj3);
         
               //BCL_vmac(Vatt0, Vxi4, Vxj4);
               //BCL_vmac(Vatt1, Vxi5, Vxj5);
               //BCL_vmac(Vatt2, Vxi6, Vxj6);
               //BCL_vmac(Vatt3, Vxi7, Vxj7);
      
               BCL_vadd(Vatt0, Vatt0, Vatt1);
               BCL_vadd(Vatt2, Vatt2, Vatt3);
               BCL_vadd(Vatt0, Vatt0, Vatt2);
         
               BCL_vrsum1(attrc, Vatt0);

		VALUETYPE d1 = fast_SM(attrc);
               d1 = STEP * degi * (1.0 - d1); 
               BCL_vset1(Vd1, d1);

		BCL_vmac(Vy0, Vd1, Vxj0); 
               BCL_vmac(Vy1, Vd1, Vxj1); 
               BCL_vmac(Vy2, Vd1, Vxj2); 
               BCL_vmac(Vy3, Vd1, Vxj3); 
               //BCL_vmac(Vy4, Vd1, Vxj4); 
               //BCL_vmac(Vy5, Vd1, Vxj5); 
               //BCL_vmac(Vy6, Vd1, Vxj6); 
               //BCL_vmac(Vy7, Vd1, Vxj7); 
            }

	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               VTYPE Vs0, Vs1, Vs2, Vs3, Vs4, Vs5, Vs6, Vs7;
               VTYPE Vd1;
               VTYPE Vrep0, Vrep1, Vrep2, Vrep3;  
      
               BCL_vldu(Vs0, S + jindex + VLEN*0);
               BCL_vldu(Vs1, S + jindex + VLEN*1);
               BCL_vldu(Vs2, S + jindex + VLEN*2);
               BCL_vldu(Vs3, S + jindex + VLEN*3);
               //BCL_vldu(Vs4, S + jindex + VLEN*4);
               //BCL_vldu(Vs5, S + jindex + VLEN*5);
               //BCL_vldu(Vs6, S + jindex + VLEN*6);
               //BCL_vldu(Vs7, S + jindex + VLEN*7);

		BCL_vzero(Vrep0); 
               BCL_vzero(Vrep1); 
               BCL_vzero(Vrep2); 
               BCL_vzero(Vrep3); 
         
               BCL_vmac(Vrep0, Vxi0, Vs0);
               BCL_vmac(Vrep1, Vxi1, Vs1);
               BCL_vmac(Vrep2, Vxi2, Vs2);
               BCL_vmac(Vrep3, Vxi3, Vs3);
         
               //BCL_vmac(Vrep0, Vxi4, Vs4);
               //BCL_vmac(Vrep1, Vxi5, Vs5);
               //BCL_vmac(Vrep2, Vxi6, Vs6);
               //BCL_vmac(Vrep3, Vxi7, Vs7);
      
               BCL_vadd(Vrep0, Vrep0, Vrep1);
               BCL_vadd(Vrep2, Vrep2, Vrep3);
               BCL_vadd(Vrep0, Vrep0, Vrep2);
         
               BCL_vrsum1(repuls, Vrep0); 

               VALUETYPE d1 = fast_SM(repuls);
               d1 = -1.0 * STEP * d1; // Y = Y - d * S  => Y += (-d) * S;
               BCL_vset1(Vd1, d1);

		BCL_vmac(Vy0, Vd1, Vs0); // Vd1 = [-d1, -d1, ... ]
               BCL_vmac(Vy1, Vd1, Vs1); 
               BCL_vmac(Vy2, Vd1, Vs2); 
               BCL_vmac(Vy3, Vd1, Vs3); 
               //BCL_vmac(Vy4, Vd1, Vs4); 
               //BCL_vmac(Vy5, Vd1, Vs5); 
               //BCL_vmac(Vy6, Vd1, Vs6); 
               //BCL_vmac(Vy7, Vd1, Vs7); 
            }

	    BCL_vstu(Y + bindex - baseindex +VLEN*0, Vy0);
            BCL_vstu(Y + bindex - baseindex +VLEN*1, Vy1);
            BCL_vstu(Y + bindex - baseindex +VLEN*2, Vy2);
            BCL_vstu(Y + bindex - baseindex +VLEN*3, Vy3);
            //BCL_vstu(Y + bindex - baseindex +VLEN*4, Vy4);
            //BCL_vstu(Y + bindex - baseindex +VLEN*5, Vy5);
            //BCL_vstu(Y + bindex - baseindex +VLEN*6, Vy6);
            //BCL_vstu(Y + bindex - baseindex +VLEN*7, Vy7);
         }
         
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] = 
                  prevCoordinates[bindex - baseindex + d];
            }
         }
      }
	INDEXTYPE b = (graph.rows/BATCHSIZE); // can be skipped 
      if (graph.rows > b * BATCHSIZE)
      {
         INDEXTYPE baseindex = b * BATCHSIZE * DIM;
	 
         #pragma omp simd
	 for(INDEXTYPE s = 0; s < ns; s++)
         {
            int minv = 0;//b * BATCHSIZE;
	    int maxv = graph.rows-1;
	    int rindex = randIndex(maxv, minv);
	    int randombaseindex = rindex * DIM;
            int sindex = s * DIM;
            for(INDEXTYPE d = 0; d < DIM; d++)
            {
               samples[sindex + d] = nCoordinates[randombaseindex + d];
            }
	 }
	 INDEXTYPE INITPREVSIZE = graph.rows;
	 for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
	    if(i >= graph.rows) continue;
            INDEXTYPE IDIM = i*DIM;
            #pragma omp simd
	    for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               prevCoordinates[IDIM - baseindex + d] = nCoordinates[IDIM + d];
            }
         }
	 #pragma omp parallel for schedule(static)
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue; /* get rid of it !!! */
            VALUETYPE forceDiff[DIM];
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
	    INDEXTYPE wlength = 0;
	    INDEXTYPE windex = i;
            VALUETYPE degi = 1.0;
	
            VALUETYPE *Xi = nCoordinates + iindex; 

	    for(INDEXTYPE j = 0; j < WALKLENGTH; j += 1)
            {
	       VALUETYPE attrc = 0;
               INDEXTYPE colidj =  walksamples[i * WALKLENGTH + j];
               INDEXTYPE jindex = colidj*DIM;
               VALUETYPE *Xj = nCoordinates + jindex; 
 
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {

			attrc += Xi[d] * Xj[d];
		}

		VALUETYPE d1 = fast_SM(attrc);
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
			prevCoordinates[bindex - baseindex + d] += STEP * degi * (1.0 - d1) * Xj[d];
	       }
	       wlength++;
	       windex = colidj;
            }
	    for(INDEXTYPE j = 0; j < ns; j += 1)
            {
               VALUETYPE repuls = 0;
               INDEXTYPE jindex = j * DIM;
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
	          repuls += this->nCoordinates[iindex + d] * samples[jindex + d];
               }
               VALUETYPE d1 = fast_SM(repuls);
					
               for(INDEXTYPE d = 0; d < this->DIM; d++)
               {
                  prevCoordinates[bindex - baseindex + d] -= 
                     (STEP) * d1 * samples[jindex + d];
               }
            }
         }
         
         for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
         {
            if(i >= graph.rows) continue;
	    INDEXTYPE iindex = i * DIM;
            INDEXTYPE bindex = i * DIM;
            #pragma omp simd
            for(INDEXTYPE d = 0; d < this->DIM; d++)
            {
               this->nCoordinates[iindex + d] = 
                  prevCoordinates[bindex - baseindex + d];
            }
         }
      }


      LOOP++;
   }
   end = omp_get_wtime();
   cout << "optForce2VecNSRWEFF_AVXZ64 Parallel Wall time required:" << end - start << " seconds" << endl;
   
   result.push_back((float)(end - start));
   writeToFile("F2VWEFFNS_AVXZ64"+ to_string(BATCHSIZE)+"D" + to_string(this->DIM)+"IT" + to_string(LOOP)+"NS"+to_string(ns));
        
   return result;
}
#endif 
