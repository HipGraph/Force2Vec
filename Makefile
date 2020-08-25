Force2Vec = .
SAMPLE = ./sample
UNITTEST = ./Test
BIN = ./bin
INCDIR = -I$(Force2Vec) -I$(SAMPLE) -I$(UNITTEST)

COMPILER = g++

FLAGS = -g -fomit-frame-pointer -ffast-math -fopenmp -O3 -std=c++11 -DCPP 

ifeq ($(AVX512), true)
   FLAGS += -mavx512f -mavx512dq -DAVX512=1
endif

all: force2vec

algorithms.o:	$(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h $(SAMPLE)/IO.h $(SAMPLE)/CSR.h $(SAMPLE)/CSC.h
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/algorithms.o $(SAMPLE)/algorithms.cpp

force2vec.o:  $(UNITTEST)/Force2Vec.cpp $(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/force2vec.o $(UNITTEST)/Force2Vec.cpp

force2vec:    algorithms.o force2vec.o
		$(COMPILER) $(INCDIR) $(FLAGS) -o $(BIN)/Force2Vec $(BIN)/algorithms.o $(BIN)/force2vec.o

clean:
	rm -rf ./bin/*
	rm -rf Results.txt
