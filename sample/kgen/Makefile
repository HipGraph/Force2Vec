
CC = gcc
CFLAGS = -O3 -w 

# precision float=s, double=d 
pre = s

# See simd.h file to get correct vlen for the system and precision 
# example: for AVX512: single precision vlen=16, double vlen = 8 
vlen = 16

# dimension 
dim = 128

# frc calc method 
frout=frc_tdist
ifeq ($(force), s)
   frout=frc_sigmoid
endif 

kdir=.

all: $(kdir)/g$(pre)$(frout)_D$(dim).c  

xextract: extract.c 
	$(CC) $(CFLAGS) -o $@ extract.c 

$(kdir)/g$(pre)$(frout)_D$(dim).c : xextract genDimFrc.base
	./xextract -b genDimFrc.base -langC -def DIM $(dim) pre=$(pre) \
	   -def VLEN $(vlen) rout=$(frout) -o $@  

clean:
	rm -f $(kdir)/g*.c $(kdir)/g*.o  

kill:
	rm -f xextract   
