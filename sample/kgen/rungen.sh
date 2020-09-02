#!/bin/bash 

gendir=kernels
#default values of parameters 
PRE=s 
VLEN=4 
SDIM=4 
EDIM=128 
FORCE=t  

#commandline argument 

usage="Usage: $0 [OPTION] ... 
Options: 
-v [val] 	value of vector width (vlen), see simd.h to find system vlen
-f [t,s]	force calc formula, t for t-dist and s for sigmoid
-s [val]	starting value of dimension, must be multiple of vlen
-e [val]	ending value of dimension, must be multiple of vlen
-p [s.d]	precision of floating point, s for single precision, d for double precision
--help 		display help and exit 
"

while getopts "v:f:s:e:p:" opt
do
   case $opt in 
      v) 
         VLEN=$OPTARG
         ;; 
      f) 
         FORCE=$OPTARG
         ;; 
      s) 
         SDIM=$OPTARG
         ;; 
      e) 
         EDIM=$OPTARG
         ;; 
      p) 
         PRE=$OPTARG
         ;;
      \?)
         echo "$usage"
         exit 1 
         ;;
   esac
done

mkdir -p $gendir 
make clean kdir=$gendir 

# generate all kernels 
echo "Generating kernels in directory: " $gendir 
echo "===========================================" 
for (( d=$SDIM; d <= $EDIM; d=$d+$VLEN ))
{
   make pre=$PRE force=$FORCE vlen=$VLEN dim=$d kdir=$gendir 
}
