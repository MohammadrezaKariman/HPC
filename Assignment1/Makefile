#  makefile for a  cu-programm
#
# use NVCC-Compiler tools
COMPILER=NVCC_
# alternatively from the shell
# export COMPILER=NVCC_
# or, alternatively from the shell
# make COMPILER=NVCC_

# skalar_{0,1,2} are peliminary versions od the inner produkt in CUDA
VERSION = main
# fastest CUDA version
#VERSION = skalar_3_fast
# Using CuBLAS
#VERSION = skalar_4
# Using thrust  (STL on GPU)
#VERSION = skalar_5
#
SOURCES=  $(VERSION).cu
OBJECTS = $(SOURCES:%.cu=%.o)

PROGRAM	= main.${COMPILER}

# uncomment the next to lines for debugging and detailed performance analysis
#ARCHES :=-gencode arch=compute_70,code=\"compute_70,sm_70\" \
#		-gencode arch=compute_75,code=\"compute_75,sm_75\" \
#		-gencode arch=compute_80,code=\"compute_80,sm_80\"
		
#CXXFLAGS += ${ARCHES}
#-g -G -lineinfo
LINKFLAGS +=
#-g -G -lineinfo

include ${COMPILER}default.mk

#
#  Remark on PGI
#
# pgCC -Mcudax86 skalar_3.cu
#
# should compiles CUDA with PGI compiler for x86
#
#  pgCC -Mcudax86 -D__GNUC__ skalar_3.cu
