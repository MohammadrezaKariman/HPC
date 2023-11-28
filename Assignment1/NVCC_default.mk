# Basic Defintions for using GNU-compiler suite sequentially
# requires setting of COMPILER=GCC_

ifeq ($(UBUNTU),1)
# on UBUNTU
CUDABIN = 
#/usr/local/cuda-11/bin/
else
# on manjaro
CUDABIN = 
endif

CXX     = $(CUDABIN)nvcc
#CXX     = $(CUDABIN)nvcc -ccbin /usr/bin/g++-10
#F77	= gfortran
LINKER  = ${CXX}


#WARNINGS = -Wall -Weffc++ -Woverloaded-virtual -W -Wfloat-equal -Wshadow \
#           -Wredundant-decls -Winline
#  -Wunreachable-code
WARNINGS =  --resource-usage -src-in-ptx --restrict --Wreorder --ftemplate-backtrace-limit 1
#WARNINGS +=  -sp-bound-check -warn-double-usage -warn-lmem-usage -warn-spills -res-usage
#            |--CUDA 7.5   |-- slow !!
# CXXFLAGS += -DNDEBUG ${WARNINGS}

## CUDA
PERF = -use_fast_math -restrict
#DEBUG = -G
#DEBUG = -lineinfo

#OPT_GPU = -Xptxas --allow-expensive-optimizations
#OPT_GPU = --gpu-architecture compute_62
# A100
#OPT_GPU = --gpu-architecture compute_80

CXXFLAGS += -O3  -std=c++17 --expt-relaxed-constexpr ${PERF}  --ptxas-options=-v ${OPT_GPU} ${WARNINGS} -I$(SDK_HOME)/inc
#CXXFLAGS += ${PERF} 

CUDA_LIBS += -lcudart -lcublas
#CUDA_LIBS += -L/opt/cuda/lib64 -lcudart -lcusolver_lapack_static -lcublas_static -lcublasLt_static -lcudart 


## OpenMP
# https://stackoverflow.com/questions/3211614/using-openmp-in-the-cuda-host-code
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options
# https://gcc.gnu.org/wiki/FloatingPointMath
CXXFLAGS += --compiler-options=-fopenmp,-O3,-funsafe-math-optimizations
#CXXFLAGS += -Xcompiler -fopenmp,-O3
#-fopenmp -lgomp 
LINKFLAGS += --compiler-options=-fopenmp,-O3,-funsafe-math-optimizations
LINKFLAGS   += -lgomp

## BLAS, LAPACK
LINKFLAGS   += -llapack -lblas
ifeq ($(UBUNTU),0)
# on archlinux/manjaro
LINKFLAGS += -lcblas
else
# on  Ubuntu
endif
LINKFLAGS += -lm ${BLAS} ${CUDA_LIBS}

default:	${PROGRAM}

${PROGRAM}:	${OBJECTS}
	$(LINKER)  $^  ${LINKFLAGS} -o $@

clean::
	@rm -f ${PROGRAM} ${OBJECTS}

clean_all:: clean
	-@rm -f *_ *~ *.bak *.log *.out *.tar *.orig
	-@rm -rf html

run: clean ${PROGRAM}
	${OPTIRUN} ./${PROGRAM}

# tar the current directory
MY_DIR = `basename ${PWD}`
tar: clean_all
	@echo "Tar the directory: " ${MY_DIR}
	@cd .. ;\
	tar cf ${MY_DIR}.tar ${MY_DIR} *default.mk ;\
	cd ${MY_DIR}
# 	tar cf `basename ${PWD}`.tar *

zip: clean_all
	@echo "Zip the directory: " ${MY_DIR}
	@cd .. ;\
	zip ${MY_DIR}.zip ${MY_DIR}/* *default.mk ;\
	cd ${MY_DIR}

doc:
	doxygen Doxyfile

info:
	inxi -C
	lspci | grep NVIDIA
	nsys status --environment	
#	nvidia-smi topo -m
	nvidia-smi
	nvcc -V

#########################################################################
.PRECIOUS: .cu .h
.SUFFIXES: .cu .h .o

.cu.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

.f.o:
	$(F77) -c $(FFLAGS) -o $@ $<

##################################################################################################
# NVIDIA tools in Ubuntu:
#    sudo apt install openjdk-8-jdk
#    nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java ./main.NVCC_ &
ifeq ($(UBUNTU),1)
# on UBUNTU
JDK8 = -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java 
endif


# Check for wrong memory accesses, memory leaks, ...
# use smaller data sets
#  CXXFLAGS += -g -G
#  LINKFLAGS += -pg
cache: ${PROGRAM}
	${OPTIRUN} nvprof ${JDK8} --print-gpu-trace  ./$^ > out_prof.txt
	#${OPTIRUN} nvprof --events l1_global_load_miss,l1_local_load_miss  ./$^ > out_prof.txt

mem: ${PROGRAM}
	${OPTIRUN} cuda-memcheck ./$^

#  Simple run time profiling of your code
#  CXXFLAGS  += -g -G -lineinfo
#  LINKFLAGS += -g -G -lineinfo
#  See also https://docs.nvidia.com/cuda/profiler-users-guide/index.html

NVVP_FLAGS = -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
prof: ${PROGRAM}
	${OPTIRUN} ./$^
	${OPTIRUN} $(CUDABIN)nvvp  ${NVVP_FLAGS} ./$^ &

# see also   https://gist.github.com/sonots/5abc0bccec2010ac69ff74788b265086
prof2: ${PROGRAM}
	$(CUDABIN)nvprof --print-gpu-trace ./$^ 2> prof2.txt

NSYS_OPTIONS = profile --trace=cublas,cuda  --sample=none --cuda-memory-usage=true --cudabacktrace=all --stats=true
# https://docs.nvidia.com/nsight-systems/UserGuide/index.html
prof3: ${PROGRAM}
	$(CUDABIN)nsys $(NSYS_OPTIONS) ./$^ 
	$(CUDABIN)nsys-ui `ls -1tr  report*.*rep|tail -1`  &
	
prof4: ${PROGRAM}
	$(CUDABIN)nsys-ui ./$^ 
	
prof5: ${PROGRAM}
#	$(CUDABIN)cu --kernel-name scalar --launch-skip 1 --launch-count 1 "./$^"
	sudo /usr/local/cuda/bin/ncu --kernel-name scalar --launch-skip 1 --launch-count 1 "./main.NVCC_"
	
top:
	nvtop

	



