#makefile

PYTHON=python3
CC=g++
C=gcc-7
NVCC=nvcc
CUDA_DIR=/usr/local/cuda

CUDAFLAG=-DGOOGLE_CUDA
CUDA_I=-I${CUDA_DIR}/include
CUDA_L=${CUDA_DIR}/lib64/libcudart.so ${CUDA_DIR}/lib64/libcublas.so

CFLAGS := -std=c++11 $(CUDA_I) $(CUDAFLAG) -I../CPP -fPIC -shared 
CIFLAGS := -std=c++11 $(CUDA_I) $(CUDAFLAG) -I../CPP -fPIC -c
CUFLAGS = -std=c++11 -c $(CUDAFLAG) --expt-relaxed-constexpr -x cu -Xcompiler -fPIC -ccbin $(C) -DNDEBUG
LFLAGS := $(CUDA_L)


binary_mean_pass_solver = ../CPP/binary_meanpass*_cpu_solver.cc
binary_mean_pass_solver_h = ../CPP/binary_meanpass*_cpu_solver.h
binary_aug_lag_solver = ../CPP/binary_auglag*_cpu_solver.cc
binary_aug_lag_solver_h = ../CPP/binary_auglag*_cpu_solver.h

potts_mean_pass_solver = ../CPP/potts_meanpass*_cpu_solver.cc
potts_mean_pass_solver_h = ../CPP/potts_meanpass*_cpu_solver.h
potts_aug_lag_solver = ../CPP/potts_auglag*_cpu_solver.cc
potts_aug_lag_solver_h = ../CPP/potts_auglag*_cpu_solver.h

hmf_mean_pass_solver = ../CPP/hmf_meanpass*_cpu_solver.cc
hmf_mean_pass_solver_h = ../CPP/hmf_meanpass*_cpu_solver.h
hmf_aug_lag_solver = ../CPP/hmf_auglag*_cpu_solver.cc
hmf_aug_lag_solver_h = ../CPP/hmf_auglag*_cpu_solver.h



ifneq (,$(CUDAFLAG))
binary_mean_pass_solver += ../CPP/binary_meanpass*_gpu_solver.cc
binary_mean_pass_solver_h += ../CPP/binary_meanpass*_gpu_solver.h
binary_aug_lag_solver += ../CPP/binary_auglag*_gpu_solver.cc
binary_aug_lag_solver_h += ../CPP/binary_auglag*_gpu_solver.h
potts_mean_pass_solver += ../CPP/potts_meanpass*_gpu_solver.cc
potts_mean_pass_solver_h += ../CPP/potts_meanpass*_gpu_solver.h
potts_aug_lag_solver += ../CPP/potts_auglag*_gpu_solver.cc
potts_aug_lag_solver_h += ../CPP/potts_auglag*_gpu_solver.h
hmf_mean_pass_solver += ../CPP/hmf_meanpass*_gpu_solver.cc
hmf_mean_pass_solver_h += ../CPP/hmf_meanpass*_gpu_solver.h
hmf_aug_lag_solver += ../CPP/hmf_auglag*_gpu_solver.cc
hmf_aug_lag_solver_h += ../CPP/hmf_auglag*_gpu_solver.h
endif

ifneq (,$(CUDAFLAG))
gpu_kernels.o : ../CPP/gpu_kernels.cu ../CPP/gpu_kernels.h
	$(NVCC) $(CUFLAGS) -o gpu_kernels.o ../CPP/gpu_kernels.cu 
endif

cpu_kernels.o : ../CPP/cpu_kernels.cc ../CPP/cpu_kernels.h
	$(CC) $(CIFLAGS) ../CPP/cpu_kernels.cc

hmf_trees.o : ../CPP/hmf_trees.cc ../CPP/hmf_trees.h
	$(CC) $(CIFLAGS) ../CPP/hmf_trees.cc

binary_solver.o : $(binary_aug_lag_solver) $(binary_aug_lag_solver_h) $(binary_mean_pass_solver) $(binary_mean_pass_solver_h) cpu_kernels.o
	cat $(binary_aug_lag_solver) $(binary_mean_pass_solver) > binary_solver.cpp
	$(CC) $(CIFLAGS) binary_solver.cpp

potts_solver.o : $(potts_aug_lag_solver) $(potts_aug_lag_solver_h) $(potts_mean_pass_solver) $(potts_mean_pass_solver_h) cpu_kernels.o
	cat $(potts_aug_lag_solver) $(potts_mean_pass_solver) > potts_solver.cpp
	$(CC) $(CIFLAGS) potts_solver.cpp

hmf_solver.o : $(hmf_aug_lag_solver) $(hmf_aug_lag_solver_h) $(hmf_mean_pass_solver) $(hmf_mean_pass_solver_h) cpu_kernels.o hmf_trees.o
	cat $(hmf_aug_lag_solver) $(hmf_mean_pass_solver) > hmf_solver.cpp
	$(CC) $(CIFLAGS) hmf_solver.cpp

kernels_list =cpu_kernels.o
ifneq (,$(CUDAFLAG))
	kernels_list += gpu_kernels.o
endif

testing_potts.o : testing_potts.cpp
	$(CC) $(CIFLAGS) testing_potts.cpp
testing_binary.o : testing_binary.cpp
	$(CC) $(CIFLAGS) testing_binary.cpp

testing_potts : testing_potts.o potts_solver.o  $(kernels_list)
	$(CC) testing_potts.o potts_solver.o  $(kernels_list) $(LFLAGS) -o testing_potts

testing_binary : testing_binary.o binary_solver.o $(kernels_list)
	$(CC) testing_binary.o binary_solver.o $(kernels_list) $(LFLAGS) -o testing_binary
	

all : testing_potts.out testing_binary.out

clean :
	rm -rf *.o *.out
