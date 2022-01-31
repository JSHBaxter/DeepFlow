#makefile

PYTHON=python3
CC=g++
C=gcc-7
NVCC=nvcc
CUDA_DIR=/usr/local/cuda

CUDAFLAG=-DGOOGLE_CUDA
CUDA_I=-I${CUDA_DIR}/include
CUDA_L=${CUDA_DIR}/lib64/libcudart.so ${CUDA_DIR}/lib64/libcublas.so

TF_CFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; [print(i) for i in tf.sysconfig.get_compile_flags()]') 
TF_LFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; [print(i) for i in tf.sysconfig.get_link_flags()]')

CFLAGS := -std=c++11 $(CUDA_I) $(TF_CFLAGS) $(CUDAFLAG) -I../CPP -fPIC -shared 
CIFLAGS := -std=c++11 $(CUDA_I) $(TF_CFLAGS) $(CUDAFLAG) -I../CPP -fPIC -c
CUFLAGS = -std=c++11 -c $(TF_CFLAGS) $(CUDAFLAG) --expt-relaxed-constexpr -x cu -Xcompiler -fPIC -ccbin $(C) -DNDEBUG
LFLAGS := $(CUDA_L) $(TF_LFLAGS)


interfacing = regularNd.cc hmfNd.cc tf_memory_utils.cc
interfacing_h = regularNd.h hmfNd.h tf_memory_utils.h

binary_mean_pass = binary_meanpass*d.cc
binary_mean_pass_h = binary_meanpass*d.h
binary_aug_lag = binary_auglag*d.cc
binary_aug_lag_h = binary_auglag*d.h
binary_mean_pass_solver = ../CPP/binary_meanpass*_cpu_solver.cc
binary_mean_pass_solver_h = ../CPP/binary_meanpass*_cpu_solver.h
binary_aug_lag_solver = ../CPP/binary_auglag*_cpu_solver.cc
binary_aug_lag_solver_h = ../CPP/binary_auglag*_cpu_solver.h

potts_mean_pass = potts_meanpass*d.cc
potts_mean_pass_h = potts_meanpass*d.h
potts_aug_lag = potts_auglag*d.cc
potts_aug_lag_h = potts_auglag*d.h
potts_mean_pass_solver = ../CPP/potts_meanpass*_cpu_solver.cc
potts_mean_pass_solver_h = ../CPP/potts_meanpass*_cpu_solver.h
potts_aug_lag_solver = ../CPP/potts_auglag*_cpu_solver.cc
potts_aug_lag_solver_h = ../CPP/potts_auglag*_cpu_solver.h

hmf_mean_pass = hmf_meanpass*d.cc
hmf_mean_pass_h = hmf_meanpass*d.h
hmf_aug_lag = hmf_auglag*d.cc
hmf_aug_lag_h = hmf_auglag*d.h
hmf_mean_pass_solver = ../CPP/hmf_meanpass*_cpu_solver.cc
hmf_mean_pass_solver_h = ../CPP/hmf_meanpass*_cpu_solver.h
hmf_aug_lag_solver = ../CPP/hmf_auglag*_cpu_solver.cc
hmf_aug_lag_solver_h = ../CPP/hmf_auglag*_cpu_solver.h

taylor_series = taylor_series.cpp taylor_series_cpu.cc
taylor_series_h = taylor_series.h

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
taylor_series += taylor_series_gpu.cc
endif

ifneq (,$(CUDAFLAG))
gpu_kernels.o : ../CPP/gpu_kernels.cu ../CPP/gpu_kernels.h
	$(NVCC) $(CUFLAGS) -o gpu_kernels.o ../CPP/gpu_kernels.cu 
endif

cpu_kernels.o : ../CPP/cpu_kernels.cc ../CPP/cpu_kernels.h
	$(CC) $(CIFLAGS) ../CPP/cpu_kernels.cc

hmf_trees.o : ../CPP/hmf_trees.cc ../CPP/hmf_trees.h
	$(CC) $(CIFLAGS) ../CPP/hmf_trees.cc

interfacing.o : $(interfacing) $(interfacing_h)
	cat $(interfacing) > interfacing.cpp
	$(CC) $(CIFLAGS) interfacing.cpp
	rm interfacing.cpp

binary_solver.o : $(binary_aug_lag_solver) $(binary_aug_lag_solver_h) $(binary_mean_pass_solver) $(binary_mean_pass_solver_h) cpu_kernels.o
	cat $(binary_aug_lag_solver) $(binary_mean_pass_solver) > binary_solver.cpp
	$(CC) $(CIFLAGS) binary_solver.cpp
	rm binary_solver.cpp

binary.o : $(binary_mean_pass) $(binary_aug_lag) $(binary_mean_pass_h) $(binary_aug_lag_h) interfacing.o binary_solver.o
	cat $(binary_mean_pass) $(binary_aug_lag) > binary.cpp
	$(CC) $(CIFLAGS) binary.cpp
	rm binary.cpp

potts_solver.o : $(potts_aug_lag_solver) $(potts_aug_lag_solver_h) $(potts_mean_pass_solver) $(potts_mean_pass_solver_h) cpu_kernels.o
	cat $(potts_aug_lag_solver) $(potts_mean_pass_solver) > potts_solver.cpp
	$(CC) $(CIFLAGS) potts_solver.cpp
	rm potts_solver.cpp

potts.o : $(potts_mean_pass) $(potts_aug_lag) $(potts_mean_pass_h) $(potts_aug_lag_h) interfacing.o potts_solver.o
	cat $(potts_mean_pass) $(potts_aug_lag) > potts.cpp
	$(CC) $(CIFLAGS) potts.cpp
	rm potts.cpp

hmf_solver.o : $(hmf_aug_lag_solver) $(hmf_aug_lag_solver_h) $(hmf_mean_pass_solver) $(hmf_mean_pass_solver_h) cpu_kernels.o hmf_trees.o
	cat $(hmf_aug_lag_solver) $(hmf_mean_pass_solver) > hmf_solver.cpp
	$(CC) $(CIFLAGS) hmf_solver.cpp
	rm hmf_solver.cpp

hmf.o : $(hmf_mean_pass) $(hmf_aug_lag) $(hmf_mean_pass_h) $(hmf_aug_lag_h) interfacing.o hmf_solver.o
	cat $(hmf_mean_pass) $(hmf_aug_lag) > hmf.cpp
	$(CC) $(CIFLAGS) hmf.cpp
	rm hmf.cpp

taylor_series.o : $(taylor_series) $(taylor_series_h)
	$(CC) $(CIFLAGS) -o taylor_series.o taylor_series.cpp

object_list = interfacing.o binary.o potts.o hmf.o cpu_kernels.o hmf_trees.o taylor_series.o
ifneq (,$(CUDAFLAG))
	object_list += gpu_kernels.o
endif

flow.so : $(object_list)
	$(CC) $(CFLAGS) -o flow.so $(object_list) $(LFLAGS)

all : flow.so

clean :
	rm -rf *.o *.so
