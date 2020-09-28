#makefile

PYTHON=python3
CC=g++
C=gcc
NVCC=nvcc

CUDAFLAG = -DGOOGLE_CUDA
CUDA_I = -I/usr/local/cuda/include
CUDA_L = /usr/local/cuda/lib64/libcudart.so /usr/local/cuda-10.2/lib64/libcublas.so

TF_CFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; [print(i) for i in tf.sysconfig.get_compile_flags()]') 
TF_LFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; [print(i) for i in tf.sysconfig.get_link_flags()]')

CFLAGS := -std=c++11 $(CUDA_I) $(TF_CFLAGS) $(CUDAFLAG) -fPIC -shared 
CIFLAGS := -std=c++11 $(CUDA_I) $(TF_CFLAGS) $(CUDAFLAG) -fPIC -c
CUFLAGS = -std=c++11 -c $(TF_CFLAGS) $(CUDAFLAG) --expt-relaxed-constexpr -x cu -Xcompiler -fPIC -ccbin $(C) -DNDEBUG
LFLAGS := $(CUDA_L) $(TF_LFLAGS)

interfacing = regularNd.cc hmfNd.cc tf_memory_utils.cc
interfacing_h = regularNd.h hmfNd.h tf_memory_utils.h
binary_mean_pass = binary_meanpass3d.cc binary_meanpass2d.cc binary_meanpass1d.cc binary_meanpass_cpu_solver.cc
binary_mean_pass_h = binary_meanpass3d.h binary_meanpass2d.h binary_meanpass1d.h binary_meanpass_cpu_solver.h
binary_aug_lag = binary_auglag3d.cc binary_auglag2d.cc binary_auglag1d.cc binary_auglag_cpu_solver.cc
binary_aug_lag_h = binary_auglag3d.h binary_auglag2d.h binary_auglag1d.h binary_auglag_cpu_solver.h
potts_mean_pass = potts_meanpass3d.cc potts_meanpass2d.cc potts_meanpass1d.cc potts_meanpass_cpu_solver.cc
potts_mean_pass_h = potts_meanpass3d.h potts_meanpass2d.h potts_meanpass1d.h potts_meanpass_cpu_solver.h
potts_aug_lag = potts_auglag3d.cc potts_auglag2d.cc potts_auglag1d.cc potts_auglag_cpu_solver.cc
potts_aug_lag_h = potts_auglag3d.h potts_auglag2d.h potts_auglag1d.h potts_auglag_cpu_solver.h
hmf_mean_pass = hmf_meanpass3d.cc  hmf_meanpass2d.cc  hmf_meanpass1d.cc hmf_meanpass_cpu_solver.cc
hmf_mean_pass_h = hmf_meanpass3d.cc  hmf_meanpass2d.cc  hmf_meanpass1d.cc hmf_meanpass_cpu_solver.cc
hmf_aug_lag = hmf_auglag1d.cc hmf_auglag2d.cc hmf_auglag3d.cc hmf_auglag_cpu_solver.cc
hmf_aug_lag_h = hmf_auglag1d.h hmf_auglag2d.h hmf_auglag3d.h hmf_auglag_cpu_solver.h
taylor_series = taylor_series.cpp taylor_series_cpu.cc
taylor_series_h = taylor_series.h
cpu_kernels = cpu_kernels.cc hmf_trees.cc hmf_trees.h

ifneq (,$(CUDAFLAG))
	binary_mean_pass += binary_meanpass_gpu_solver.cc
	binary_mean_pass_h += binary_meanpass_gpu_solver.h
	binary_aug_lag += binary_auglag_gpu_solver.cc
	binary_aug_lag_h += binary_auglag_gpu_solver.h
	potts_mean_pass += potts_meanpass_gpu_solver.cc
	potts_mean_pass_h += potts_meanpass_gpu_solver.h
	potts_aug_lag += potts_auglag_gpu_solver.cc
	potts_aug_lag_h += potts_auglag_gpu_solver.h
	hmf_mean_pass += hmf_meanpass_gpu_solver.cc
	hmf_mean_pass_h += hmf_meanpass_gpu_solver.h
	hmf_aug_lag += hmf_auglag_gpu_solver.cc
	hmf_aug_lag_h += hmf_auglag_gpu_solver.h
	taylor_series += taylor_series_gpu.cc
endif

ifneq (,$(CUDAFLAG))
gpu_kernels.o : gpu_kernels.cu.cc
	$(NVCC) $(CUFLAGS) -o gpu_kernels.o gpu_kernels.cu.cc 
endif

cpu_kernels.o : cpu_kernels.cc cpu_kernels.h
	$(CC) $(CIFLAGS) cpu_kernels.cc
	
hmf_trees.o : hmf_trees.cc hmf_trees.h
	$(CC) $(CIFLAGS) hmf_trees.cc

interfacing.o : $(interfacing) $(interfacing_h)
	cat $(interfacing) > interfacing.cpp
	$(CC) $(CIFLAGS) interfacing.cpp
	rm interfacing.cpp
	
binary.o : $(binary_mean_pass) $(binary_aug_lag) $(binary_mean_pass_h) $(binary_aug_lag_h) interfacing.o cpu_kernels.o
	cat $(binary_mean_pass) $(binary_aug_lag) > binary.cpp
	$(CC) $(CIFLAGS) binary.cpp
	rm binary.cpp
	
potts.o : $(potts_mean_pass) $(potts_aug_lag) $(potts_mean_pass_h) $(potts_aug_lag_h) interfacing.o cpu_kernels.o
	cat $(potts_mean_pass) $(potts_aug_lag) > potts.cpp
	$(CC) $(CIFLAGS) potts.cpp
	rm potts.cpp
	
hmf.o : $(hmf_mean_pass) $(hmf_aug_lag) $(hmf_mean_pass_h) $(hmf_aug_lag_h) interfacing.o cpu_kernels.o hmf_trees.o
	cat $(hmf_mean_pass) $(hmf_aug_lag) > hmf.cpp
	$(CC) $(CIFLAGS) hmf.cpp
	rm hmf.cpp

taylor_series.o : $(taylor_series) $(taylor_series_h)
	$(CC) $(CIFLAGS) -o taylor_series.o taylor_series.cpp

object_list = interfacing.o potts.o binary.o hmf.o cpu_kernels.o hmf_trees.o taylor_series.o
ifneq (,$(CUDAFLAG))
	object_list += gpu_kernels.o
endif

flow.so : $(object_list)
	$(CC) $(CFLAGS) -o flow.so $(object_list) $(LFLAGS)

all : flow.so

clean :
	rm -rf *.o *.so
