#makefile

PYTHON=python3
CC=g++
C=gcc
NVCC=nvcc

CUDAFLAG = -DGOOGLE_CUDA
CUDA_I = -I/usr/local/cuda/include
CUDA_L = /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/lib64/libcublas.so

TF_CFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; [print(i) for i in tf.sysconfig.get_compile_flags()]') 
TF_LFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; [print(i) for i in tf.sysconfig.get_link_flags()]')

CFLAGS := -std=c++11 $(CUDA_I) $(TF_CFLAGS) $(CUDAFLAG) -fPIC -shared 
CUFLAGS = -std=c++11 -c $(TF_CFLAGS) $(CUDAFLAG) --expt-relaxed-constexpr -x cu -Xcompiler -fPIC -ccbin $(C) -DNDEBUG
LFLAGS := $(CUDA_L) $(TF_LFLAGS)

binary_mean_pass = binary_meanpass3d.cc binary_meanpass2d.cc binary_meanpass1d.cc binary_meanpass_cpu_solver.cc
binary_aug_lag = binary_auglag3d.cc binary_auglag2d.cc binary_auglag1d.cc binary_auglag_cpu_solver.cc
potts_mean_pass = potts_meanpassNd.cc potts_meanpass3d.cc potts_meanpass2d.cc potts_meanpass1d.cc potts_meanpass_cpu_solver.cc
potts_aug_lag = potts_auglag3d.cc potts_auglag2d.cc potts_auglag1d.cc potts_auglag_cpu_solver.cc
hmf_mean_pass = hmf_meanpass3d.cc  hmf_meanpass2d.cc  hmf_meanpass1d.cc hmf_meanpass_cpu_solver.cc
hmf_aug_lag = hmf_auglag1d.cc hmf_auglag2d.cc hmf_auglag3d.cc hmf_auglag_cpu_solver.cc

ifneq (,$(CUDAFLAG))
	helpers_binary = binary_meanpass_gpu_solver.cc binary_auglag_gpu_solver.cc
	helpers_potts = potts_auglag_gpu_solver.cc potts_meanpass_gpu_solver.cc 
	helpers_hmf = hmf_auglag_gpu_solver.cc hmf_meanpass_gpu_solver.cc
	helpers = tf_memory_utils.cc cpu_kernels.cc hmf_trees.cc kernels.o $(helpers_binary) $(helpers_potts) $(helpers_hmf)
else
	helpers = tf_memory_utils.cc cpu_kernels.cc hmf_trees.cc
endif

ifneq (,$(CUDAFLAG))

kernels.o : gpu_kernels.cu.cc
	$(NVCC) $(CUFLAGS) -o kernels.o gpu_kernels.cu.cc
    
endif

flow.so : $(binary_mean_pass) $(binary_aug_lag) $(potts_mean_pass) $(potts_aug_lag) $(hmf_mean_pass) $(hmf_aug_lag) $(helpers)
	$(CC) $(CFLAGS) -o flow.so $(binary_mean_pass) $(binary_aug_lag) $(potts_mean_pass) $(potts_aug_lag) $(hmf_mean_pass) $(hmf_aug_lag) $(helpers) $(LFLAGS)

all : flow.so

clean :
	rm -rf *.o *.so
