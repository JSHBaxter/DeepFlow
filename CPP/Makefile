#makefile

MAKEFFL:=$(shell echo "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))" > flags_l.tmp)
MAKEFFC:=$(shell echo "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))" > flags_c.tmp)
TF_LFLAGS:=$(shell python3 flags_l.tmp)
TF_CFLAGS:=$(shell python3 flags_c.tmp) -I/usr/local/cuda/include

CUDAFLAG = -DGOOGLE_CUDA=1
CUDA_I = /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/lib64/libcublas.so

kernels.o :
	nvcc -std=c++11 -c -o kernels.o gpu_kernels.cu.cc $(TF_CFLAGS) $(CUDAFLAG) --expt-relaxed-constexpr -x cu -Xcompiler -fPIC
potts_mean_pass = potts_meanpass3d.cc
hmf_mean_pass = hmf_meanpass3d.cc hmf_trees.cc
helpers = cpu_kernels.cc
flow.so : kernels.o
	g++ -std=c++11 -shared -o flow.so $(potts_mean_pass) $(hmf_mean_pass) $(helpers) kernels.o $(CUDA_I) $(TF_CFLAGS) $(CUDAFLAG) -fPIC $(TF_LFLAGS)
	rm flags_l.tmp flags_c.tmp

all : flow.so

clean :
	rm kernels.o flow.so flags_l.tmp flags_c.tmp