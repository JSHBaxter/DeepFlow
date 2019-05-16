#makefile

CUDAFLAG =
ifneq (,$(CUDAFLAG))
	CUDA_I = /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/lib64/libcublas.so
else
	CUDA_I =
endif

MAKEFFL:=$(shell echo "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))" > flags_l.tmp)
MAKEFFC:=$(shell echo "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))" > flags_c.tmp)
TF_LFLAGS:=$(shell python3 flags_l.tmp)
TF_CFLAGS:=$(shell python3 flags_c.tmp)
ifneq (,$(CUDAFLAG))
	TF_CFLAGS:=$(TF_CFLAGS) -I/usr/local/cuda/include
endif

potts_mean_pass = potts_meanpass3d.cc potts_meanpass2d.cc potts_meanpass1d.cc
potts_aug_lag = potts_auglag3d.cc potts_auglag2d.cc potts_auglag1d.cc
hmf_mean_pass = hmf_meanpass3d.cc  hmf_meanpass2d.cc  hmf_meanpass1d.cc
hmf_aug_lag = hmf_auglag1d.cc
helpers = tf_memory_utils.cc cpu_kernels.cc hmf_trees.cc

ifneq (,$(CUDAFLAG))
kernels.o :
	nvcc -std=c++11 -c -o kernels.o gpu_kernels.cu.cc $(TF_CFLAGS) $(CUDAFLAG) --expt-relaxed-constexpr -x cu -Xcompiler -fPIC
helpers = $(helpers) kernels.o
endif


flow.so : $(helpers)
	g++ -std=c++11 -shared -o flow.so $(potts_mean_pass) $(potts_aug_lag) $(hmf_mean_pass) $(hmf_aug_lag) $(helpers) $(CUDA_I) $(TF_CFLAGS) $(CUDAFLAG) -fPIC $(TF_LFLAGS)
	rm flags_l.tmp flags_c.tmp

all : flow.so

clean :
	rm kernels.o flow.so flags_l.tmp flags_c.tmp