import setuptools
from setuptools import setup, Extension
import torch
from torch.utils import cpp_extension
import itertools
from itertools import product

cpu_basic_files = ["../CPP/cpu_kernels.cc"]
gpu_basic_files = ["../CPP/gpu_kernels.cu"]
include_dirs = ["/usr/local/cuda/include"]

t_list = ["binary","potts","hmf"]
dev_list = ["cpu","gpu"]

d_list = ["1d","2d","3d"]

module_name = [  t+"_"+dev for (t,dev) in product(t_list,dev_list)]
module_files = [[t+"_"+dev+".cpp"] +
                        (cpu_basic_files if dev == "cpu" else gpu_basic_files) + 
                        (["../CPP/hmf_trees.cc",] if t == "hmf" else []) +
                        (["../CPP/"+t+"_auglag"+d+"_"+dev+"_solver.cc" for d in d_list]) +
                        (["../CPP/"+t+"_meanpass"+d+"_"+dev+"_solver.cc" for d in d_list]) +
                        (["../CPP/"+t+"_auglag_"+dev+"_solver.cc","../CPP/"+t+"_meanpass_"+dev+"_solver.cc"])
                                                              for (t,dev) in product(t_list,dev_list)]

for n,m in zip(module_name,module_files):
	print(n,m)

for m,f in zip(module_name,module_files):
    setup(name=m, ext_modules=[cpp_extension.CppExtension(m, f, include_dirs=include_dirs)],
                            cmdclass={'build_ext': cpp_extension.BuildExtension})
