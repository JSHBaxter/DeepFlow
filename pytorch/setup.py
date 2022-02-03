import setuptools
from setuptools import setup, Extension
import torch
from torch.utils import cpp_extension
import itertools
from itertools import product

basic_files = ["deepflow_bindings.cpp","../CPP/hmf_trees.cc","../CPP/cpu_kernels.cc"]
include_dirs = []
define_macros = []
if torch.has_cuda:
    basic_files += ["../CPP/gpu_kernels.cu"]
    include_dirs += ["/usr/local/cuda/include"]
    define_macros += [("DEEPFLOWUSECUDA","TRUE")]
t_list = ["binary","potts","hmf"]
dev_list = ["cpu","gpu"]

d_list = ["1d","2d","3d"]

module_files = [b for b in basic_files]
module_files += [t+"_"+dev+".cpp" for t,dev in product(t_list,dev_list)]
module_files += ["../CPP/"+t+"_auglag_"+dev+"_solver.cc" for t,dev in product(t_list,dev_list)]
module_files += ["../CPP/"+t+"_meanpass_"+dev+"_solver.cc" for t,dev in product(t_list,dev_list)]
module_files += ["../CPP/"+t+"_auglag"+d+"_"+dev+"_solver.cc" for t,d,dev in product(t_list,d_list,dev_list)]
module_files += ["../CPP/"+t+"_meanpass"+d+"_"+dev+"_solver.cc" for t,d,dev in product(t_list,d_list,dev_list)]
                 
setup(name="deepflow", ext_modules=[cpp_extension.CppExtension("deepflow", module_files, include_dirs=include_dirs, define_macros=define_macros)],
                       cmdclass={'build_ext': cpp_extension.BuildExtension})
