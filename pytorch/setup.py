import setuptools
from setuptools import setup, Extension
import torch
from torch.utils import cpp_extension
import itertools
from itertools import product

basic_files = ["deepflow_bindings.cpp","../CPP/hmf_trees.cc","../CPP/common.cc","../CPP/cpu_kernels.cc","../CPP/algorithm.cc"]
include_dirs = []
define_macros = [("DEBUG_PRINT","false"),("DEBUG_ITER","true"),("DEBUG_ITER_EXTREME","false")]
if torch.has_cuda:
    basic_files += ["../CPP/gpu_kernels.cu"]
    include_dirs += ["/usr/local/cuda/include"]
    define_macros += [("USE_CUDA","TRUE")]
t_list = ["binary","potts","hmf","dagmf"]
dev_list = ["cpu"]
if torch.has_cuda:
    dev_list.append("gpu");
alg_list = ["auglag","meanpass"]


module_files = [b for b in basic_files]
module_files += [t+"_"+dev+".cpp" for t,dev in product(t_list,dev_list)]
module_files += ["../CPP/spatial_"+a+".cc" for a in alg_list]
module_files += ["../CPP/"+t+"_"+a+"_solver.cc" for t,a in product(t_list,alg_list)]
     
print(module_files)
            
setup(name="deepflow", ext_modules=[cpp_extension.CppExtension("deepflow", module_files, include_dirs=include_dirs, define_macros=define_macros)],
                       cmdclass={'build_ext': cpp_extension.BuildExtension})
