from itertools import product

import torch
from setuptools import setup

#note that this needs to run without build isolation to get this extension type to handle CUDA, unfortunately.
#looking for fix now so build isolation can be used
from torch.utils import cpp_extension

#configuration 
include_dirs = []
define_macros = [("DEBUG_PRINT","false"),("DEBUG_ITER","false"),("DEBUG_ITER_EXTREME","false")]
has_cuda = False #torch.backends.cuda.is_built()

t_list = ["binary","potts","hmf","dagmf"]
alg_list = ["auglag","meanpass"]
dev_list = ["cpu"]
if has_cuda:
    dev_list.append("gpu");

#get all the "basic" files, i.e. ones that are common across multiple model types
basic_files = ["deepflow_bindings.cpp","../CPP/hmf_trees.cc","../CPP/common.cc","../CPP/algorithm.cc"]
basic_files += ["../CPP/cpu_kernels.cc"] +["../CPP/cpu_kernels_"+a+".cc" for a in alg_list]
if has_cuda:
    basic_files += ["../CPP/gpu_kernels.cu"] +["../CPP/gpu_kernels_"+a+".cu" for a in alg_list]
    include_dirs += ["/usr/local/cuda/include"]
    define_macros += [("USE_CUDA","TRUE")]
basic_files += ["../CPP/spatial_"+a+".cc" for a in alg_list]
basic_files += ["../CPP/spatial_star_auglag.cc"]

#get all the model-type specific files
module_files = [b for b in basic_files]
module_files += [t+"_"+dev+".cpp" for t,dev in product(t_list,dev_list)]
module_files += ["../CPP/"+t+"_"+a+"_solver.cc" for t,a in product(t_list,alg_list)]

#and make the project
setup(name="deepflow",
      ext_modules=[cpp_extension.CppExtension("deepflow", module_files, include_dirs=include_dirs, define_macros=define_macros)],
      install_requires=['torch','torchvision',],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
