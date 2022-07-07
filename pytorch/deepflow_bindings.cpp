#include <pybind11/pybind11.h>

namespace py = pybind11;

void binary_cpu_bindings(py::module &);
void potts_cpu_bindings(py::module &);
void hmf_cpu_bindings(py::module &);
void dagmf_cpu_bindings(py::module &);

#ifdef USE_CUDA
    void binary_gpu_bindings(py::module &);
    void potts_gpu_bindings(py::module &);
    void hmf_gpu_bindings(py::module &);
    void dagmf_gpu_bindings(py::module &);
#endif


PYBIND11_MODULE(deepflow, m) {
    binary_cpu_bindings(m);
    potts_cpu_bindings(m);
    hmf_cpu_bindings(m);
    dagmf_cpu_bindings(m);
    #ifdef USE_CUDA
        binary_gpu_bindings(m);
        potts_gpu_bindings(m);
        hmf_gpu_bindings(m);
        dagmf_gpu_bindings(m);
    #endif
}