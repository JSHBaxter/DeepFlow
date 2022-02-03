#include <pybind11/pybind11.h>

namespace py = pybind11;

void binary_cpu_bindings(py::module &);
void potts_cpu_bindings(py::module &);
void hmf_cpu_bindings(py::module &);

#ifdef DEEPFLOWUSECUDA
    void binary_gpu_bindings(py::module &);
    void potts_gpu_bindings(py::module &);
    void hmf_gpu_bindings(py::module &);
#endif


PYBIND11_MODULE(deepflow, m) {
    binary_cpu_bindings(m);
    potts_cpu_bindings(m);
    hmf_cpu_bindings(m);
    #ifdef DEEPFLOWUSECUDA
        binary_gpu_bindings(m);
        potts_gpu_bindings(m);
        hmf_gpu_bindings(m);
    #endif
}