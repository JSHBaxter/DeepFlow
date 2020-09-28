#ifndef TF_MEMORY_UTILS_H
#define TF_MEMORY_UTILS_H

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

void get_temporary_buffers(OpKernelContext* context, float**& buffers_full, const int size, const int number, const Tensor* d);
void get_temporary_buffers(OpKernelContext* context, float**& buffers1, const int size1, const int number1, float**& buffers2, const int size2, const int number2, const Tensor* d);
void clear_temporary_buffers(OpKernelContext* context, float** buffers, const int size, const int number);


#endif //TF_MEMORY_UTILS_H