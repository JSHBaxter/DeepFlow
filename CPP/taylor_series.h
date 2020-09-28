// taylor_series.h
#ifndef TAYLOR_SERIES_H_
#define TAYLOR_SERIES_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

template <typename Device>
struct TaylorSeriesFunctor {
  void operator()(
      const Device& d,
	  bool channels_first,
      int* size_data,
      int* size_coeffs,
	  int dimension,
      const float* input,
	  const float* coeffs,
      float* out);
};

template <typename Device>
struct TaylorSeriesGradFunctor {
  void operator()(
      const Device& d,
	  bool channels_first,
      int* size_data,
      int* size_coeffs,
	  int dimension,
      const float* input,
	  const float* coeffs,
      const float* g,
      float* g_input,
      float* g_coeffs);
};

#endif // TAYLOR_SERIES_H_
