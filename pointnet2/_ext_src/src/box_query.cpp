

#include "box_query.h"
#include "utils.h"
#include <stdio.h>

void query_box_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *xyz, const float *boxes, int *idx);

at::Tensor box_query(at::Tensor xyz, at::Tensor boxes, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(boxes);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(boxes);

  if (boxes.is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::ones({boxes.size(0), boxes.size(1), nsample},
                   at::device(boxes.device()).dtype(at::ScalarType::Int)) * -1;
  
  if (boxes.is_cuda()) {
    query_box_point_kernel_wrapper(xyz.size(0), xyz.size(1), boxes.size(1),
                                    radius, nsample, xyz.data<float>(), boxes.data<float>(), idx.data<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return idx;
}
