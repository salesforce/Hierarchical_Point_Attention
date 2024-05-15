

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda_utils.h"

#include <fstream>

// input: new_xyz(b, m, 3) xyz(b, n, 3) boxes(b, m, 6)
// output: idx(b, m, nsample)
__global__ void query_box_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ xyz,
					                              const float *__restrict__ boxes,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  boxes += batch_index * m * 6;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;

  for (int j = index; j < m; j += stride) {
    float box_x = boxes[j * 6 + 0];
    float box_y = boxes[j * 6 + 1];
    float box_z = boxes[j * 6 + 2];
    float box_w = 0.5 * boxes[j * 6 + 3];
    float box_l = 0.5 * boxes[j * 6 + 4];
    float box_h = 0.5 * boxes[j * 6 + 5];

    int cnt = 0;
    for (int k = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
    
      if (fabs(x - box_x) <= box_w && fabs(y - box_y) <= box_l && fabs(z - box_z) <= box_h) {
	      idx[j * nsample + cnt] = k;
        ++cnt;
      }  
    }
    // TODO: random shuffle samples idx and take the first nsample idx

    if (cnt == 0) {
      // exactly follow the ball_query implementation 
      // first init all idx to 0
      for (int l = 0; l < nsample; ++l) {
	            idx[j * nsample + l] = 0;
	    }
      for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
      
        float d2 = (box_x - x) * (box_x - x) + (box_y - y) * (box_y - y) +
                 (box_z - z) * (box_z - z);

        if (d2 < radius2) {
          if (cnt == 0) {
	          for (int l = 0; l < nsample; ++l) {
	            idx[j * nsample + l] = k;
	          }
	        }
	        idx[j * nsample + cnt] = k;
          ++cnt;
        }  
      }
    }
  }
}

void query_box_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *xyz, const float *boxes, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_box_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, xyz, boxes, idx);

  CUDA_CHECK_ERRORS();
}
