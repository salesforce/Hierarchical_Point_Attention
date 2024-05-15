// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <torch/extension.h>

at::Tensor box_query(at::Tensor xyz, at::Tensor boxes, const float radius,
                      const int nsample);
