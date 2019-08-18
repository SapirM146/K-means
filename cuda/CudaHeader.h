#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MainHeader.h"

// cuda
cudaError_t changePos(Point* arrPoint, const int arr_size, const double dt, const int cuda_blocks, const int cuda_threads_in_block);
cudaError_t copyToSource(Point* arrPoint, const int arr_size);

