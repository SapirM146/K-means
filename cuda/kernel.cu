
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CudaHeader.h"
#include "MainHeader.h"

#define CUDA_BLOCKS_FOR_REMAINING 1
#define START_INDEX 0
#define NUM_OF_WORKS_REMAINS 1

//change Points positions in cuda threads
__global__ void changePosKernel(Point* arrPoint, const double dt, const int numOfWorks, const int cuda_threads_in_block, const int startIndex)
{
	int j, i = (blockIdx.x * cuda_threads_in_block * numOfWorks)+ (threadIdx.x* numOfWorks) + startIndex;
	for (j = 0; j < numOfWorks; j++) {
		arrPoint[i + j].x += dt * arrPoint[i + j].vx; // xi(t) = xi + dt*vxi
		arrPoint[i + j].y += dt * arrPoint[i + j].vy; // yi(t) = yi + dt*vyi
	}
}

Point *dev_arr = NULL;

// initial cuda and run kernel for histogram in cuda
cudaError_t changePos(Point* arrPoint, const int arr_size, const double dt, const int cuda_blocks, const int cuda_threads_in_block)
{
	int error_flag = 0;
	const int numOfWorks_Thousands = (arr_size)/ (cuda_blocks * cuda_threads_in_block);
	const int theRemainsPoints = arr_size - (numOfWorks_Thousands*(cuda_blocks * cuda_threads_in_block));
	const int cuda_threads_in_block_remains = theRemainsPoints;
	const int remainsIndex = (numOfWorks_Thousands*(cuda_blocks * cuda_threads_in_block));

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		error_flag = 1;
        goto Error;
    }

    // Allocate GPU buffers for one arrays 
	cudaStatus = cudaMalloc((void**)&dev_arr, arr_size * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		error_flag = 1;
		goto Error;
	}

    // Copy input array from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_arr, arrPoint, arr_size * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		error_flag = 1;
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	changePosKernel <<<cuda_blocks, cuda_threads_in_block >>>(dev_arr, dt, numOfWorks_Thousands, cuda_threads_in_block, START_INDEX);
	if(theRemainsPoints > 0)
		changePosKernel <<<CUDA_BLOCKS_FOR_REMAINING, cuda_threads_in_block_remains >>>(dev_arr, dt, NUM_OF_WORKS_REMAINS, cuda_threads_in_block_remains, remainsIndex); // remainsIndex = (the remains points start index)

Error:
	if(error_flag == 1)
		cudaFree(dev_arr);

    return cudaStatus;
}

cudaError_t copyToSource(Point* arrPoint, const int arr_size)
{
	cudaError_t cudaStatus;

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching changePosKernel!\n", cudaStatus);
		goto Error;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "changePosKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Copy output points array from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(arrPoint, dev_arr, arr_size * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_arr);

	return cudaStatus;
}
