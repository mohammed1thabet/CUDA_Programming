
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

/*device code(kernel)*/
__global__ void print3dArray(int* inputArr)
{
	int x_offset, y_offset, z_offset;
	int blockSize = blockDim.x * blockDim.y * blockDim.z;
	z_offset = (blockIdx.z * (gridDim.x * gridDim.y * blockSize)) + (threadIdx.z * (blockDim.x * blockDim.y));
	y_offset = (blockIdx.y * gridDim.x * blockSize) + (threadIdx.y * blockDim.x);
	x_offset = (blockIdx.x * blockSize) + threadIdx.x;
	int gid = x_offset + y_offset + z_offset;
	printf("tid : %3d, gid: %3d, data : %3d, blocIdx.x:%3d, blocIdx.y:%3d, blocIdx.z:%3d\n", threadIdx.x, gid, inputArr[gid], blockIdx.x, blockIdx.y, blockIdx.z);
}

/*host code*/
int main()
{
	int arrSize = 64;
	int* h_input = (int*)malloc(arrSize * sizeof(int));

	time_t t;
	srand((unsigned)time(&t));

	/*intialize host buffer with random values*/
	for (int i = 0; i < arrSize; i++)
	{
		h_input[i] = (int)(rand() & 0xff);
	}

	int* d_input;
	cudaMalloc((void**)&d_input, arrSize * sizeof(int));

	cudaMemcpy(d_input, h_input, arrSize * sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(2,2,2);
	dim3 grid(2,2,2);
	print3dArray << <grid, block >> > (d_input);

	cudaFree(d_input);
	free(h_input);

	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}