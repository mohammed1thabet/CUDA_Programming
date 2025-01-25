
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

/*device code(kernel)*/
__global__ void mem_transfer_test(int* inputArr)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("tid : %d, gid: %d, data : %d\n", threadIdx.x, gid, inputArr[gid]);
}

/*host code*/
int main()
{	
	int arrSize = 128;
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

	dim3 block(64);
	dim3 grid(2);
	mem_transfer_test<< <grid, block>> > (d_input);
	cudaDeviceSynchronize();

	cudaFree(d_input);
	free(h_input);


	cudaDeviceReset();
	return 0;
}