
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*device code(kernel), executed on GPU*/
__global__ void HelloCuda()
{
	printf("Hello CUDA, blockIdx : %d, threadIdx : %2d\n", blockIdx.x, threadIdx.x);
}

/*host code, executed on CPU*/
int main()
{

	/*kernel launch parameters*/
	dim3 block(32);//block contains 32 threads
	dim3 grid(2);//grid contains 2 thread blocks

	/*launch kernel to print "Hello CUDA" on 32 threads * 2 blocks in parallel*/
	HelloCuda << <grid, block >> > ();

	/*wait untill previous kernel execution is completed*/
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}