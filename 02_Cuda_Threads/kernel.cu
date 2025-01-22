
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*device code(kernel)*/
__global__ void print_threadIdx()
{
	printf("thread and block coordinations\nthreadIdx.x :%d, threadIdx.y :%d, threadIdx.z :%d, blockIdx.x :%d, blockIdx.y :%d, blockIdx.z :%d\n\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

__global__ void print_gridDetails()
{
	printf("block and grid dimensions\nblockDim.x :%d, blockDim.y :%d, blockDim.z :%d, gridDim.x :%d, gridDim.y :%d, gridDim.z :%d\n\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}
/*host code*/
int main()
{
	int nx, ny, nz;
	nx = 4;
	ny = 4;
	nz = 4;
	/*kernel launch parameters*/
	dim3 block(2, 2, 2);
	dim3 grid(nx / block.x, ny / block.y, nz / block.z);
	print_threadIdx << <grid, block >> > ();
	/*wait untill previous kernel is completed*/
	cudaDeviceSynchronize();

	print_gridDetails << <grid, block >> > ();
	cudaDeviceSynchronize();
	
	cudaDeviceReset();
	return 0;
}