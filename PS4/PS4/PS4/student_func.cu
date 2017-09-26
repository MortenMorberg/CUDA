//Udacity HW 4
//Radix Sorting

//#include "reference_calc.cpp"
#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
===============

For this assignment we are implementing red eye removal.  This is
accomplished by first creating a score for every pixel that tells us how
likely it is to be a red eye pixel.  We have already done this for you - you
are receiving the scores and need to sort them in ascending order so that we
know which pixels to alter to remove the red eye.

Note: ascending order == smallest to largest

Each score is associated with a position, when you sort the scores, you must
also move the positions accordingly.

Implementing Parallel Radix Sort with CUDA
==========================================

The basic idea is to construct a histogram on each pass of how many of each
"digit" there are.   Then we scan this histogram so that we know where to put
the output of each digit.  For example, the first 1 must come after all the
0s so we have to know how many 0s there are to be able to start moving 1s
into the correct position.

1) Histogram of the number of occurrences of each digit
2) Exclusive Prefix Sum of Histogram
3) Determine relative offset of each digit
For example [0 0 1 1 0 0 1]
->  [0 1 0 1 2 3 2]
4) Combine the results of steps 2 & 3 to determine the final
output location for each element and move it there

LSB Radix sort is an out-of-place sort and you will need to ping-pong values
between the input and output buffers we have provided.  Make sure the final
sorted results end up in the output buffer!  Hint: You may need to do a copy
at the end.

*/
#define numBits 2
#define numBins (1<<numBits)


__global__ void kernel_histogram(
	const unsigned int* const d_input,
	unsigned int * const d_output,
	const int bitCount,
	const size_t numElems)
{
	// Loading whole block of data
	extern __shared__ unsigned int tempElems[];
	
	unsigned int tId = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x + tId;

	// Running through elements
	if (idx < numElems) {
		tempElems[tId] = d_input[idx];
		__syncthreads();
		unsigned int binCount = (tempElems[tId] >> bitCount)&(numBins - 1);
		atomicAdd(&d_output[binCount], 1);
	}
}

/*
__global__ void kernel_scan_hilisSteele(
	const unsigned int * const d_input, 
	unsigned int *d_blockLastElems,
	unsigned int * const d_output)
{
	extern __shared__ unsigned int temp[];

	const int tId = threadIdx.x;
	int p_out = 0, p_in = 1;

	// Exclusive scan
	temp[tId] = tId > 0 ? d_input[tId - 1] : 0;

	__syncthreads();

	for (unsigned int i = 1; i < blockDim.x; i <<= 1)
	{
		// Swaping double buffer indicies
		p_out = 1 - p_out;
		p_in = 1 - p_out;

		if (tId >= i)
			temp[p_out*blockDim.x + tId] = temp[p_in*blockDim.x + tId] + temp[p_in*blockDim.x + tId - i];
		else
			temp[p_out*blockDim.x + tId] = temp[p_in*blockDim.x + tId];
		__syncthreads();
	}

	d_output[tId] = temp[p_out*blockDim.x + tId];

	if (tId == (blockDim.x-1))
		d_blockLastElems[blockIdx.x] = temp[tId];
}*/

__global__
void scan(const unsigned int * const d_in, unsigned int *d_out, unsigned int *d_blockLastElems, const size_t numElems)
{
	//use shared memory to load the whole block data
	extern __shared__ unsigned int temp[];

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x*blockDim.x + tid;

	if (idx >= numElems)
		return;
	temp[tid] = d_in[idx];
	__syncthreads();

	for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
		unsigned int temp_val = temp[tid];
		__syncthreads();

		if (tid + stride < blockDim.x)
			temp[tid + stride] += temp_val;
		__syncthreads();
	}

	// exclusive scan  
	d_out[idx] = tid > 0 ? temp[tid - 1] : 0;

	if (tid == (blockDim.x - 1))
		d_blockLastElems[blockIdx.x] = temp[tid];

}

__global__ void kernel_add(
	const unsigned int * const d_input,
	unsigned int * const d_output,
	const size_t numElems)
{
	unsigned int tId = threadIdx.x;
	unsigned int bIdx = blockIdx.x;
	unsigned int idx = bIdx * blockDim.x + tId;

	if (idx < numElems) {
		d_output[idx] += d_input[bIdx];
	}

}

// https://github.com/raoqiyu/CS344-Problem-Sets/blob/master/Problem%20Set%204/student_func.cu
void prefix_sum(
	unsigned int *d_input,
	unsigned int *d_output,
	const size_t numElems)
{
	const dim3 blockSize(min(1024, (int)numElems));
	const dim3 gridSize(ceil((float)numElems / blockSize.x));

	unsigned int *d_blockLastElems;
	checkCudaErrors(cudaMalloc((void**)&d_blockLastElems, gridSize.x * sizeof(unsigned int)));

	// Scanning each block
	scan << <gridSize, blockSize, blockSize.x * sizeof(unsigned int) >> > (d_input, d_output, d_blockLastElems, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	if (gridSize.x > 1) {
		// Scanning all blocks last elements
		prefix_sum(d_blockLastElems, d_blockLastElems, gridSize.x);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// Add block's last element to its next block
		kernel_add << <gridSize, blockSize >> > (d_blockLastElems, d_output, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

__global__ void kernel_map(
	const unsigned int * const d_input,
	unsigned int * const d_output,
	const size_t numElems,
	const int mask,
	const int bitCount)
{
	unsigned int tId = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x + tId;

	if (idx < numElems)
	{
		d_output[idx] = ((d_input[idx] >> bitCount)&(numBins - 1)) == mask;
	}
}

__global__ void kernel_move(
	unsigned int* const d_inputVals, 
	unsigned int* const d_inputPos,
	unsigned int* const d_outputVals, 
	unsigned int* const d_outputPos,

	unsigned int* const d_elemsBin, 
	unsigned int *d_scanBin, 
	unsigned int *d_histogramBin, 
	
	const size_t numElems, 
	const int mask)
{
	unsigned int tId = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x + tId;

	if (idx < numElems && d_elemsBin[idx] == 1) 
	{
		unsigned int outputIdx = d_histogramBin[mask] + d_scanBin[idx];
		d_outputVals[outputIdx] = d_inputVals[idx];
		d_outputPos[outputIdx] = d_inputPos[idx];
	}
}

void your_sort(
	unsigned int*  d_inputVals,
	unsigned int*  d_inputPos,
	unsigned int*  d_outputVals,
	unsigned int*  d_outputPos,
	const size_t numElems)
{
	const dim3 blockSize(1024); // Thread dimension
	const dim3 gridSize(ceil((float)numElems / 1024));

	unsigned int *d_histogramBin, *d_scanBin, *d_elemsBin;

	checkCudaErrors(cudaMalloc((void**)&d_histogramBin, numBins * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_scanBin, numElems * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_elemsBin, numElems * sizeof(unsigned int)));

	for (int i = 0; i < 8 * (int)sizeof(unsigned int); i += numBits)
	{
		/*
		1) Histogram of the number of occurrences of each digit

		2) Exclusive Prefix Sum of Histogram

		3) Determine relative offset of each digit
		For example[0 0 1 1 0 0 1]
		->[0 1 0 1 2 3 2]

		4) Combine the results of steps 2 & 3 to determine the final
		output location for each element and move it there
		*/
		checkCudaErrors(cudaMemset(d_histogramBin, 0, numBins * sizeof(unsigned int)));

		/* 1) */
		kernel_histogram << <gridSize, blockSize, blockSize.x * sizeof(unsigned int) >> > (d_inputVals, d_histogramBin, i, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		/* 2) */
		prefix_sum(d_histogramBin, d_histogramBin, numBins);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		/* 3) */
		for (int j = 0; j < numBins; j++)
		{
			kernel_map << <gridSize, blockSize >> > (d_inputVals, d_elemsBin, numElems, j, i);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			/* 4) */
			prefix_sum(d_elemsBin, d_scanBin, numElems);

			kernel_move << <gridSize, blockSize >> > (d_inputVals, d_inputPos, d_outputVals, d_outputPos,
													  d_elemsBin, d_scanBin, d_histogramBin, numElems, j);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}

		std::swap(d_inputPos, d_outputPos);
		std::swap(d_inputVals, d_outputVals);

	}
	
	cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(int), cudaMemcpyDeviceToDevice);

	checkCudaErrors(cudaFree(d_histogramBin));
	checkCudaErrors(cudaFree(d_scanBin));
	checkCudaErrors(cudaFree(d_elemsBin));
}
