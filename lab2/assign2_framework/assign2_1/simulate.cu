/*
 * simulate.cu
 *
 * SSID: 15225054 - Boris Vukajlovic
 * SSID: 14675218 - Marouan Bellari
 *
 * Program Description:
 * This program simulates a wave equation over a 1D array using CUDA for parallel processing on a GPU.
 * The simulation runs for a specified number of time steps, with each CUDA thread working on a different element of the array.
 * The program handles memory allocation and copying between host and device, and uses pointer swapping to update arrays efficiently between iterations.
 */

#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include "simulate.hh"

using namespace std;

#define C 0.15  // wave equation constant

/* Utility function, used to do error checking for CUDA calls
 *
 * Use this function like this:
 *     checkCudaCall(<cuda_call>);
 *
 * For example:
 *     checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
 *
 * Special case to check the result of the last kernel invocation:
 *     kernel<<<...>>>(...);
 *     checkCudaCall(cudaGetLastError());
 */
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

/* CUDA kernel to compute the wave equation for one time step.
 *
 * device_old: array of size i_max with data for t-1
 * device_current: array of size i_max with data for t
 * device_next: array of size i_max. Filled with t+1 in this kernel
 * i_max: size of the arrays
 */
__global__ void wave_kernel(const double *old_array, const double *current_array, double *next_array, long i_max) {

    long i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // start from index 1

    // ensure index is within valid range (excluding boundaries)
    if (i < i_max - 1) {
        next_array[i] = 2.0 * current_array[i] - old_array[i] +
                        C * (current_array[i - 1] - (2.0 * current_array[i] - current_array[i + 1]));
    }
}

/* Function that will simulate the wave equation, parallelized using CUDA.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * block_size: the number of threads per block
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const long i_max, const long t_max, const long block_size,
                 double *old_array, double *current_array, double *next_array) {

    // device arrays
    double *d_old_array = nullptr;
    double *d_current_array = nullptr;
    double *d_next_array = nullptr;

    size_t size = i_max * sizeof(double);

    // allocate device memory
    checkCudaCall(cudaMalloc((void **)&d_old_array, size));
    checkCudaCall(cudaMalloc((void **)&d_current_array, size));
    checkCudaCall(cudaMalloc((void **)&d_next_array, size));

    // copy data from host to device
    checkCudaCall(cudaMemcpy(d_old_array, old_array, size, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_current_array, current_array, size, cudaMemcpyHostToDevice));

    // calculate grid and block dimensions
    long num_elements = i_max - 2;  // Exclude boundary elements
    long grid_size = (num_elements + block_size - 1) / block_size;

    // time stepping loop
    for (long t = 0; t < t_max; t++) {
        // launch kernel
        wave_kernel<<<grid_size, block_size>>>(d_old_array, d_current_array, d_next_array, i_max);
        checkCudaCall(cudaGetLastError());

        // synchronize device
        checkCudaCall(cudaDeviceSynchronize());

        // swap pointers
        double *temp = d_old_array;
        d_old_array = d_current_array;
        d_current_array = d_next_array;
        d_next_array = temp;
    }

    // copy result back to host
    checkCudaCall(cudaMemcpy(current_array, d_current_array, size, cudaMemcpyDeviceToHost));

    // free device memory
    checkCudaCall(cudaFree(d_old_array));
    checkCudaCall(cudaFree(d_current_array));
    checkCudaCall(cudaFree(d_next_array));

    return current_array;
}
