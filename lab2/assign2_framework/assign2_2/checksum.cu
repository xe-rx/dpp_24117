#include <cuda_device_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

#include "timer.hh"
#include "file.hh"

using namespace std;

static void checkCudaCall(cudaError_t result) {
  if (result != cudaSuccess) {
      cerr << "cuda error: " << cudaGetErrorString(result) << endl;
      exit(EXIT_FAILURE);
  }
}

// checksumKernel: CUDA kernel to compute an additive checksum for input data.
// Computes the checksum in shared memory and aggregates results using atomic operations.
__global__ void checksumKernel(unsigned int* result, const unsigned int* deviceDataIn, int n) {
    extern __shared__ unsigned int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    unsigned int sum = 0;

    if (idx < n) sum += deviceDataIn[idx];
    if (idx + blockDim.x < n) sum += deviceDataIn[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// checksumSeq: Sequentially computes an additive checksum of the input data.
unsigned int checksumSeq (int n, unsigned int* data_in) {
    int i;
    timer sequentialTime = timer("Sequential checksum");
    unsigned int checksum = 0;

    sequentialTime.start();
    for (i=0; i<n; i++) {
        checksum += data_in[i];
    }
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Checksum (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return checksum;
}

unsigned int checksumCuda (int n, unsigned int* data_in) {
    int threadBlockSize = 512;

    unsigned int* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(unsigned int)));
    if (deviceDataIn == NULL) {
        cout << "Could not allocate input data on GPU." << endl;
        exit(1);
    }
    unsigned int* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, sizeof(unsigned int)));
    if (deviceResult == NULL) {
        cout << "Could not allocate result integer on GPU." << endl;
        exit(1);
    }

    timer kernelTime  = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(unsigned int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    kernelTime.start();
    int gridSize = (n + threadBlockSize - 1 * 2) / threadBlockSize * 2;
    if (gridSize == 0) {
      gridSize = 1;
    }
    checksumKernel<<<gridSize, threadBlockSize,  threadBlockSize * sizeof(unsigned int)>>>(deviceResult, deviceDataIn, n);
    cudaDeviceSynchronize();
    kernelTime.stop();

    checkCudaCall(cudaGetLastError());

    unsigned int result;
    checkCudaCall(cudaMemcpy(&result, deviceResult, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceResult));

    cout << fixed << setprecision(6);
    cout << "Kernel: \t\t" << kernelTime.getElapsed() << " seconds." << endl;
    cout << "Memory: \t\t" << memoryTime.getElapsed() << " seconds." << endl;

    return result;
}

int main(int argc, char* argv[]) {
    int n;
    char* mode;
    char* fileName;

    if (argc == 3) {
        fileName = argv[1];
        mode = argv[2];

        cout << "Running in '" << mode << "' mode" << endl;
        cout << "Opening file " << fileName << endl;
    } else {
        cout << "Usage: " << argv[0] << " filename mode" << endl;
        cout << " - filename: name of the file for which the checksum will be "
                "computed." << endl;
        cout << " - mode: one of the three modes for which the program can "
                "run." << endl;
        cout << "   Available options are:" << endl;
        cout << "    * seq: only runs the sequential implementation" << endl;
        cout << "    * cuda: only runs the parallelized implementation" << endl;
        cout << "    * both: runs both the sequential and the parallelized "
                "implementation" << endl;

        return EXIT_FAILURE;
    }
    n = fileSize(fileName);
    if (n == -1) {
        cerr << "File '" << fileName << "' not found" << endl;
        exit(EXIT_FAILURE);
    }

    char* data_in = new char[n];
    readData(fileName, data_in);
    unsigned int *data_in_raw = new unsigned int[n];
    for (int i = 0; i < n; i++){
        data_in_raw[i] = data_in[i];
    }

    if (strcmp(mode, "seq") == 0){
        unsigned int checksum = checksumSeq(n, data_in_raw);
        cout << "Sequential checksum: " << checksum << endl;
    } else if (strcmp(mode, "cuda") == 0) {
        unsigned int checksum = checksumCuda(n, data_in_raw);
        cout << "CUDA checksum: " << checksum << endl;
    } else if (strcmp(mode, "both") == 0){
        unsigned int checksum = checksumCuda(n, data_in_raw);
        cout << "CUDA checksum: " << checksum << endl;
        checksum = checksumSeq(n, data_in_raw);
        cout << "Sequential checksum: " << checksum << endl;
    } else {
        cerr << "Unknown mode '" << mode << "'; only accepts 'seq', 'cuda' or "
                "'both'" << endl;
        delete[] data_in;
        delete[] data_in_raw;
        exit(EXIT_FAILURE);
    }

    delete[] data_in;
    delete[] data_in_raw;
    return EXIT_SUCCESS;
}

