#include "file.hh"
#include "timer.hh"
#include <cctype>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

#define MAX_KEY_LENGTH 256
__constant__ int deviceKey[MAX_KEY_LENGTH];

static void checkCudaCall(cudaError_t result) {
  if (result != cudaSuccess) {
    cerr << "cuda error: " << cudaGetErrorString(result) << endl;
    exit(EXIT_FAILURE);
  }
}

// encryptKernel: CUDA kernel that encrypts data using Caesar or Vigenère cipher.
// The kernel processes the input data on the GPU and writes the encrypted result
// to the output array based on the provided encryption key.
__global__ void encryptKernel(char *deviceDataIn, char *deviceDataOut,
                              int length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  char input = deviceDataIn[idx];

  if (length == 1) {
    if ((input >= 'A' && input <= 'Z') || (input >= 'a' && input <= 'z')) {
      int shift = deviceKey[0];

      if (input >= 'a' && input <= 'z') {
        deviceDataOut[idx] = 'a' + (input - 'a' + shift) % 26;
      } else if (input >= 'A' && input <= 'Z') {
        deviceDataOut[idx] = 'A' + (input - 'A' + shift) % 26;
      }
    } else {
      deviceDataOut[idx] = input;
    }
  }

  if (length > 1) {
    if ((input >= 'A' && input <= 'Z') || (input >= 'a' && input <= 'z')) {
      int shift = deviceKey[idx % length];
      if (input >= 'a' && input <= 'z') {
        deviceDataOut[idx] = 'a' + (input - 'a' + shift) % 26;
      } else if (input >= 'A' && input <= 'Z') {
        deviceDataOut[idx] = 'A' + (input - 'A' + shift) % 26;
      }
    } else {
      deviceDataOut[idx] = input;
    }
  }
}

// decryptKernel: CUDA kernel that decrypts data encrypted with Caesar or Vigenère cipher.
// The kernel processes the input data on the GPU and writes the decrypted result
// to the output array using the provided decryption key.
__global__ void decryptKernel(char *deviceDataIn, char *deviceDataOut,
                              int length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  char input = deviceDataIn[idx];

  if (length == 1) {
    if ((input >= 'A' && input <= 'Z') || (input >= 'a' && input <= 'z')) {
      int shift = deviceKey[0] % 26;
      if (input >= 'a' && input <= 'z') {
        deviceDataOut[idx] = 'a' + (input - 'a' - shift + 26) % 26;
      } else if (input >= 'A' && input <= 'Z') {
        deviceDataOut[idx] = 'A' + (input - 'A' - shift + 26) % 26;
      }
    } else {
      deviceDataOut[idx] = input;
    }
  }

  if (length >= 1) {
    if ((input >= 'A' && input <= 'Z') || (input >= 'a' && input <= 'z')) {
      int shift = deviceKey[idx % length] % 26;
      if (input >= 'a' && input <= 'z') {
        deviceDataOut[idx] = 'a' + (input - 'a' - shift + 26) % 26;
      } else if (input >= 'A' && input <= 'Z') {
        deviceDataOut[idx] = 'A' + (input - 'A' - shift + 26) % 26;
      }
    } else {
      deviceDataOut[idx] = input;
    }
  }
}


// EncryptSeq: Performs sequential encryption of input data using Caesar or Vigenère cipher.
// The function processes the input array, applying the encryption key to generate
// the encrypted output in the output array.
int EncryptSeq(int n, char *data_in, char *data_out, int key_length, int *key) {

  int i, valid_index = 0;
  timer sequentialTime = timer("Sequential encryption");

  sequentialTime.start();
  for (i = 0; i < n; i++) {
    if (!isalpha(data_in[i])) {
      data_out[i] = data_in[i];
      continue;
    }

    if (key_length == 1) {
      if (islower(data_in[i])) {
        data_out[i] = 'a' + ((data_in[i] - 'a' + key[0]) % 26);
      } else if (isupper(data_in[i])) {
        data_out[i] = 'A' + ((data_in[i] - 'A' + key[0]) % 26);
      }
    } else {
      int key_index = valid_index % key_length;
      if (islower(data_in[i])) {
        data_out[i] = 'a' + ((data_in[i] - 'a' + key[key_index]) % 26);
      } else if (isupper(data_in[i])) {
        data_out[i] = 'A' + ((data_in[i] - 'A' + key[key_index]) % 26);
      }
      valid_index++;
    }
  }
  sequentialTime.stop();

  cout << fixed << setprecision(6);
  cout << "Encryption (sequential): \t\t" << sequentialTime.getElapsed()
       << " seconds." << endl;

  return 0;
}

// DecryptSeq: Performs sequential decryption of data encrypted with Caesar or Vigenère cipher.
// The function processes the input array, applying the decryption key to restore
// the original data in the output array.
int DecryptSeq(int n, char *data_in, char *data_out, int key_length, int *key) {

  int i, valid_index = 0;
  timer sequentialTime = timer("Sequential decryption");

  sequentialTime.start();
  for (i = 0; i < n; i++) {
    if (!isalpha(data_in[i])) {
      data_out[i] = data_in[i];
      continue;
    }

    if (key_length == 1) {
      int shift = key[0] % 26;
      if (islower(data_in[i])) {
        data_out[i] = 'a' + ((data_in[i] - 'a' - shift + 26) % 26);
      } else if (isupper(data_in[i])) {
        data_out[i] = 'A' + ((data_in[i] - 'A' - shift + 26) % 26);
      }
    } else {
      int key_index = valid_index % key_length;
      int shift = key[key_index] % 26;
      if (islower(data_in[i])) {
        data_out[i] = 'a' + ((data_in[i] - 'a' - shift + 26) % 26);
      } else if (isupper(data_in[i])) {
        data_out[i] = 'A' + ((data_in[i] - 'A' - shift + 26) % 26);
      }
      valid_index++;
    }
  }

  sequentialTime.stop();

  cout << fixed << setprecision(6);
  cout << "Decryption (sequential): \t\t" << sequentialTime.getElapsed()
       << " seconds." << endl;

  return 0;
}

int EncryptCuda(int n, char *data_in, char *data_out, int key_length,
                int *key) {
  int threadBlockSize = 4;

  cudaMemcpyToSymbol(deviceKey, key, (key_length + 1) * sizeof(int));

  char *deviceDataIn = NULL;
  checkCudaCall(cudaMalloc((void **)&deviceDataIn, n * sizeof(char)));
  if (deviceDataIn == NULL) {
    cout << "could not allocate memory!" << endl;
    return -1;
  }
  char *deviceDataOut = NULL;
  checkCudaCall(cudaMalloc((void **)&deviceDataOut, n * sizeof(char)));
  if (deviceDataOut == NULL) {
    checkCudaCall(cudaFree(deviceDataIn));
    cout << "could not allocate memory!" << endl;
    return -1;
  }

  timer kernelTime1 = timer("kernelTime");
  timer memoryTime = timer("memoryTime");

  memoryTime.start();
  checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n * sizeof(char),
                           cudaMemcpyHostToDevice));
  memoryTime.stop();

  kernelTime1.start();

  int gridSize;
  if (n < threadBlockSize) {
    gridSize = 1;
  } else {
    gridSize = (n + threadBlockSize - 1) / threadBlockSize;
  }
  encryptKernel<<<gridSize, threadBlockSize>>>(deviceDataIn, deviceDataOut,
                                               key_length);
  cudaDeviceSynchronize();
  kernelTime1.stop();

  checkCudaCall(cudaGetLastError());

  memoryTime.start();
  checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char),
                           cudaMemcpyDeviceToHost));
  memoryTime.stop();

  checkCudaCall(cudaFree(deviceDataIn));
  checkCudaCall(cudaFree(deviceDataOut));

  cout << fixed << setprecision(6);
  cout << "Encrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds."
       << endl;
  cout << "Encrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds."
       << endl;

  return 0;
}

int DecryptCuda(int n, char *data_in, char *data_out, int key_length,
                int *key) {
  int threadBlockSize = 512;

  char *deviceDataIn = NULL;
  checkCudaCall(cudaMalloc((void **)&deviceDataIn, n * sizeof(char)));
  if (deviceDataIn == NULL) {
    cout << "could not allocate memory!" << endl;
    return -1;
  }
  char *deviceDataOut = NULL;
  checkCudaCall(cudaMalloc((void **)&deviceDataOut, n * sizeof(char)));
  if (deviceDataOut == NULL) {
    checkCudaCall(cudaFree(deviceDataIn));
    cout << "could not allocate memory!" << endl;
    return -1;
  }

  timer kernelTime1 = timer("kernelTime");
  timer memoryTime = timer("memoryTime");

  memoryTime.start();
  checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n * sizeof(char),
                           cudaMemcpyHostToDevice));
  memoryTime.stop();

  kernelTime1.start();

  int gridSize;
  if (n < threadBlockSize) {
    gridSize = 1;
  } else {
    gridSize = (n + threadBlockSize - 1) / threadBlockSize;
  }
  decryptKernel<<<gridSize, threadBlockSize>>>(deviceDataIn, deviceDataOut,
                                               key_length);
  cudaDeviceSynchronize();
  kernelTime1.stop();

  checkCudaCall(cudaGetLastError());

  memoryTime.start();
  checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char),
                           cudaMemcpyDeviceToHost));
  memoryTime.stop();

  checkCudaCall(cudaFree(deviceDataIn));
  checkCudaCall(cudaFree(deviceDataOut));

  cout << fixed << setprecision(6);
  cout << "Decrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds."
       << endl;
  cout << "Decrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds."
       << endl;

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " key..." << endl;
    cout << " - key: one or more values for the encryption key, separated "
            "by spaces"
         << endl;

    return EXIT_FAILURE;
  }

  int key_length = argc - 1;
  int *enc_key = new int[key_length];
  for (int i = 0; i < key_length; i++) {
    enc_key[i] = atoi(argv[i + 1]);
  }

  int n;
  n = fileSize("original.data");
  if (n == -1) {
    cout << "File not found! Exiting ... " << endl;
    exit(0);
  }

  char *data_in = new char[n];
  char *data_out = new char[n];
  readData("original.data", data_in);

  cout << "Encrypting a file of " << n << " characters." << endl;

  EncryptSeq(n, data_in, data_out, key_length, enc_key);
  writeData(n, "sequential.data", data_out);

  EncryptCuda(n, data_in, data_out, key_length, enc_key);
  writeData(n, "cuda.data", data_out);

  readData("cuda.data", data_in);

  cout << "Decrypting a file of " << n << "characters" << endl;
  DecryptSeq(n, data_in, data_out, key_length, enc_key);
  writeData(n, "sequential_recovered.data", data_out);
  DecryptCuda(n, data_in, data_out, key_length, enc_key);
  writeData(n, "recovered.data", data_out);

  delete[] data_in;
  delete[] data_out;

  return 0;
}

