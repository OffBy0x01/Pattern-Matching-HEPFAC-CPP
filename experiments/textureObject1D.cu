#include <iostream>

#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess){printf("\nCUDA Error in %s on line %s: %s (Error Number = %d)\n", __FILE__, __LINE__, cudaGetErrorString(a), a); cudaDeviceReset(); exit(1);} }

__device__ int
popcu(unsigned int temp){
    // This is the fastest popcount for 32-bit types as detailed in "Software Optimization Guide for AMD64 Processors"(179-180)
    temp = temp - ((temp >> 1) & 0b01010101010101010101010101010101);
    temp = (temp & 0b00110011001100110011001100110011) + ((temp >> 2) & 0b00110011001100110011001100110011);
    return (((temp + (temp >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
}

__device__ extern int // Count bits set in a given reduced node
popcount_node_gpu(unsigned bitmap, int idx){
    int count = 0;               // number of set bits
    int store = 0;
    // This uses the same popcount as above but masks to the required index
    store = bitmap & ((1<<idx)-1);
    count = popcu(store);
    return count;
}

__global__ void test(cudaTextureObject_t tex ,int* results){
    int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    int temp = tex1Dfetch<int>(tex, threadIdx.x);
    results[tidx] = popcount_node_gpu(temp, tidx%32);
}

template<typename _type>
cudaTextureObject_t * createTextureObject(_type array1D[], size_t array1D_size){
    cudaTextureObject_t * tex_p = new cudaTextureObject_t();
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = array1D;
    resDesc.res.linear.sizeInBytes = array1D_size;
    resDesc.res.linear.desc = cudaCreateChannelDesc<_type>();
    // Create texture description
    struct cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(tex_p, &resDesc, &texDesc, NULL);
    return tex_p;
}

int main(){
    int *host_arr;
    const int host_arr_size = 128;

    // Create and populate host array
    std::cout << "Host:" << std::endl;
    CUDA_CALL(cudaMallocHost(&host_arr, host_arr_size*sizeof(int)));
    for (int i = 0; i < host_arr_size; ++i){
        host_arr[i] = i;
        std::cout << host_arr[i] << std::endl;
    }

    // // Create texture
    cudaTextureObject_t * tex_p = createTextureObject<int>(host_arr, host_arr_size*sizeof(int));

    // Allocate results array
    int * result_arr;
    CUDA_CALL(cudaMalloc(&result_arr, host_arr_size*sizeof(int)));

    // launch test kernel
    test<<<1, host_arr_size>>>(*tex_p, result_arr);

    // fetch results
    std::cout << "Device:" << std::endl;
    CUDA_CALL(cudaMemcpy(host_arr, result_arr, host_arr_size*sizeof(int), cudaMemcpyDeviceToHost));
    // print results
    for (int i = 0; i < host_arr_size; ++i){
        std::cout << host_arr[i] << std::endl;
    }

    // Tidy Up
    cudaDestroyTextureObject(*tex_p);
    cudaFreeHost(host_arr);
    cudaFree(result_arr);
}
