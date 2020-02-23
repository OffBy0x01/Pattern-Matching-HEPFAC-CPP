#include <iostream>
// TODO try with pitched, 2d texture etc.
/*
PITCHED
https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c

TEXTURE (GENERAL)
http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/218100902?pgno=2
https://stackoverflow.com/questions/13119813/bound-cuda-texture-reads-zero/13120722#13120722

http://www.subdude-site.com/WebPages_Local/RefInfo/Computer/Linux/LinuxGuidesOfOthers/linuxProgrammingGuides/pdfs/3Dgpu/3D_GPGPU_beginners_tutorial_2009_155pgs.pdf

*/

texture<int, 1, cudaReadModeElementType> tex_ref;
cudaArray* cuda_array;

__global__ void test(int* results){
    const int tidx = threadIdx.x;
    results[tidx] = tex1D(tex_ref, tidx) * 3;
}

int main(){
    int *host_arr;
    int host_arr_size = 8;

    // Create and populate host array
    cudaMallocHost((void**)&host_arr, host_arr_size * sizeof(int));
    for (int i = 0; i < host_arr_size; ++i){
        host_arr[i] = i * 2;
        std::cout << host_arr[i] << std::endl;
    }

    // bind to texture
    cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc <int >();
    cudaMallocArray(&cuda_array, &cuDesc, host_arr_size);
    cudaMemcpyToArray(cuda_array, 0, 0, host_arr , host_arr_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaBindTextureToArray(tex_ref , cuda_array);
    // Allocate results array
    int * result_arr;
    cudaMalloc((void**)&result_arr, host_arr_size*sizeof(int));

    // launch kernel
    test<<<1, host_arr_size>>>(result_arr);

    // fetch results
    cudaMemcpy(host_arr, result_arr, host_arr_size * sizeof(int), cudaMemcpyDeviceToHost);
    // print results
    for (int i = 0; i < host_arr_size; ++i){
        std::cout << host_arr[i] << std::endl;
    }

    // Tidy Up
    cudaUnbindTexture(tex_ref);
    cudaFreeHost(host_arr);
    cudaFreeArray(cuda_array);
    cudaFree(result_arr);
}
