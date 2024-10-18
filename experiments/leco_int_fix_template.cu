#include "../headers/common.h"
#include "../headers/codecfactory.h"
#include "../headers/caltime.h"
#include "../headers/lr.h"
#include "../headers/piecewise_fix_integer_template.cuh"
#include "../headers/piecewise_fix_integer_template_float.h"
#include "../headers/piecewise_cost_merge_integer_template_double.h"
#include "../headers/FOR_integer_template.h"
#include "../headers/delta_integer_template.h"
#include "../headers/delta_cost_integer_template.h"
#include "../headers/delta_cost_merge_integer_template.h"
#include "../headers/piecewise_cost_merge_integer_template_test.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

typedef uint64_t leco_type;

#define M 1024
#define K 256

int random(int m)
{
    return rand() % m;
}

int hash(int x)
{
    return x / K;
}


template <typename T>
static std::vector<T> load_data_binary(const std::string& filename,
    bool print = true) {
    std::vector<T> data;

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "unable to open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    // Read size.
    uint64_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
    data.resize(size);
    // Read values.
    in.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    in.close();

    return data;
}

template <typename T>
static std::vector<T> load_data(const std::string& filename) {
    std::vector<T> data;
    std::ifstream srcFile(filename, std::ios::in);
    if (!srcFile) {
        std::cout << "error opening source file." << std::endl;
        return data;
    }

    int cnt = 0;

    while (srcFile.good()) {
        T next;
        srcFile >> next;
        if (!srcFile.good()) { break; }
        data.emplace_back(next);

        // cnt ++ ;

        // if (cnt % 10000000 == 0)
        //     std::cout << next << std::endl;

    }
    srcFile.close();

    return data;
}

int main(int argc, const char* argv[])
{
    using namespace Codecset;
    std::string method = "piecewise_fix_op_max";
    std::string source_file = std::string(argv[1]);
    int blocks = atoi(argv[2]);
    int model_size = atoi(argv[3]);
    bool binary = atoi(argv[4]);
    leco_type filter1 = 0;
    leco_type filter2 = 0;
    leco_type base = 0;
    bool filter_experiment = false;
    bool filter_close_experiment = false;
    if (argc > 5)
    {
        filter1 = atoll(argv[5]);
        filter_experiment = true;
    }
    if (argc > 6)
    {
        filter2 = atoll(argv[6]);
        filter_experiment = false;
        filter_close_experiment = true;
        base = atoll(argv[7]);
    }
    // alternatives : Delta_int, Delta_cost, Delta_cost_merge, FOR_int, Leco_int, Leco_cost, Leco_cost_merge_hc,  Leco_cost_merge, Leco_cost_merge_double

    std::vector<leco_type> data;
    if(binary){
        data = load_data_binary<leco_type>("../data/" + source_file);
    }
    else{
        data = load_data<leco_type>("../data/" + source_file);
    }
    int N = data.size();

    std::cout << "data size = " << N << std::endl;

    int block_size = data.size() / blocks;
    blocks = (data.size() + block_size - 1) / block_size;
    if (blocks * block_size < N)
    {
        blocks++;
    } // handle with the last block, maybe < block_size


    std::vector<uint8_t*> block_start_vec;

    // todo CUDA Accelerating!


    uint8_t **devicePointers = nullptr;
    uint32_t *device_size_array = nullptr;
    leco_type *data_gpu = nullptr;
    leco_type *delta_gpu = nullptr;
    bool *signvec = nullptr;
    
    uint8_t **tempHostPointers = (uint8_t **)malloc(blocks * sizeof(uint8_t *));// 在主机上创建临时指针数组
    uint32_t *size_array = (uint32_t *)malloc(blocks * sizeof(uint32_t));

    memset(size_array, 0, blocks * sizeof(uint32_t));
    
    cudaMalloc((void **)&device_size_array, blocks * sizeof(uint32_t));
    cudaMalloc((void **)&devicePointers, blocks * sizeof(uint8_t *));
    cudaMalloc((void **)&data_gpu, N * sizeof(leco_type));
    cudaMalloc((void **)&delta_gpu, N * sizeof(leco_type));
    cudaMalloc((void **)&signvec, N * sizeof(bool));

    for (int i = 0; i < blocks; i++) {
        cudaMalloc((void**)&tempHostPointers[i], block_size * sizeof(leco_type) * 4);
    }

    cudaMemcpy(devicePointers, tempHostPointers, blocks * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(device_size_array, size_array, blocks * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(data_gpu, data.data(), N * sizeof(leco_type), cudaMemcpyHostToDevice);

    int grid_size = (blocks + K - 1) / K;

    double valid = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    kernel_encodeArray8_int<<<grid_size, K>>>(data_gpu, block_size, blocks, N, devicePointers, device_size_array, delta_gpu, signvec);

    cudaEventRecord(stop);

    uint8_t * ptr_8 =  (uint8_t*)malloc(block_size * sizeof(uint64_t));

    uint64_t totalsize = 0;

    cudaMemcpy(size_array, device_size_array, blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);


    for (int i = 0;i < blocks;i ++ )
    {
        totalsize += size_array[i];
        // outfile << size_array[i] << std::endl;
    }

    // outfile.close();

    

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();

    printf("totalsize = %lu, time = %f\n", totalsize, milliseconds);

    cudaFree(device_size_array);
    cudaFree(data_gpu);
    cudaFree(delta_gpu);
    cudaFree(signvec);


    double origin_size = (sizeof(leco_type) * N * 1.0);
    double total_model_size = model_size * blocks;
    double cr_wo_model = (totalsize - total_model_size) * 100.0 / origin_size;
    double cr_model = total_model_size * 100.0 / origin_size;
    double compressrate = (totalsize) * 100.0 / origin_size;

    bool flag = true;

    leco_type *recover = (leco_type *)malloc(data.size() * sizeof(leco_type));

    double totaltime = 0.0;

    int repeat = 10;

    leco_type *compressed_data_gpu = nullptr;

    cudaMalloc((void **)&compressed_data_gpu, data.size() * sizeof(leco_type));

    int decode_grid_size = (blocks + K - 1) / K;

    cudaEvent_t dc_start, dc_stop;

    cudaEventCreate(&dc_start);
    cudaEventCreate(&dc_stop);

    cudaEventRecord(dc_start);

    kernel_decodeArray8<<<decode_grid_size, K>>>(devicePointers, compressed_data_gpu, blocks, block_size, N);

    cudaEventRecord(dc_stop);

    cudaEventSynchronize(dc_stop);

    float dc_time = 0;
    

    cudaEventElapsedTime(&dc_time, dc_start, dc_stop);
    cudaDeviceSynchronize();
    

    printf("Decompress Time = %fsm\n", dc_time);

    cudaMemcpy(recover, compressed_data_gpu, data.size() * sizeof(leco_type), cudaMemcpyDeviceToHost);

    cudaFree(compressed_data_gpu);

    cudaDeviceSynchronize();
    


    for (int j = 0; j < N; j++)
    {
        if (data[j ] != recover[j])
        {
            std::cout << "something wrong! decompress all failed" << std::endl;
            flag = false;
        }
    }

    std::cout << "random access decompress!" << std::endl;
    uint32_t * ra_pos = (uint32_t *)malloc(N * sizeof(uint32_t));
    leco_type *res = (leco_type *)malloc(N * sizeof(leco_type));

    for (int i = 0;i < N;i++) {
        ra_pos[i] = random(N);
    }


    
    
    float randomaccesstime = 0.0;

    cudaEvent_t ra_start, ra_end;

    cudaEventCreate(&ra_start);
    cudaEventCreate(&ra_end);

    int random_access_grid_size = (N + K - 1) / K;

    uint32_t *gpu_ra_pos = nullptr;

    cudaMalloc((void **)&gpu_ra_pos, sizeof(uint32_t) * N);
    cudaMemcpy(gpu_ra_pos, ra_pos, sizeof(uint32_t) * N, cudaMemcpyHostToDevice);

    int *bucket_cnt = (int *)malloc(sizeof(int) * blocks);
    memset(bucket_cnt, 0, sizeof(int) * blocks);

    int *bucket_cnt_gpu = nullptr;     // 计算每个块的访问次数
 
    cudaMalloc((void **)&bucket_cnt_gpu, sizeof(int) * blocks);
    cudaMemcpy(bucket_cnt_gpu, bucket_cnt, sizeof(int) * blocks, cudaMemcpyHostToDevice);

    preprocess_bucket_cnt<<<(N + K - 1) / K, K>>>(gpu_ra_pos, bucket_cnt_gpu, N, block_size);

    cudaMemcpy(bucket_cnt, bucket_cnt_gpu, sizeof(int) * blocks, cudaMemcpyDeviceToHost);


    // for (int i = 0;i < blocks; i ++ )
    // {
    //     if (bucket_cnt[i] > 400)
    //         std::cout << i << " " << bucket_cnt[i] << std::endl;
    // }

    leco_type *gpu_res = nullptr;
    cudaMalloc((void **)&gpu_res, sizeof(leco_type) * N);

    cudaEventRecord(ra_start);

    
    
    kernel_randomdecodeArray<<<random_access_grid_size, K, K * sizeof(leco_type)>>>(devicePointers, gpu_ra_pos, gpu_res, N, block_size);


    cudaEventRecord(ra_end);

    cudaEventSynchronize(ra_end);

    float ra_time = 0.0;

    cudaEventElapsedTime(&ra_time, ra_start, ra_end);

    cudaDeviceSynchronize();

    printf("Random access time = %fsm\n", ra_time);

    cudaMemcpy(res, gpu_res, sizeof(leco_type) * N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    int error_count2 = 0;

    for (int i = 0;i < N;i ++ )
    {
        if (data[ra_pos[i]] != res[i])
        {
            // std::cout << "num: " << ra_pos[i] << " true is: " << std::hex << data[ra_pos[i]] << " predict is: " << std::hex <<  res[i] << std::endl;
            // flag = false;
            // std::cout << "something wrong! random access failed" << std::endl;
            

            std::cout << "pos = " << ra_pos[i] << " res = " << res[i] << std::endl;
            error_count2 ++ ;
        }
        // if (!flag)
        // {
        //     break;
        // }
    }

    std::cout << "Error Count = " << error_count2 << std::endl;
}
