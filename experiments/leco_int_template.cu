#include "../headers/common.h"
#include "../headers/codecfactory.h"
#include "../headers/caltime.h"
#include "../headers/lr.h"
#include "../headers/piecewise_cost_merge_integer_template_link.cuh"
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

typedef uint32_t leco_type;
// #define K 32

int random(int m)
{
    return rand() % m;
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
        std::cerr << "Error opening source file: " << filename << std::endl;
        return data;
    }

    std::cout << "Read Data\n";

    while (true) {
        T next;
        srcFile >> next;
        if (srcFile.eof()) {
            break; // End of file reached
        }
        if (srcFile.fail()) {
            std::cerr << "Failed to read data or data type mismatch." << std::endl;
            break; // Error in reading
        }
        data.emplace_back(next);
    }
    srcFile.close();

    std::cout << "data length = " << data.size() << std::endl;

    return data;
}

int main(int argc, const char* argv[])
{
    std::string method = "leco_cost";
    std::string source_file = std::string(argv[1]);

    int blocks = atoi(argv[2]);
    int delta = atoi(argv[3]);
    int model_size = atoi(argv[4]);
    bool binary = atoi(argv[5]);
    int K = atoi(argv[6]);

    printf("blocks = %d, K = %d\n", blocks, K);

    std::vector<leco_type> data;

    if (binary) {
        data = load_data_binary<leco_type>("../data/" + source_file);
    } else {
        data = load_data<leco_type>("../data/" + source_file);
    }

    size_t data_size = data.size();
    int overhead = delta;

    size_t chunk_size = 200000; 
    size_t num_chunks = (data_size + chunk_size - 1) / chunk_size; 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0;

    clock_t start_time_out, end_time_out;

    start_time_out = clock();

    int total_blocks = blocks;

    blocks = blocks / num_chunks;

    double decompress_time = 0.0;

    int *segment_cnt_total = nullptr;

    cudaMalloc((void**)&segment_cnt_total, sizeof(int));

    size_t total_byte_total_global = 0;

    printf("num_chunks = %lu\n", num_chunks);

    for (size_t chunk = 0; chunk < num_chunks; chunk++) {

        
        if (chunk == num_chunks - 1) {
            blocks = total_blocks - blocks * chunk;
        }

        size_t start_idx = chunk * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, data_size);
        size_t current_chunk_size = end_idx - start_idx;

        // printf("current_chunk_size = %lu\n", current_chunk_size);

        std::vector<leco_type> data_chunk(data.begin() + start_idx, data.begin() + end_idx);

        leco_type* data_gpu = nullptr;
        cudaError_t err = cudaMalloc(&data_gpu, sizeof(leco_type) * current_chunk_size);

        if (err != cudaSuccess) {
            printf("1 CUDA error: %s\n", cudaGetErrorString(err));
        }

        err = cudaMemcpy(data_gpu, data_chunk.data(), sizeof(leco_type) * current_chunk_size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            printf("2 CUDA error: %s\n", cudaGetErrorString(err));
        }

        Segment<int64_t>* segs_gpu = nullptr;
        cudaMalloc(&segs_gpu, sizeof(Segment<int64_t>) * current_chunk_size * 3);

        uint32_t* segment_index_gpu = nullptr;
        cudaMalloc(&segment_index_gpu, sizeof(uint32_t) * current_chunk_size * 3);

        uint32_t* segment_length_gpu = nullptr;
        cudaMalloc(&segment_length_gpu, sizeof(uint32_t) * current_chunk_size * 3);

        uint32_t* new_segment_index_gpu = nullptr;
        cudaMalloc(&new_segment_index_gpu, sizeof(uint32_t) * current_chunk_size * 3);

        uint32_t* new_segment_length_gpu = nullptr;
        cudaMalloc(&new_segment_length_gpu, sizeof(uint32_t) * current_chunk_size * 3);

        uint32_t* segment_index_total_gpu = nullptr;
        cudaMalloc(&segment_index_total_gpu, sizeof(uint32_t) * current_chunk_size * 3);

        uint32_t* segment_length_total_gpu = nullptr;
        cudaMalloc(&segment_length_total_gpu, sizeof(uint32_t) * current_chunk_size * 3);

        uint8_t* res_total_gpu = nullptr;
        cudaMalloc(&res_total_gpu, sizeof(uint8_t) * current_chunk_size * 8 + blocks * 1000);

        bool* signvec_gpu = nullptr;
        cudaMalloc(&signvec_gpu, sizeof(bool) * current_chunk_size * 3);

        leco_type* delta_final_gpu = nullptr;
        cudaMalloc(&delta_final_gpu, sizeof(leco_type) * current_chunk_size * 3);

        int block_size = current_chunk_size / blocks;
        int grid_size = (blocks + K - 1) / K;

        size_t *byte_count = nullptr;
        cudaMalloc((void**)&byte_count, sizeof(size_t) * blocks);

        cudaEventRecord(start, 0);



        kernel_var_encodeArray8_int<leco_type><<<grid_size, K>>>(data_gpu, block_size, blocks, current_chunk_size, segs_gpu, overhead,
                                                                  segment_index_gpu, segment_length_gpu,
                                                                  new_segment_index_gpu, new_segment_length_gpu,
                                                                  segment_index_total_gpu, segment_length_total_gpu,
                                                                  res_total_gpu, signvec_gpu, delta_final_gpu, segment_cnt_total, byte_count);





        // total_byte_total_global += byte_count_local;


        // printf("chunk_byte_count = %lu\n", chunk_byte_count);


        
        
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        size_t *byte_count_local = (size_t*)malloc(sizeof(size_t) * blocks);
        cudaMemcpy(byte_count_local, byte_count, sizeof(size_t) * blocks, cudaMemcpyDeviceToHost);

        size_t chunk_byte_count = 0;

        for (int i = 0; i < blocks; i++) {
            // printf("block %d byte count = %lu\n", i, byte_count_local[i]);

            chunk_byte_count += byte_count_local[i];

            total_byte_total_global += byte_count_local[i];
        }

        free(byte_count_local);


        cudaFree(data_gpu);
        cudaFree(segs_gpu);
        cudaFree(segment_index_gpu);
        cudaFree(segment_length_gpu);
        cudaFree(new_segment_index_gpu);
        cudaFree(new_segment_length_gpu);
        cudaFree(byte_count);



        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

    

        total_time += milliseconds / 1000.0;


        leco_type* recover_gpu = nullptr;
        cudaMalloc(&recover_gpu, sizeof(leco_type) * current_chunk_size);

        cudaEventRecord(start, 0);

        kernel_decodeArray8_var<leco_type><<<grid_size, K>>>(res_total_gpu, recover_gpu, blocks, block_size, segment_index_total_gpu, segment_length_total_gpu, current_chunk_size);

        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);
        decompress_time += milliseconds / 1000.0;

        leco_type* recover = (leco_type*)malloc(sizeof(leco_type) * current_chunk_size);
        cudaMemcpy(recover, recover_gpu, sizeof(leco_type) * current_chunk_size, cudaMemcpyDeviceToHost);

        for (size_t j = 0; j < current_chunk_size; j++) {
            if (data_chunk[j] != recover[j]) {
                std::cout << "解压失败，块 " << chunk + 1 << " 第 " << j << " 个数据不匹配" << std::endl;
                break;
            }
        }


        cudaFree(recover_gpu);
        cudaFree(segment_index_total_gpu);
        cudaFree(segment_length_total_gpu);
        cudaFree(res_total_gpu);
        cudaFree(signvec_gpu);
        cudaFree(delta_final_gpu);

        free(recover);
    }

    cudaEventSynchronize(stop);

    cudaError_t err = cudaDeviceSynchronize(); 
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    int *cpu_segment_cnt_total = (int *)malloc(sizeof(int));

    cudaMemcpy(cpu_segment_cnt_total, segment_cnt_total, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(segment_cnt_total);

    std::cout << "Total segment = " << *cpu_segment_cnt_total << std::endl;

    end_time_out = clock();

    std::cout <<  double(end_time_out - start_time_out) / CLOCKS_PER_SEC << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout <<  total_byte_total_global << std::endl;

    std::cout <<  double(data_size * sizeof(leco_type)) / total_byte_total_global << std::endl;
    

    std::cout <<  total_time <<std::endl;
    std::cout <<  decompress_time << std::endl;

    return 0;
}