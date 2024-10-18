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

// int main(int argc, const char* argv[])
// {
//     std::string method = "leco_cost";
//     std::string source_file = std::string(argv[1]);

//     // printf("POPOP\n");
//     int blocks = atoi(argv[2]);
//     int delta = atoi(argv[3]);

//     // printf("KOKOK\n");
//     int model_size = atoi(argv[4]);
//     bool binary = atoi(argv[5]);
//     // alternatives : Delta_int, Delta_cost, Delta_cost_merge, FOR_int, Leco_int, Leco_cost, Leco_cost_merge_hc,  Leco_cost_merge, Leco_cost_merge_double
//     std::vector<leco_type> data;

//     // printf("LOLOL\n");

//     if (binary)
//     {
//          data = load_data_binary<leco_type>( "../data/"+ source_file);
//     }
//     else
//     {

//         data = load_data<leco_type>( "../data/"+ source_file);
//     }

    

//     int overhead = delta;

//     size_t data_size = data.size();

//     // for (int i = 0;i < data_size;i ++ )
//     // {
//     //     if (data[i] == 2310041185)
//     //     {
//     //         printf("i = %d\n", i);
//     //         break;
//     //     }

//     // }

//     int block_size = data.size() / blocks;
//     blocks = data.size() / block_size;
//     if (blocks * block_size < data_size)
//     {
//         blocks++;
//     } 

//     leco_type *data_gpu = nullptr;
//     cudaError_t err = cudaMalloc(&data_gpu, sizeof(leco_type) * data_size);

//     if (err != cudaSuccess) {
//         printf("1 CUDA error: %s\n", cudaGetErrorString(err));
//     }

//     err = cudaMemcpy(data_gpu, data.data(), sizeof(leco_type) * data_size, cudaMemcpyHostToDevice);

//     if (err != cudaSuccess) {
//         printf("2 CUDA error: %s\n", cudaGetErrorString(err));
//     }


//     Segment<int64_t> *segs_gpu = nullptr;
//     cudaMalloc(&segs_gpu, sizeof(Segment<int64_t>) * data_size * 3);

//     printf("Segment size = %d\n", sizeof(Segment<int64_t>));

//     int grid_size = (blocks + K - 1) / K;

//     printf("data_size = %lu, block_size = %d, blocks = %d, grid_size = %d\n", data_size, block_size, blocks, grid_size);

//     uint32_t *segment_index_gpu = nullptr;
//     cudaMalloc(&segment_index_gpu, sizeof(uint32_t) * data_size * 3);

//     uint32_t *segment_length_gpu = nullptr;
//     cudaMalloc(&segment_length_gpu, sizeof(uint32_t) * data_size * 3);


//     uint32_t *new_segment_index_gpu = nullptr;
//     cudaMalloc(&new_segment_index_gpu, sizeof(uint32_t) * data_size * 3);

//     uint32_t *new_segment_length_gpu = nullptr;
//     cudaMalloc(&new_segment_length_gpu, sizeof(uint32_t) * data_size * 3);

//     uint32_t *segment_index_total_gpu = nullptr;
//     cudaMalloc(&segment_index_total_gpu, sizeof(uint32_t) * data_size * 3);

//     uint32_t *segment_length_total_gpu = nullptr;
//     cudaMalloc(&segment_length_total_gpu, sizeof(uint32_t) * data_size * 3);

//     uint8_t *res_total_gpu = nullptr;
//     cudaMalloc(&res_total_gpu, sizeof(uint8_t) * data_size * 8 + blocks * 1000);

//     bool *signvec_gpu = nullptr;
//     cudaMalloc(&signvec_gpu, sizeof(bool) * data_size * 3);

//     leco_type *delta_final_gpu = nullptr;
//     cudaMalloc(&delta_final_gpu, sizeof(leco_type) * data_size * 3);

//     std::cout << "Start Encoding\n";

//     // CUDA 事件对象
//     cudaEvent_t start, stop;

//     // 创建 CUDA 事件
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start, 0);

//     kernel_var_encodeArray8_int<leco_type><<<grid_size, K>>>(data_gpu, block_size, blocks, data_size, segs_gpu, overhead, segment_index_gpu, segment_length_gpu, new_segment_index_gpu, new_segment_length_gpu, segment_index_total_gpu, segment_length_total_gpu, res_total_gpu, signvec_gpu, delta_final_gpu);

//     cudaEventRecord(stop, 0);

//     cudaEventSynchronize(stop);

//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);

//     std::cout << "CUDA 内核执行时间: " << milliseconds / 1000.0 << " 秒" << std::endl;

//     cudaDeviceSynchronize();

//     std::cout << "Finish Encoding\n";

//     cudaFree(data_gpu);
//     cudaFree(segs_gpu);
//     cudaFree(segment_index_gpu);
//     cudaFree(segment_length_gpu);
//     cudaFree(new_segment_index_gpu);
//     cudaFree(new_segment_length_gpu);

//     leco_type *recover_gpu = nullptr;
//     cudaMalloc(&recover_gpu, sizeof(leco_type) * data_size);

//     kernel_decodeArray8_var<leco_type><<<grid_size, K>>>(res_total_gpu, recover_gpu, blocks, block_size, segment_index_total_gpu, segment_length_total_gpu, data_size);

//     cudaDeviceSynchronize();

//     leco_type *recover = (leco_type *)malloc(sizeof(leco_type) * data_size);

//     cudaMemcpy(recover, recover_gpu, sizeof(leco_type) * data_size, cudaMemcpyDeviceToHost);

//     cudaDeviceSynchronize();

//     bool flag = true;

//     size_t count = 0;

//     std::ofstream out("recover.txt");

//     for (int j = 0; j < data_size; j++)
//     {
//         if (data[j] != recover[j])
//         {
//             out <<"num: " << j << " true is: " << data[j] << " predict is: " << recover[j] << std::endl;
//             std::cout << "something wrong! decompress failed" << std::endl;
//             count ++ ;
//             break;
//         }
//     }

//     out.close();

//     // printf("count = %lu\n", count);

//     return 0;
// }




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

    // 将数据分块处理，计算每块大小
    size_t chunk_size = 200000; 
    size_t num_chunks = (data_size + chunk_size - 1) / chunk_size; // 计算总块数

    // CUDA 事件对象，用于测量每次处理的时间
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

        // 计算当前块的大小和起始位置
        size_t start_idx = chunk * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, data_size);
        size_t current_chunk_size = end_idx - start_idx;

        // printf("current_chunk_size = %lu\n", current_chunk_size);

        std::vector<leco_type> data_chunk(data.begin() + start_idx, data.begin() + end_idx);

        // 在GPU上分配当前数据块的内存
        leco_type* data_gpu = nullptr;
        cudaError_t err = cudaMalloc(&data_gpu, sizeof(leco_type) * current_chunk_size);

        if (err != cudaSuccess) {
            printf("1 CUDA error: %s\n", cudaGetErrorString(err));
        }

        // 将数据传输到GPU
        err = cudaMemcpy(data_gpu, data_chunk.data(), sizeof(leco_type) * current_chunk_size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            printf("2 CUDA error: %s\n", cudaGetErrorString(err));
        }

        // 在GPU上分配其他需要的内存（如segments, index等）
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

        // 设置网格和块大小
        int block_size = current_chunk_size / blocks;
        int grid_size = (blocks + K - 1) / K;

        size_t *byte_count = nullptr;
        cudaMalloc((void**)&byte_count, sizeof(size_t) * blocks);

        // 记录起始时间
        cudaEventRecord(start, 0);



        // 调用编码核函数
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

        

        // std::cout << "第 " << chunk + 1 << " 块 CUDA 内核执行时间: " << milliseconds / 1000.0 << " 秒" << std::endl;

        total_time += milliseconds / 1000.0;

        // 解压缩部分，类似于压缩过程，逐块解压并进行验证
        leco_type* recover_gpu = nullptr;
        cudaMalloc(&recover_gpu, sizeof(leco_type) * current_chunk_size);

        cudaEventRecord(start, 0);

        kernel_decodeArray8_var<leco_type><<<grid_size, K>>>(res_total_gpu, recover_gpu, blocks, block_size, segment_index_total_gpu, segment_length_total_gpu, current_chunk_size);

        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);

        // std::cout << "第 " << chunk + 1 << " 块 解压缩时间: " << milliseconds / 1000.0 << " 秒" << std::endl;

        decompress_time += milliseconds / 1000.0;

        leco_type* recover = (leco_type*)malloc(sizeof(leco_type) * current_chunk_size);
        cudaMemcpy(recover, recover_gpu, sizeof(leco_type) * current_chunk_size, cudaMemcpyDeviceToHost);

        // 验证每块数据
        for (size_t j = 0; j < current_chunk_size; j++) {
            if (data_chunk[j] != recover[j]) {
                std::cout << "解压失败，块 " << chunk + 1 << " 第 " << j << " 个数据不匹配" << std::endl;
                break;
            }
        }

        // 清理当前块的GPU内存

        cudaFree(recover_gpu);
        cudaFree(segment_index_total_gpu);
        cudaFree(segment_length_total_gpu);
        cudaFree(res_total_gpu);
        cudaFree(signvec_gpu);
        cudaFree(delta_final_gpu);

        free(recover);
    }

    cudaEventSynchronize(stop);

    cudaError_t err = cudaDeviceSynchronize(); // 等待所有设备任务完成
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    int *cpu_segment_cnt_total = (int *)malloc(sizeof(int));

    cudaMemcpy(cpu_segment_cnt_total, segment_cnt_total, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(segment_cnt_total);

    std::cout << "Total segment = " << *cpu_segment_cnt_total << std::endl;

    end_time_out = clock();

    std::cout << "包括内存分配与销毁在内, 总耗时 = " << double(end_time_out - start_time_out) / CLOCKS_PER_SEC << " 秒" << std::endl;

    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "压缩后大小: " << total_byte_total_global << std::endl;

    std::cout << "压缩比: " << double(data_size * sizeof(leco_type)) / total_byte_total_global << std::endl;
    

    std::cout << "所有块处理完毕" << "用时: " << total_time <<std::endl;
    std::cout << "解压缩总时间: " << decompress_time << std::endl;

    return 0;
}