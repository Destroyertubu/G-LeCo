
#ifndef PIECEWISEFIX_INTEGER_TEMPLATE_H_
#define PIECEWISEFIX_INTEGER_TEMPLATE_H_

#include "common.h"
#include "codecs.h"
#include "bit_write.h"
#include "lr.h"
#include "bit_read.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#define INF 0x7f7fffff

namespace Codecset
{

    template <typename T>
    __device__ void read_all_bit_fix_d(const uint8_t* in, int start_byte, int start_index, int numbers, int l, double slope, double start_key, T* out, int tid)
    {

        // printf("%d, read_all_bit_fix_d\n", tid);
        int left = 0;
        uint128_t decode = 0;
        uint64_t start = start_byte;
        uint64_t end = 0;
        uint64_t total_bit = l * numbers;
        int writeind = 0;
        end = start + (int)(total_bit / (sizeof(uint64_t) * 8));
        T* res = out;
        if (total_bit % (sizeof(uint64_t) * 8) != 0)
        {
            end++;
        }

        int cnt = 0;


        while (start <= end && writeind < numbers)
        {
            while (left >= l && writeind < numbers)
            {
                // int128_t tmp = decode & (((T)1 << l) - 1);
                int64_t tmp = (decode & (((T)1 << l) - 1));
                bool sign = (tmp >> (l - 1)) & 1;
                T tmpval = (tmp & (((T)1 << (uint8_t)(l - 1)) - 1));
                decode = (decode >> l);
                int64_t decode_val = (long long)(start_key + (double)writeind * slope);



                // int128_t decode_val = (long long)(start_key + (double)writeind * slope);
                if (!sign)
                {
                    decode_val = decode_val - tmpval;
                }
                else
                {
                    decode_val = decode_val + tmpval;
                }

                // if (tid == 1)
                //     printf("decode_val = %lx, res addr = %p\n", decode_val, res);

                
                

                // if (decode_val == 77308810092)
                //     printf("Find TID = %d\n", tid);

                *res = (T)decode_val;
                res++;
                writeind++;
                if(writeind >= numbers){
                    return;
                }
                left -= l;
                if (left == 0)
                {
                    decode = 0;
                }
            }
            
            // if (tid == 0)
            // {
            //     printf("in address 1 = %lu\n", in);
            //     printf("in address 2 = %lu\n", reinterpret_cast<const uint64_t*>(in));
            // }

            uint64_t tmp_64;

            memcpy(&tmp_64, in + start * 8, sizeof(uint64_t));
        
            // uint64_t tmp_64 = (reinterpret_cast<const uint64_t*>(in))[start];    //! 有问题

            // uint64_t tmp_64 = ((uint64_t *)in)[start];

            decode += ((uint128_t)tmp_64 << left);
            start++;
            left += sizeof(uint64_t) * 8;
        }
    }

    template <typename T>
    __device__ uint8_t * write_delta_int_T_d(T *in, bool *signvec, uint8_t *out, uint8_t l, int numbers)
    {
        uint128_t code = 0;
        int occupy = 0;
        uint64_t endbit = (l * (uint64_t)numbers);
        uint64_t end = 0;
        int writeind = 0;
        
        int readind = 0;
        if (endbit % 8 == 0)
        {
            end = endbit / 8;
        }
        else
        {
            end = endbit / 8 + 1;
        }
        uint8_t *last = out + end;
        uint64_t left_val = 0;

        while (out <= last)
        {
            while (occupy < 8)
            {
                if (readind >= numbers)
                {
                    occupy = 8;
                    break;
                }

                
                T tmpnum = in[readind];
                bool sign = signvec[readind];
                T value1 =
                    (tmpnum & (((T)1 << (uint8_t)(l - 1)) - 1))     //* 外面传进来的 max_bit 是数值位数 + 1, 所以这里的掩码 (l - 1) 就是数值位数
                + (((T)sign) << (uint8_t)(l - 1));         //* 在解码函数中，sign为1的加，sign为0的减


                code += ((uint128_t)value1 << (uint8_t)occupy);
                occupy += l;
            
                readind++;
            } //end while
            while (occupy >= 8)
            {
                left_val = code >> (uint8_t)8;
                //std::cout<<code<<std::endl;
                code = code & ((1 << 8) - 1);
                uint8_t tmp_char = code;
                occupy -= 8;
                out[0] = tmp_char;
                code = left_val;
                //std::cout<< writeind<<std::endl;
                //std::cout<<occupy<<" "<<left_val<<" "<<unsigned(out[0])<<std::endl;
                out++;
            }
        }
        
        int pad = 8 - end % 8;
        for (int i = 0; i < pad; i++)
        {
            out[0] = 0;
            out++;
        }
        return out;
    }


    template <typename T>
    __device__ inline uint32_t bits_int_T(T v) {
        if (v < 0) {
            v = -v;
        }
        if (v == 0) return 0;
        // Assuming the code is compiled with NVCC, __builtin_clzll can be used.
        return 64 - __builtin_clzll(v);
    }

    
    template <typename T>
    __global__ void calc_delta(T *data, double theta0, double theta1, int local_block_length, T* delta, bool* signvec, T *max_error)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid >= local_block_length)
            return ;

        T tmp_val;
        int64_t pred = theta0 + theta1 * (double)tid;

        if (data[tid] > pred)
        {
            tmp_val = data[tid] - pred;
            signvec[tid] = true; // means positive
        }
        else
        {
            tmp_val = pred - data[tid];
            signvec[tid] = false; // means negative
        }

        delta[tid] = tmp_val;

        atomicMax((int *)max_error, (int)tmp_val);
    }


    template <typename T>
    __device__ void caltheta(const T *y, int m, double *theta0, double *theta1){

        double sumx = 0;
        double sumy = 0;
        double sumxy = 0;
        double sumxx = 0;

        // printf("first element of segment data = %lu\n", y[0]);

        // printf("the m = %d, tid = %d\n", m, tid);
        for(int i=0;i<m;i++){
            sumx = sumx + (double)i;
            sumy = sumy + (double)y[i];
            sumxx = sumxx+(double)i*i;
            sumxy = sumxy+(double)i*y[i];
        }
        
        double ccc= sumxy * m - sumx * sumy;
        double xxx = sumxx * m - sumx * sumx;
    
        *theta1 = ccc/xxx;
        *theta0 = (sumy - (*theta1) * sumx)/(double)m;

        // printf("theta0 = %lf, theta1 = %lf\n", theta0, theta1);
        
    }


    template <typename T>
    __global__ void kernel_encodeArray8_int(T *data, int block_size, int blocks, int N, uint8_t **pointers, uint32_t *size_array, T *delta_gpu, bool *signvec_gpu)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid >= blocks)
            return ;

        int local_block_length = block_size;
        if (tid == blocks - 1)
        {
            local_block_length =  N - (blocks - 1) * block_size;
            // printf("last tid = %d, local_block_length = %d\n", tid, local_block_length);
        }



        uint8_t *out = pointers[tid];

        uint8_t *ptr = pointers[tid];

        T *y = data + (tid * block_size);

        double theta0, theta1;

        caltheta(y, local_block_length, &theta0, &theta1); 

        T max_error = 0;

        // calc_delta<<<tmp_blocks, 32>>>(data + (tid * block_size), theta0, theta1, local_block_length, delta, signvec, &max_error);




        int64_t max_error_delta = INT64_MIN;
        int64_t min_error_delta = INT64_MAX;
        for (auto i = 0; i < local_block_length; i++)
        {
            int64_t tmp_val = (int64_t)y[i] - (int64_t)(theta0 + theta1 * (double)i);
            if (tmp_val > max_error_delta)
                max_error_delta = tmp_val;
            if (tmp_val < min_error_delta)
                min_error_delta = tmp_val;
        }
        theta0 += (max_error_delta + min_error_delta) / 2.;

        T *delta = delta_gpu + (tid * block_size);
        bool *signvec = signvec_gpu + (tid * block_size);     

        

        
        for (auto i = 0; i < local_block_length; i++)       // TODO: Change to kernel function
        {
            T tmp_val;
            int64_t pred = theta0 + theta1 * (double)i;

            if (y[i] > pred)
            {
                tmp_val = y[i] - pred;
                signvec[i] = true; // means negative
            }
            else
            {
                tmp_val = pred - y[i];
                signvec[i] = false;
            }

            delta[i] = tmp_val;

            // if (tid == 0)
            //     printf("delta = %lu, signvec = %lu, data = %lu\n", delta[i], signvec[i], y[i]);

            if (tmp_val > max_error)
            {
                max_error = tmp_val;
            }
        }



        // if (tid == blocks - 1)
        //     printf("the last thread get2!\n");

        uint8_t max_bit = 0;
        if (max_error)
        {
            max_bit = bits_int_T(max_error) + 1;
        }
        // std::cout<< "max_bit: " << (int)max_bit << std::endl;
        if (max_bit > sizeof(T) * 8)
        {
            max_bit = sizeof(T) * 8;
        }

        memcpy(out, &max_bit, sizeof(max_bit));
        // out[0] = max_bit;
        out += sizeof(max_bit);


        if (max_bit == sizeof(T) * 8)         //* Copy the data to out array directly if the bits needed to store maxerror exceed  
        {                                     //* the  sizeof(T) * 8
            for (auto i = 0; i < local_block_length; i++)
            {
                memcpy(out, &y[i], sizeof(T));
                // if (tid == blocks - 1)
                //     printf("Last block out[%d] = %llx, y[%d] = %llx\n", i, (uint64_t*)out, i, y[i]);
                out += sizeof(T);
            }

            // if (tid == blocks - 1)
            // {
            //     int start = 0;
            //     for (int i = 0;i < local_block_length;i ++ )
            //     {
            //         uint64_t tmp_64;

            //         memcpy(&tmp_64, ptr + start * 8, sizeof(uint64_t));

            //         printf("last tmp_64 = %lu\n", tmp_64);
            //         start ++ ;

            //     }
            // }

            uint32_t tmp_size = out - pointers[tid];
            
            size_array[tid] = tmp_size;

            return ;
        }

        memcpy(out, &theta0, sizeof(double));
        // out[0] = theta0;
        out += sizeof(double);
        memcpy(out, &theta1, sizeof(double));
        // out[0] = theta1;
        out += sizeof(double);


        if (max_bit)
        {
            out = write_delta_int_T_d(delta, signvec, out, max_bit, local_block_length);
        }

        uint32_t tmp_size = out - pointers[tid];
        
        size_array[tid] = tmp_size;
    }

    template <typename T>
    __global__ void kernel_decodeArray8(uint8_t **gpu_in, T *out, int blocks, int block_size, int N)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid >= blocks)
            return ;

        

        int length = block_size;
        if (tid == blocks - 1)
        {
            length =  N - (blocks - 1) * block_size;
        }

        uint8_t *in = gpu_in[tid];



        // if (tid == 0)
        //     printf("out address = %lx\n", out);

        T *res = out + tid * length;
        double theta0;
        double theta1;
        const uint8_t *tmpin = in;

        uint8_t maxerror = tmpin[0];
        tmpin++;

        if (tid == blocks - 1)
        {
            // // int start = 0;
            // for (int i = 0;i < length;i ++ )
            // {
            //     uint64_t tmp_64;

            //     memcpy(&tmp_64, tmpin + i * 8, sizeof(uint64_t));

            //     printf("in[%u] = %llx\n", i, tmp_64);
            // }


            // printf("last block res address = %p, out address = %p\n", res, out);
        }

        if (maxerror == 127)
        {
            // if (tid == blocks - 1)
            //     printf("127 last res addr = %p\n", res);  
            T tmp_val;
            memcpy(&tmp_val, tmpin, sizeof(tmp_val));
            res[0] = tmp_val;
            res++;
            return ;
        }
        if (maxerror == 126)
        {
            // if (tid == blocks - 1)
            //     printf("126 last res addr = %p\n", res);  
            T tmp_val;
            memcpy(&tmp_val, tmpin, sizeof(tmp_val));
            res[0] = tmp_val;
            res++;
            memcpy(&tmp_val, tmpin + sizeof(T), sizeof(tmp_val));
            res[0] = tmp_val;
            res++;
            return ;
        }
        if (maxerror >= sizeof(T) * 8 - 1)
        {
            // if (tid == blocks - 1)
            //     {
            //         printf("125 last res addr = %p\n", res);

            //         // for (int i = 0;i < length;i ++ )
            //         // {
            //         //     printf("tmpin[%d] = %lx\n", i, *tmpin);
            //         //     tmpin++;
            //         // }
            //     }  
            // out = reinterpret_cast<T*>(tmpin);
            memcpy(res, tmpin, sizeof(T) * length);

            // for (int i = 0;i < length;i ++ )
            // {
            //     printf("tid = %d, direct copy out[%d] = %llx, address = %p\n", tid, i, out[i], &out[i]);
            // }


            // read_all_default(tmpin, 0, 0, length, maxerror, theta1, theta0, res);
            return ;
        }

        memcpy(&theta0, tmpin, sizeof(theta0));
        tmpin += sizeof(theta0);
        memcpy(&theta1, tmpin, sizeof(theta1));
        tmpin += sizeof(theta1);

        
        // if (tid == 0)
        //     printf("res addr = %p\n", res);
        



        if (maxerror)
        {

            // read_all_bit_fix_add<T>(tmpin, 0, 0, length, maxerror, theta1, theta0, res);
            read_all_bit_fix_d<T>(tmpin, 0, 0, length, maxerror, theta1, theta0, res, tid);
            
            // if (tid == 0)
            // {
            //     // int cnt = 0;
            //     // printf("res addr = %p\n", res[2]);
            //     T* tmp = res;
            //     for (int i = 0;i < length;i ++ )
            //     {
            //         printf("tid = %d, res %d = %lx, addr = %p\n", tid, i, *tmp, tmp);
            //         tmp++;
            //     }
            // }

            // T* tmp = res;
            // for (int i = 0;i < length;i ++ )
            // {
            //     // printf("res %d = %lx, addr = %p\n", i, *tmp, tmp);
            //     if (*tmp == 0x11fff6d36c)
            //         printf("TID = %d\n", tid);
            //     tmp++;
            // }

        }
        else
        {
            for (int j = 0; j < length; j++)
            {
                res[j] = (long long)(theta0 + theta1 * (double)j);
            }

            // T* tmp = res;
            // for (int i = 0;i < length;i ++ )
            // {
            //     // printf("res %d = %lx, addr = %p\n", i, *tmp, tmp);
            //     if (*tmp == 0x11fff6d36c)
            //         printf("TID = %d\n", tid);
            //     tmp++;
            // }


            // double pred = theta0;
            // for (int i = 0;i < length;i++) {
            //     res[i] = (long long)pred;
            //     pred += theta1;
            // }
        }

    }


    template <typename T>
    __device__ T read_bit_fix_int_wo_round(uint8_t* in, uint8_t l, int to_find, double slope, double start_key, int tid)
    {
        uint64_t find_bit = to_find * (int)l;
        uint64_t start_byte = find_bit / 8;
        uint8_t start_bit = find_bit % 8;
        uint64_t occupy = start_bit;
        uint64_t total = 0;

        // uint128_t decode = (reinterpret_cast<const uint128_t*>(in + start_byte))[0];

        uint128_t decode = 0;

        memcpy(&decode, in + start_byte, sizeof(uint128_t));

        // for (int i = 0; i < sizeof(uint128_t); ++i) {
        //     decode <<= 8;
        //     decode |= in[start_byte * sizeof(uint128_t) + i];
        // }

        // memcpy(&decode, in+start_byte, sizeof(uint64_t));
        decode >>= start_bit;
        uint64_t decode_64 = decode & (((T)1 << l) - 1);
        // decode &= (((T)1 << l) - 1);

        bool sign = (decode_64 >> (l - 1)) & 1;
        T value = (decode_64 & (((T)1 << (uint8_t)(l - 1)) - 1));
        int64_t out = (start_key + (double)to_find * slope);

        // if (tid == 0)
        //     printf("tid 0: out = %lu, value = %lu, start_key = %ld, to_find = %d, slope = %ld\n", out, value, start_key, to_find, slope);
        if (!sign)
        {
            out = out - value;
        }
        else
        {
            out = out + value;
        }

        return (T)out;

    }

    template <typename T>
    __global__ void kernel_randomdecodeArray(uint8_t **gpu_in, uint32_t *gpu_data, T *res, int N, int block_size)
    {
        extern __shared__ T shared_res[];
        
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid >= N)
            return ;

        int local_tid = threadIdx.x;

        uint32_t index = gpu_data[tid];

        int to_find = index % block_size;
        uint8_t *in = gpu_in[(int)index / block_size];
        
        uint8_t *tmpin = in;
        uint8_t maxbits;
        memcpy(&maxbits, tmpin, sizeof(uint8_t));
        tmpin += sizeof(uint8_t);

        if(maxbits==sizeof(T)*8){
            // T tmp_val = reinterpret_cast<T *>(tmpin)[to_find];
            T tmp_val;
            memcpy(&tmp_val, tmpin + to_find * sizeof(T), sizeof(T));
            shared_res[local_tid] = tmp_val;

            // res[tid] = tmp_val;
            return ;
        }

        double theta0;
        memcpy(&theta0, tmpin, sizeof(double));
        tmpin += sizeof(double);

        double theta1;
        memcpy(&theta1, tmpin, sizeof(double));
        tmpin += sizeof(double);
        
        
        if(maxbits){
            shared_res[local_tid] = read_bit_fix_int_wo_round<T>(tmpin, maxbits, to_find, theta1, theta0, tid);

            // res[tid] = read_bit_fix_int_wo_round<T>(tmpin, maxbits, to_find, theta1, theta0, tid);

            // if (tid == 0)
            //     printf("tid 0: res = %llx, pos = %llx\n", res[tid], index);
        }
        else{
            shared_res[local_tid] = ((double)theta0 + (float)to_find * theta1);

            // res[tid] = ((double)theta0 + (float)to_find * theta1);

            // if (tid == 0)
            //     printf("tid 0: res = %llx, pos = %llx\n", res[tid], index);
        }
        

        __syncthreads();

        if (tid % 256 == 0)
        {
            // printf("tid %d: res = %d\n", tid, shared_res[local_tid]);
            for (int i = 0;i < 256;i ++ )
            {
                if (tid + i >= N || res[tid + i])
                    continue;
                res[tid + i] =  shared_res[i];
            }
        }  

        // res[tid] = shared_res[tid % 256];

        return ;
    }

    __global__ void preprocess_bucket_cnt(uint32_t *ra_pos, int *bucket_cnt_gpu, int N, int block_size)
    {
        int bid = blockIdx.x;
        int tid = bid * blockDim.x + threadIdx.x;

        if (tid >= N)
            return ;
        
        atomicAdd(&bucket_cnt_gpu[ra_pos[tid] / block_size], 1) ;
    }


    } // namespace FastPFor

#endif /* SIMDFASTPFOR_H_ */
