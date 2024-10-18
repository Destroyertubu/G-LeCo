#include "common.h"
#include "codecs.h"
#include "time.h"
#include "bit_read.cuh"
#include "bit_write.cuh"
#include "caltime.cuh"
#include "lr.cuh"
#define INF 0x7f7fffff
#include "stx-btree/btree.h"
#include "stx-btree/btree_map.h"
#include "ALEX/alex.h"
#include "art/art32.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>

typedef uint32_t leco_type;

struct Pair{
    size_t length;
    uint8_t *out;
};

template <typename S>
struct alignas(8) Segment {
    // [start, end], this maintains delta information
    int start_index;
    int end_index;
    S max_delta;
    S min_delta;
    S next_delta; // this is not contained in the segment
    int double_delta_next;
    Segment* prev;
    Segment* next;

    __host__ __device__ 
        Segment()
    : start_index(0), end_index(0), max_delta(0), min_delta(0),
        next_delta(0), double_delta_next(0), next(nullptr), prev(nullptr) {}

    __host__ __device__ Segment(int start, int end, S max, S min, S next, int bit_next) {
        start_index = start;
        end_index = end;
        max_delta = max;
        min_delta = min;
        next_delta = next;
        double_delta_next = bit_next;
    }
};



      
template <typename T>
struct lr_int_T_cuda{
    double theta0;
    double theta1;

    
__device__ void caltheta_cuda(const T *y, int m){

    double sumx = 0;
    double sumy = 0;
    double sumxy = 0;
    double sumxx = 0;
    for(int i=0;i < m;i++){
        sumx = sumx + (double)i;
        sumy = sumy + (double)y[i];
        sumxx = sumxx+(double)i*i;
        sumxy = sumxy+(double)i*y[i];
    }
    
    double ccc= sumxy * m - sumx * sumy;
    double xxx = sumxx * m - sumx * sumx;

    theta1 = ccc/xxx;
    theta0 = (sumy - theta1 * sumx)/(double)m;
    
}

};

__device__ void print_int128(__int128 num) {
uint64_t high = (uint64_t)(num >> 64);
uint64_t low = (uint64_t)num;

printf("0x%016lX%016lX\n", high, low);
}

template <typename T>
__device__ void read_all_bit_fix_gpu(const uint8_t* in, int start_byte, int start_index, int numbers, int l, double slope, double start_key, T* out, int index)
    {
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
    


    template <typename T,
            typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    __device__ inline uint32_t bits_int_T_cuda(T v) {
    if(v<0){
        v=-v;
    }
    if (v == 0) return 0;
    #if defined(__clang__) || defined(__GNUC__)
    // std::cout<<__builtin_clzll(v)<<" "<<64 - __builtin_clzll(v)<<std::endl;
    return 64 - __builtin_clzll(v);
    #else
    assert(false);
    #endif
    }
    template <typename T, typename std::enable_if<std::is_same<__uint128_t, T>::value ||
                                                    std::is_same<leco_uint256, T>::value,
                                                bool>::type = true>
    __device__ inline uint32_t bits_int_T_cuda(T v) {
    if(v<0){
        v = -v;
    }
    uint32_t r(0);
    constexpr int length = sizeof(T) * 8;
    if constexpr (length > 255) {
        if (v >= ((T)1 << (uint8_t)255)) {
        v >>= 255;
        // v = v/2;
        r += 256;
        }
    }
    if constexpr (length > 127) {
        if (v >= ((T)1 << (uint8_t)127)) {
        v >>= 128;
        r += 128;
        }
    }
    if constexpr (length > 63) {
        if (v >= ((T)1 << (uint8_t)63)) {
        v >>= 64;
        r += 64;
        }
    }
    if (length > 31 && v >= ((T)1 << (uint8_t)31)) {
        v >>= 32;
        r += 32;
    }
    if (length > 15 && v >= ((T)1 << (uint8_t)15)) {
        v >>= 16;
        r += 16;
    }
    if (length > 7 && v >= ((T)1 << (uint8_t)7)) {
        v >>= 8;
        r += 8;
    }
    if (length > 3 && v >= ((T)1 << (uint8_t)3)) {
        v >>= 4;
        r += 4;
    }
    if (length > 1 && v >= ((T)1 << (uint8_t)1)) {
        v >>= 2;
        r += 2;
    }
    if (v >= (T)1) {
        r += 1;
    }

    return r;
    }


    template <typename T>
    __device__ uint8_t * write_delta_int_T(T *in, bool* signvec, uint8_t *out, uint32_t l, int numbers, size_t index)
    {
        uint128_t code = 0;
        int occupy = 0;
        uint64_t endbit = (l * (uint64_t)numbers);
        uint64_t end = 0;
        int writeind = 0;

        // if (index == 10379)
        //     for(int i = 0;i < numbers;i ++ )
        //     {
        //         printf("delta[%d] = %u\n", i, in[i]);
        //     }

        
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

        int count_loc = 0;

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
                    (tmpnum & (((T)1 << (uint8_t)(l - 1)) - 1))     
                + (((T)sign) << (uint8_t)(l - 1));      


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

    double split_time = 0;
    double merge_time = 0;



    uint64_t total_byte_total = 0;
    
    int overhead = 0;
    leco_type* array;
    
    int block_size;
    int segment_index_total_idx = 0;

    alex::Alex<int, int> alex_tree;
    ART32 art;


    template <typename T>
    __device__ uint64_t newsegment_size(uint32_t origin_index, uint32_t end_index, T *array, int block_size, int tid) {

        if (origin_index == end_index) {
            return 9;
        }
        if (end_index == origin_index + 1) {
            return 13;
        }

        // printf("enter newsegment_size! tid = %d\n", tid);

        uint64_t overhead = sizeof(float) * 2 + 5;
        int length = end_index - origin_index + 1;

        lr_int_T_cuda<T> mylr;
        mylr.caltheta_cuda(array + origin_index, length);

        // printf("tid = %d, origin_index = %d, block_size = %d\n", tid, origin_index, block_size);

        float final_slope = mylr.theta1;
        float theta0 = mylr.theta0;

        int64_t max_error_delta = INT64_MIN;
        int64_t min_error_delta = INT64_MAX;
        for (int j = origin_index;j <= end_index; j++) {
            int64_t tmp = array[j] - (long long)(theta0 + final_slope * (double)(j - origin_index));
            if (tmp > max_error_delta) {
                max_error_delta = tmp;
            }
            if (tmp < min_error_delta) {
                min_error_delta = tmp;
            }
        }
        theta0 += (max_error_delta + min_error_delta) / 2.0;

        T final_max_error = 0;
        // std::vector<bool> signvec;
        // std::vector<T> delta_final;

        for (int j = origin_index;j <= end_index;j++) {
            T tmp_val;
            int128_t pred = theta0 + final_slope * (double)(j - origin_index);
            if (array[j] > pred)
            {
                tmp_val = array[j] - pred;
            }
            else
            {
                tmp_val = pred - array[j];
            }

            if (tmp_val > final_max_error)
            {
                final_max_error = tmp_val;
            }
        }

        uint32_t delta_final_max_bit = 0;
        if (final_max_error) {
            delta_final_max_bit = bits_int_T_cuda<T>(final_max_error) + 1;
        }


        if (delta_final_max_bit >= sizeof(T) * 8) {
            delta_final_max_bit = sizeof(T) * 8;
            overhead = 5 + sizeof(T) * length;
            return overhead;
        }

        overhead += ceil((delta_final_max_bit * length) / 8.0);

        return overhead;

    }

    
    template <typename T>
    __device__ uint8_t* newsegment_2(uint32_t origin_index, uint32_t end_index, T* array, uint32_t *segment_index_total, size_t &segment_index_total_cnt, uint32_t *segment_length_total, size_t &segment_length_total_cnt, int tid, uint8_t *out, size_t &total_byte_total, size_t &segment_size) {

        uint8_t *ptr = out;
        memcpy(out, &origin_index, sizeof(origin_index));

        // if (tid == 4 && origin_index == 0)
        // {
        //     int start_ind = 0;
        //     memcpy(&start_ind, out, sizeof(int));
        //     printf("start_ind = %d\n", start_ind);
        // }

        out += sizeof(origin_index);
        out[0] = (uint8_t)254; // this means that this segment only has two points
        out++;
        memcpy(out, &array[origin_index], sizeof(T));

        // if (tid == 4 && origin_index == 0)
        // {

        //     printf("origin_index = %d, array[origin_index] = %u\n, array[origin_index + 1] = %u, \n\n", origin_index, array[origin_index], array[origin_index + 1]);
        // }
        out += sizeof(T);
        memcpy(out, &(array[origin_index + 1]), sizeof(T));
        out += sizeof(T);

        segment_size = out - ptr;



        segment_index_total[segment_index_total_cnt ++ ] = origin_index;
        segment_length_total[segment_length_total_cnt ++ ] = segment_size;

        total_byte_total += segment_size;

        uint8_t* check_address = (uint8_t*)0x72ee0b370;


        return out;
    }

    template <typename T>
    __device__ uint8_t* newsegment_1(uint32_t origin_index, uint32_t end_index, T* array, uint32_t *segment_index_total, size_t &segment_index_total_cnt, uint32_t *segment_length_total, size_t &segment_length_total_cnt, int tid, uint8_t *out, size_t &total_byte_total, size_t &segment_size) {

        // uint8_t* descriptor = (uint8_t*)malloc(10 * sizeof(T));
        // uint8_t* out = descriptor;

        uint8_t *ptr = out;

        memcpy(out, &origin_index, sizeof(origin_index));
        out += sizeof(origin_index);
        out[0] = (uint8_t)255; // this means that this segment only has one point
        out++;
        memcpy(out, &array[origin_index], sizeof(T));
        out += sizeof(T);

        segment_size = out - ptr;
        segment_index_total[segment_index_total_cnt ++ ] = origin_index;
        segment_length_total[segment_length_total_cnt ++ ] = segment_size;

        total_byte_total += segment_size;

        uint8_t* check_address = (uint8_t*)0x72ee0b370;

        return out;
    }

    template <typename T> 
    __device__ uint64_t merge_both_direction(int loop_cnt, int tid, int block_size, uint32_t *segment_index, size_t &local_segment_index_cnt, uint32_t *segment_length, size_t &local_segment_length_cnt,  uint32_t *new_segment_index, uint32_t *new_segment_length, T *array, int local_block_size) {
        
        int start_index = segment_index[0];

        int segment_num = 0;
        int total_segments = local_segment_index_cnt;

        uint64_t totalbyte_after_merge = 0;

        segment_index[local_segment_index_cnt ++ ] = local_block_size;

        uint32_t* local_new_segment_index = new_segment_index + tid * block_size * 2;

        uint32_t* local_new_segment_length = new_segment_length + tid * block_size * 2;

        int local_new_segment_index_cnt = 0;
        int local_new_segment_length_cnt = 0;

        local_new_segment_index[local_new_segment_index_cnt ++ ] = segment_index[segment_num];

        local_new_segment_length[local_new_segment_length_cnt ++ ] = segment_length[segment_num];

        totalbyte_after_merge += segment_length[segment_num];

        segment_num ++ ;
        

        while (segment_num < total_segments){


            if (segment_num == total_segments - 1)
            {
                int last_merged_segment = local_new_segment_length_cnt - 1;

                uint32_t init_cost_front = segment_length[segment_num] + local_new_segment_length[last_merged_segment];

                uint32_t merge_cost_front = newsegment_size(local_new_segment_index[last_merged_segment], segment_index[segment_num + 1] - 1, array, block_size, tid);

                int saved_cost_front = init_cost_front - merge_cost_front;

                if (saved_cost_front > 0) {
                    totalbyte_after_merge -= local_new_segment_length[local_new_segment_length_cnt - 1];

                    local_new_segment_length[local_new_segment_length_cnt - 1] = merge_cost_front;

                    totalbyte_after_merge += merge_cost_front;

                    segment_num ++ ;
                }
                else {
                    local_new_segment_index[local_new_segment_index_cnt ++ ] = segment_index[segment_num];

                    local_new_segment_length[local_new_segment_length_cnt ++ ] = segment_length[segment_num];

                    totalbyte_after_merge += segment_length[segment_num];

                    segment_num ++ ;
                }
                break;
            }
            
            

            int last_merged_segment = local_new_segment_length_cnt - 1;

            uint32_t init_cost_front = segment_length[segment_num] +  local_new_segment_length[last_merged_segment];

            uint32_t merge_cost_front = newsegment_size(local_new_segment_index[last_merged_segment], segment_index[segment_num + 1] - 1, array, block_size, tid);

            int saved_cost_front = init_cost_front - merge_cost_front;

            uint32_t init_cost_back = segment_length[segment_num] + segment_length[segment_num + 1];
            uint32_t merge_cost_back = newsegment_size(segment_index[segment_num], segment_index[segment_num + 2] - 1, array, block_size, tid);

            int saved_cost_back = init_cost_back - merge_cost_back;

            int saved_cost = max(saved_cost_front, saved_cost_back);
            // if (tid == 0)
            //     printf("last_merge_segment = %d init_cost_front = %d merge_cost_front = %d saved_cost_front = %d init_cost_back = %d merge_cost_back = %d saved_cost_back = %d saved_cost = %d\n", last_merged_segment, init_cost_front, merge_cost_front, saved_cost_front, init_cost_back, merge_cost_back, saved_cost_back, saved_cost);

            if (saved_cost <= 0) {
                local_new_segment_index[local_new_segment_index_cnt ++ ] = segment_index[segment_num];

                local_new_segment_length[local_new_segment_length_cnt ++ ] = segment_length[segment_num];

                totalbyte_after_merge += segment_length[segment_num];

                start_index = segment_index[segment_num + 1];

                segment_num ++ ;

                // if (tid == 0)
                //     printf("1 if\n");

                continue;
            }

            if (saved_cost_back > saved_cost_front) {

                local_new_segment_index[local_new_segment_index_cnt ++ ] = segment_index[segment_num];

                local_new_segment_length[local_new_segment_length_cnt ++ ] = merge_cost_back;

                // if (tid == 0)
                //     printf("1 totalbyte_after_merge = %d\n", totalbyte_after_merge);

                totalbyte_after_merge += merge_cost_back;

                // if (tid == 0)
                //     printf("2 totalbyte_after_merge = %d\n", totalbyte_after_merge);

                start_index = segment_index[segment_num + 2];

                segment_num += 2;

                // if (tid == 0)
                //     printf("2 if\n");


            }
            else {
                // if (tid == 0)
                //     printf("1 totalbyte_after_merge = %d\n", totalbyte_after_merge);

                totalbyte_after_merge -= local_new_segment_length[local_new_segment_length_cnt - 1];

                // if (tid == 0)
                //     printf("2 totalbyte_after_merge = %d\n", totalbyte_after_merge);

                local_new_segment_length[local_new_segment_length_cnt - 1] = merge_cost_front;

                totalbyte_after_merge += merge_cost_front;

                start_index = segment_index[segment_num + 1];

                segment_num += 1 ;

                // if (tid == 0)
                //     printf("3 else\n");
            }
        }

        // printf("tid = %d, loop_cnt = %d, segment_num = %d\n", tid, loop_cnt, segment_num);

        uint64_t total_byte = totalbyte_after_merge;

        

        local_segment_index_cnt = local_new_segment_index_cnt;

        memcpy(segment_index, local_new_segment_index, local_new_segment_index_cnt * sizeof(uint32_t));

        local_segment_length_cnt = local_new_segment_length_cnt;

        memcpy(segment_length, local_new_segment_length, local_new_segment_length_cnt * sizeof(uint32_t));


        // if (tid == 0)
        // printf("tid = %d, loop_cnt = %d, total_byte = %d\n", tid, loop_cnt,  total_byte); 



        return total_byte;
    }



    template <typename T>
    __device__ Pair newsegment(uint32_t origin_index, uint32_t end_index, T* array, uint32_t *segment_index_total, size_t &segment_index_total_cnt, uint32_t *segment_length_total, size_t &segment_length_total_cnt, int tid, uint8_t *out, size_t &total_byte_total, size_t block_size, size_t length, bool *signvec, T *delta_final) {


        // printf("tid = %d, out = %p\n", tid, out);

        size_t segment_size = 0;

        if (origin_index == end_index) {
            uint8_t *tmp_1 = newsegment_1<leco_type>(origin_index, origin_index, array, segment_index_total, segment_index_total_cnt, segment_length_total, segment_length_total_cnt, tid, out, total_byte_total, segment_size);
            return Pair{1, tmp_1};
        }
        if (end_index == origin_index + 1) {
            uint8_t *tmp_2 = newsegment_2<leco_type>(origin_index, end_index, array, segment_index_total, segment_index_total_cnt,  segment_length_total, segment_length_total_cnt, tid, out, total_byte_total, segment_size);
            return Pair{2, tmp_2};
        }

        uint8_t *ptr = out;

        size_t index = tid * block_size + origin_index;


        length = end_index - origin_index + 1;

        lr_int_T_cuda<T> mylr;
        mylr.caltheta_cuda(array + origin_index, length);
        float final_slope = mylr.theta1;
        float theta0 = mylr.theta0;

        int64_t max_error_delta = INT64_MIN;
        int64_t min_error_delta = INT64_MAX;
        for (int j = origin_index;j <= end_index;j++) {
            int64_t tmp = array[j] - (long long)(theta0 + final_slope * (double)(j - origin_index));
            if (tmp > max_error_delta) {
                max_error_delta = tmp;
            }
            if (tmp < min_error_delta) {
                min_error_delta = tmp;
            }
        }
        theta0 += (max_error_delta + min_error_delta) / 2.0;

        T final_max_error = 0;

        int count = 0;

        for (int j = origin_index;j <= end_index;j++) {            
            T tmp_val;
            int128_t pred = theta0 + final_slope * (double)(j - origin_index);
            if (array[j] > pred)
            {
                tmp_val = array[j] - pred;
                signvec[j] = (true); // means positive
            }
            else
            {
                tmp_val = pred - array[j];
                signvec[j] = (false); // means negative
            }

            

            delta_final[j] = (tmp_val);

            if (tmp_val > final_max_error)
            {
                final_max_error = tmp_val;
            }
        }

        uint32_t delta_final_max_bit = 0;
        if (final_max_error) {
            delta_final_max_bit = bits_int_T_cuda<T>(final_max_error) + 1;
        }


        if (delta_final_max_bit >= sizeof(T) * 8) {
            delta_final_max_bit = sizeof(T) * 8;
        }

        memcpy(out, &origin_index, sizeof(origin_index));
        out += sizeof(origin_index);
        out[0] = (uint8_t)delta_final_max_bit;
        if(abs(final_slope) >= 0.00000001 && delta_final_max_bit != sizeof(T) * 8){ 
            out[0] += (1<<7); 
        }
        out++;

        if (delta_final_max_bit == sizeof(T) * 8) {
            for (auto i = origin_index; i <= end_index; i++)
            {
                memcpy(out, &array[i], sizeof(T));
                out += sizeof(T);
            }
            uint64_t segment_size = out - ptr;

            segment_index_total[segment_index_total_cnt ++ ] = origin_index;
            segment_length_total[segment_length_total_cnt ++ ] = segment_size;

            total_byte_total += segment_size;
            return Pair{length, out};
        }

        memcpy(out, &theta0, sizeof(theta0));
        out += sizeof(theta0);
        if(abs(final_slope) >= 0.00000001){ 
            memcpy(out, &final_slope, sizeof(final_slope));
            out += sizeof(final_slope);
        }

        if (delta_final_max_bit) {

            out = write_delta_int_T<T>(delta_final + origin_index, signvec + origin_index, out, delta_final_max_bit, (end_index - origin_index + 1), index);
        }

        segment_size = out - ptr;

        segment_index_total[segment_index_total_cnt ++ ] = origin_index;
        segment_length_total[segment_length_total_cnt ++ ] = segment_size;

        total_byte_total += segment_size;

        uint8_t* check_address = (uint8_t*)0x72ee0b370;


        return Pair{length, out};

    }

    template<typename T>
    __device__ uint32_t cal_bits(int64_t min, int64_t max) {
        int64_t range = ceil(abs(max - min) / 2.);
        uint32_t bits = 0;
        if (range) {
            bits = bits_int_T_cuda<T>(range) + 1;
        }
        return bits;
    }


    
    template <typename T>
    __global__ void kernel_var_encodeArray8_int(T* in, const size_t block_size, int blocks, size_t data_size, Segment<int64_t>* gpu_segs, int overhead, uint32_t *segment_index_gpu, uint32_t *segment_length_gpu, uint32_t *new_segment_index_gpu, uint32_t *new_segment_length_gpu, uint32_t *segment_index_total, uint32_t *segment_length_total, uint8_t *res_total_gpu, bool *signvec_gpu, T *delta_final_gpu, int *segment_cnt_total,  size_t *byte_count) {

        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid >= blocks)
            return;

        T* array = in + tid * block_size;
        auto local_gpu_segs = gpu_segs + tid * block_size * 2;

        size_t local_block_size;;

        if (tid == blocks - 1)
        {
            local_block_size = data_size - tid * block_size;
        }
        else
        {
            local_block_size = block_size;
        }       

        int seg_cnt = 0;
        local_gpu_segs[seg_cnt].start_index = 0;  // head
        local_gpu_segs[seg_cnt].end_index = 0;
        local_gpu_segs[seg_cnt].max_delta = 0;
        local_gpu_segs[seg_cnt].min_delta = 0;
        local_gpu_segs[seg_cnt].next_delta = 0;
        local_gpu_segs[seg_cnt].double_delta_next = 10000;

        auto& head = local_gpu_segs[seg_cnt];

        seg_cnt ++ ;

        local_gpu_segs[seg_cnt].start_index = 0;  // head
        local_gpu_segs[seg_cnt].end_index = 0;
        local_gpu_segs[seg_cnt].max_delta = 0;
        local_gpu_segs[seg_cnt].min_delta = 0;
        local_gpu_segs[seg_cnt].next_delta = 0;
        local_gpu_segs[seg_cnt].double_delta_next = 10000;

        auto& tail = local_gpu_segs[seg_cnt];

        seg_cnt ++ ;

        Segment<int64_t> *current = &head;

        int min_second_bit = 10000;
        int max_second_bit = -1;

        int64_t delta_prev = int64_t(array[1]) - int64_t(array[0]);

        for (int i = 1; i < local_block_size - 1;i ++ )
        {
            int64_t delta = int64_t(array[i + 1]) - int64_t(array[i]);
            int second_delta_bit = cal_bits<T>(delta_prev, delta);
            if (second_delta_bit < min_second_bit) {
                min_second_bit = second_delta_bit;
            }
            if (second_delta_bit > max_second_bit) {
                max_second_bit = second_delta_bit;
            }        

            // if (blockIdx.x == 31 && threadIdx.x == 8)
            // {
            //     printf("seg_cnt = %d, i = %d, delta_prev = %d, delta = %d, second_delta_bit = %d\n", seg_cnt, i, delta_prev, delta, second_delta_bit);
            // }    

            local_gpu_segs[seg_cnt].start_index = i - 1;  // head
            local_gpu_segs[seg_cnt].end_index = i - 1;
            local_gpu_segs[seg_cnt].max_delta = 0;
            local_gpu_segs[seg_cnt].min_delta = 0;
            local_gpu_segs[seg_cnt].next_delta = delta_prev;
            local_gpu_segs[seg_cnt].double_delta_next = second_delta_bit;

            current->next = &local_gpu_segs[seg_cnt];
            local_gpu_segs[seg_cnt].prev = current;
            current = &local_gpu_segs[seg_cnt];
            delta_prev = delta;

            seg_cnt ++ ;
        }

        local_gpu_segs[seg_cnt].start_index = local_block_size - 2;  // head
        local_gpu_segs[seg_cnt].end_index = local_block_size - 2;
        local_gpu_segs[seg_cnt].max_delta = 0;
        local_gpu_segs[seg_cnt].min_delta = 0;
        local_gpu_segs[seg_cnt].next_delta = delta_prev;
        local_gpu_segs[seg_cnt].double_delta_next = 10000;

        current->next = &local_gpu_segs[seg_cnt];
        local_gpu_segs[seg_cnt].prev = current;
        current = &local_gpu_segs[seg_cnt];
        current->next = &tail;
        tail.prev = current;


        bool flag = false;

        for (int aim_bit = min_second_bit; aim_bit <= max_second_bit;aim_bit++) {
            current = (&head)->next;
            // if (blockIdx.x == 4 && threadIdx.x == 31)
            // {
            //     printf("Current node address: %p, Next node address: %p, Tail address: %p\n", current, current->next, tail);
            // }
            // printf("Current node address: %p, Next node address: %p\n", current, current->next);
            while (current && current != &tail && current->next != &tail) {

                if (current->double_delta_next == aim_bit) {
                    Segment<int64_t>* next = current->next;
                    int former_index = current->start_index; // former_index ~ start_index - 1
                    int start_index = next->start_index;
                    int now_index = next->end_index; // start_index ~ now_index
                    int left_bit_origin = cal_bits<T>(current->min_delta, current->max_delta);
                    int right_bit_origin = cal_bits<T>(next->min_delta, next->max_delta);

                    int64_t new_max_delta = max(current->max_delta, next->max_delta);
                    new_max_delta = max(new_max_delta, current->next_delta);
                    int64_t new_min_delta = min(current->min_delta, next->min_delta);
                    new_min_delta = min(new_min_delta, current->next_delta);
                    int new_bit = cal_bits<T>(new_min_delta, new_max_delta);

                    int origin_cost = (start_index - former_index) * left_bit_origin + (now_index - start_index + 1) * right_bit_origin;
                    int merged_cost = new_bit * (now_index - former_index + 1);
                    if (merged_cost - origin_cost < overhead) {
                        // merge
                        current->end_index = now_index;
                        current->next = next->next;
                        next->next->prev = current;
                        current->next_delta = next->next_delta;
                        current->max_delta = new_max_delta;
                        current->min_delta = new_min_delta;
                        current->double_delta_next = next->double_delta_next;

                    }
                    else {
                        current = current->next;
                        continue;
                    }
                    // look left
                    Segment<int64_t>* prev = current->prev;
                    while (prev != &head && prev->prev != &head) {
                        int left_index = prev->start_index;
                        int64_t left_max_delta = max(prev->max_delta, current->max_delta);
                        left_max_delta = max(left_max_delta, prev->next_delta);
                        int64_t left_min_delta = min(prev->min_delta, current->min_delta);
                        left_min_delta = min(left_min_delta, prev->next_delta);

                        int new_bit = cal_bits<T>(left_min_delta, left_max_delta);
                        int origin_left_delta_bit = cal_bits<T>(prev->min_delta, prev->max_delta);
                        int origin_right_delta_bit = cal_bits<T>(current->min_delta, current->max_delta);
                        int origin_cost = (current->start_index - left_index) * origin_left_delta_bit + (current->end_index - current->start_index + 1) * origin_right_delta_bit;
                        int merged_cost = new_bit * (current->end_index - left_index + 1);

                        if (merged_cost - origin_cost < overhead) {
                            // merge
                            current->start_index = left_index;
                            current->prev = prev->prev;
                            prev->prev->next = current;
                            current->min_delta = left_min_delta;
                            current->max_delta = left_max_delta;
                            prev = current->prev;
                        }
                        else {

                            break;
                        }

                    }

                    next = current->next;
                    while (next != &tail && next->next != &tail) {
                        int right_index = next->end_index;
                        int64_t right_max_delta = max(next->max_delta, current->max_delta);
                        right_max_delta = max(right_max_delta, current->next_delta);
                        int64_t right_min_delta = min(next->min_delta, current->min_delta);
                        right_min_delta = min(right_min_delta, current->next_delta);

                        int new_bit = cal_bits<T>(right_min_delta, right_max_delta);
                        int origin_left_delta_bit = cal_bits<T>(current->min_delta, current->max_delta);
                        int origin_right_delta_bit = cal_bits<T>(next->min_delta, next->max_delta);
                        int origin_cost = (right_index - next->start_index + 1) * origin_right_delta_bit + (next->start_index - current->start_index) * origin_left_delta_bit;
                        int merged_cost = new_bit * (right_index - current->start_index + 1);

                        if (merged_cost - origin_cost < overhead) {
                            // merge
                            current->end_index = right_index;
                            current->next = next->next;
                            next->next->prev = current;
                            current->max_delta = right_max_delta;
                            current->min_delta = right_min_delta;
                            current->double_delta_next = next->double_delta_next;
                            current->next_delta = next->next_delta;
                            next = current->next;
                        }
                        else {
                            break;
                        }

                    }

                    current = current->next;
                }
                else{
                    current = current->next;
                }

            }
        }

        uint32_t *local_segment_index_gpu = segment_index_gpu + tid * block_size * 2;
        size_t local_segment_index_cnt = 0;
        
        while (current && current->next != &tail)
        {
            // printf("enter while loop! tid = %d\n", tid);
            local_segment_index_gpu[local_segment_index_cnt ++ ] = current->start_index;
            current = current->next;
        }

        int local_segment_total = local_segment_index_cnt;
        local_segment_index_gpu[local_segment_index_cnt ++ ] = local_block_size;

        size_t local_total_byte = 0;

        uint32_t *local_segment_length_gpu = segment_length_gpu + tid * block_size * 2;
        size_t local_segment_length_cnt = 0;

        for (int i = 0;i < local_segment_total;i ++ )
        {
            uint64_t tmp_size = newsegment_size(local_segment_index_gpu[i], local_segment_index_gpu[i + 1] - 1, array, block_size, tid);
            local_total_byte += tmp_size;
            local_segment_length_gpu[local_segment_length_cnt ++ ] = tmp_size;
        }


        local_segment_index_cnt -- ;

        int iter = 0;

        uint64_t cost_decline = local_total_byte;

        int loop_cnt = 0;
        
        while(cost_decline > 0)
        {
            iter ++ ;
            cost_decline = local_total_byte;
            loop_cnt ++ ;
            local_total_byte = merge_both_direction(loop_cnt, tid, block_size, local_segment_index_gpu,
            local_segment_index_cnt,  
            local_segment_length_gpu, local_segment_length_cnt, new_segment_index_gpu, new_segment_length_gpu, array, local_block_size);
            double compressrate = (local_total_byte) * 100.0 / (sizeof(T) * block_size * 1.0);
            cost_decline = cost_decline - local_total_byte;
            double cost_decline_percent = cost_decline * 100.0 / (sizeof(T) * block_size * 1.0);
            if (cost_decline_percent < 0.01) {
                break;
            }
        }

        int segment_number = (int)local_segment_index_cnt;
        
        uint8_t *local_res_total = res_total_gpu + tid * block_size * 8 + 1000 * tid;

        uint32_t *local_segment_index_total = segment_index_total + tid * block_size * 2;
        size_t local_segment_index_total_cnt = 1;

        uint32_t *local_segment_length_total = segment_length_total + tid * block_size * 2;
        size_t local_segment_length_total_cnt = 1;

        size_t total_byte_total = 0;


        bool *signvec = signvec_gpu + tid * block_size * 2;
        T* delta_final = delta_final_gpu + tid * block_size * 2;
        

        // if (segment_number <= 1)
        // {

        //     // printf("tid = %d\n", tid);
        //     local_segment_index_cnt = 0;
        //     local_segment_index_gpu[local_segment_index_cnt ++ ] = 0;
        //     local_segment_index_gpu[local_segment_index_cnt ++ ] = local_block_size;
        //     newsegment(0, local_block_size - 1, array, local_segment_index_total, local_segment_index_total_cnt, local_segment_length_total, local_segment_length_total_cnt,
        //     tid, 
        //     local_res_total, total_byte_total, block_size, local_block_size, signvec, delta_final);
        // }
        // else{
        //     local_segment_index_gpu[local_segment_index_cnt ++ ] = local_block_size;
        //     for (int i = 0;i < segment_number;i ++ )
        //     {

        //         Pair p = newsegment(local_segment_index_gpu[i], local_segment_index_gpu[i + 1] - 1, array, local_segment_index_total, local_segment_index_total_cnt, local_segment_length_total, local_segment_length_total_cnt, tid, local_res_total, total_byte_total, block_size, local_block_size, signvec, delta_final);

        //         local_res_total = p.out;
        //     }

            
        // }

        local_segment_index_gpu[local_segment_index_cnt ++ ] = local_block_size;
        for (int i = 0;i < segment_number;i ++ )
        {

            Pair p = newsegment(local_segment_index_gpu[i], local_segment_index_gpu[i + 1] - 1, array, local_segment_index_total, local_segment_index_total_cnt, local_segment_length_total, local_segment_length_total_cnt, tid, local_res_total, total_byte_total, block_size, local_block_size, signvec, delta_final);

            local_res_total = p.out;
        }

        

        local_segment_index_cnt -- ;

        local_segment_index_total[0] = local_segment_index_total_cnt;
        local_segment_length_total[0] = local_segment_length_total_cnt;

        atomicAdd(segment_cnt_total, (int)local_segment_index_total_cnt);

        // printf("total_byte_total = %lu\n", total_byte_total);
        
        byte_count[tid] = total_byte_total;


        return ;

    }

    template <typename T>
    __global__ void kernel_decodeArray8_var(uint8_t *gpu_in, T *out, int blocks, int block_size, uint32_t* segment_index_total, uint32_t *segment_length_total, size_t data_size) {
        

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int local_block_size;
        if (tid >= blocks)
            return ;
        
        if (tid == blocks - 1)
        {
            local_block_size = data_size - tid * block_size;
        }
        else
        {
            local_block_size = block_size;
        }


        T* res = out + block_size * tid;
        uint8_t* local_gpu_in = gpu_in + block_size * tid * 8 + 1000 * tid;


        uint32_t *local_segment_index_total = segment_index_total + tid * block_size * 2;
        
        uint32_t *local_segment_length_total = segment_length_total + tid * block_size * 2;

        size_t local_segment_index_total_cnt = local_segment_index_total[0];

        local_segment_index_total[local_segment_index_total_cnt ++ ] = local_block_size;

        float theta0 = 0;
        float theta1 = 0;
        uint8_t maxerror;
        

        for (int i = 1;i < local_segment_index_total_cnt - 1;i ++ )
        {
            int segment_length = local_segment_index_total[i + 1] - local_segment_index_total[i];


            uint8_t* tmpin = local_gpu_in;

            uint32_t start_ind = 0;
            memcpy(&start_ind, tmpin, sizeof(uint32_t));

            // if (tid == 4 && i < 100)
            // {
            //     printf("start_ind = %d\n", start_ind);
            // }

            tmpin += sizeof(uint32_t);
            maxerror = tmpin[0];
            tmpin ++;

            int index = tid * block_size + start_ind; 
            if (maxerror == 255) {
                T tmp_val;
                memcpy(&tmp_val, tmpin, sizeof(tmp_val));
                res[0] = tmp_val;
                res++;
                local_gpu_in += local_segment_length_total[i];
                continue;
            }
            if (maxerror == 254) {
                T tmp_val;
                memcpy(&tmp_val, tmpin, sizeof(tmp_val));
                res[0] = tmp_val;

                res++;
                memcpy(&tmp_val, tmpin + sizeof(T), sizeof(tmp_val));
                res[0] = tmp_val;
                res++;

                local_gpu_in += local_segment_length_total[i];
                continue;
            }
            if (maxerror==sizeof(T)*8) {

                memcpy(res, tmpin, sizeof(T) * segment_length);
                res += segment_length;
                local_gpu_in += local_segment_length_total[i];
                continue;
            }

            memcpy(&theta0, tmpin, sizeof(theta0));
            tmpin += sizeof(theta0);
            theta1 = 0;
            if ((maxerror >> 7) == 1) {
                memcpy(&theta1, tmpin, sizeof(theta1));
                tmpin += sizeof(theta1);
                maxerror -= 128;
            }
            if (maxerror) {
                read_all_bit_fix_gpu<T>(tmpin, 0, 0, segment_length, maxerror, theta1, theta0, res, index);

            }
            else {
                for (int j = 0;j < segment_length;j ++ )
                {
                    res[j] = (long long)(theta0 + theta1 * (double)(j));
                }
                
            }
            res += segment_length;
            local_gpu_in += local_segment_length_total[i];
        }
        

        return ;
    }

    // int get_segment_id(int to_find) {
    //     // int segment_id = art.upper_bound_new(to_find, search_node) - 1;
    //     // int segment_id = lower_bound(to_find, segment_index_total.size(), segment_index_total);
    //     int segment_id = alex_tree.upper_bound(to_find).payload() - 1;
    //     __builtin_prefetch(block_start_vec_total.data() + segment_id, 0, 3);
    //     return segment_id;
    // }

    // template <typename T>
    // T randomdecodeArray8(int segment_id, uint8_t* in, int to_find, uint32_t* out, size_t nvalue) {


    //     // uint32_t length = segment_index_total.size();

    //     // use btree to find the segment
    //     // auto it = btree_total.upper_bound(to_find);
    //     // int segment_num = it.data();
    //     // uint8_t* this_block = block_start_vec_total[segment_num-1];

    //     // use ALEX
    //     // auto it = alex_tree.upper_bound(to_find);
    //     //  segment_id = it.payload() - 1;

    //     // use ART
    //     // s td::cout<<to_find<<std::endl;

    //     // int segment_id = art.upper_bound_new(to_find, search_node) - 1;

    //     // std::cout<<to_find<<" "<<segment_id<<std::endl;

    //     // normal binary search
    //     // segment_id = lower_bound(to_find, length, segment_index_total);
    //     // int segment_id = binarySearch2(0, length-1, to_find, segment_index_total);


    //     uint8_t* this_block = block_start_vec_total[segment_id];

    //     uint8_t* tmpin = this_block;

    //     uint32_t start_ind;
    //     memcpy(&start_ind, tmpin, 4);
    //     tmpin += 4;

    //     uint8_t maxerror;
    //     memcpy(&maxerror, tmpin, 1);
    //     tmpin++;
    //     if (maxerror == sizeof(T) * 8) {
    //         T tmp_val = reinterpret_cast<T*>(tmpin)[to_find-start_ind];
    //         return tmp_val;
    //     }

    //     T tmp_val = 0;
    //     if (maxerror == 255) {
    //         memcpy(&tmp_val, tmpin, sizeof(tmp_val));
    //         return tmp_val;
    //     }
    //     if (maxerror == 254) {
    //         if (to_find - start_ind == 0) {
    //             memcpy(&tmp_val, tmpin, sizeof(tmp_val));
    //         }
    //         else {
    //             tmpin += sizeof(tmp_val);
    //             memcpy(&tmp_val, tmpin, sizeof(tmp_val));
    //         }
    //         return tmp_val;
    //     }


    //     float theta0;
    //     memcpy(&theta0, tmpin, sizeof(theta0));
    //     tmpin += sizeof(theta0);

    //     float theta1=0;
    //     if((maxerror>>7) == 1){
    //         memcpy(&theta1, tmpin, sizeof(theta1));
    //         tmpin += sizeof(theta1);
    //         maxerror -= 128;
    //     } 


    //     if (maxerror) {
    //         // tmp_val = read_bit_fix_int_float<T>(tmpin, maxerror, to_find-start_ind, theta1, theta0);
    //         // tmp_val = read_bit_fix_int_float<T>(tmpin, maxerror, to_find - start_ind, theta1, theta0);
    //         tmp_val = read_bit_fix_T(tmpin, maxerror, to_find - start_ind, (double)theta1, theta0, 0);
    //     }
    //     else {
    //         tmp_val = (T)(theta0 + theta1 * (double)(to_find - start_ind));
    //     }



    //     return tmp_val;

    // }


    __device__ uint64_t summation(uint8_t* in, const size_t l, size_t nvalue) {

        return 0;
    }
    // __device__ uint32_t* encodeArray(uint32_t* in, const size_t length, uint32_t* out,
    //     size_t nvalue) {
    //     std::cout << "Haven't implement. Please try uint8_t one..." << std::endl;
    //     return out;
    // }
    // __device__ uint32_t* decodeArray(uint32_t* in, const size_t length,
    //     uint32_t* out, size_t nvalue) {
    //     std::cout << "Haven't implement. Please try uint8_t one..." << std::endl;
    //     return out;
    // }
    // __device__ uint32_t randomdecodeArray(uint32_t* in, const size_t l, uint32_t* out, size_t nvalue) {
    //     std::cout << "Haven't implement. Please try uint8_t one..." << std::endl;
    //     return 1;
    // }
    // __device__ uint32_t get_total_byte() {
    //     return total_byte_total;
    // }
    // uint32_t get_total_blocks() {
    //     // std::cout << "split time " << split_time << std::endl;
    //     // std::cout << "merge time " << merge_time << std::endl;
    //     return block_start_vec_total.size();
    // }


    



