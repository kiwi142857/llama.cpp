#pragma once

#include "ggml.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

#ifdef __cplusplus

#include <utility>

// 为CPU亲和性添加必要的头文件
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
static inline ggml_fp16_t f32_to_f16(float x) {
    return GGML_CPU_FP32_TO_FP16(x);
}

static inline float f16_to_f32(ggml_fp16_t x) {
    return GGML_CPU_FP16_TO_FP32(x);
}

static inline ggml_bf16_t f32_to_bf16(float x) {
    return GGML_FP32_TO_BF16(x);
}

static inline float bf16_to_f32(ggml_bf16_t x) {
    return GGML_BF16_TO_FP32(x);
}

static inline float f32_to_f32(float x) {
    return x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
struct type_conversion_table;

template <>
struct type_conversion_table<ggml_fp16_t> {
    static constexpr float (*to_f32)(ggml_fp16_t) = f16_to_f32;
    static constexpr ggml_fp16_t (*from_f32)(float) = f32_to_f16;
};

template <>
struct type_conversion_table<float> {
    static constexpr float (*to_f32)(float) = f32_to_f32;
    static constexpr float (*from_f32)(float) = f32_to_f32;
};

template <>
struct type_conversion_table<ggml_bf16_t> {
    static constexpr float (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    static constexpr ggml_bf16_t (*from_f32)(float) = f32_to_bf16;
};

// 修改方案：按照ith绑定到对应的cpu核心上，每个cpu核心负责的行数不是固定的dr,而是按照比例来，根据cpu每个核心的频率来计算，频率高的核心负责的行数多，频率低的核心负责的行数少
// 当前我0～3核最高频率为1800000，4～7核最高频率为2400000。因此按照这个比例来计算，前4核的负载为每个核占3/((3+4)*4)，后4核的负载为每个核占4/((3+4)*4)
// 因此，前4核的负载为每个核占3/((3+4)*4)，后4核的负载为每个核占4/((3+4)*4)
// 因此，前4核的负载为每个核占3/((3+4)*4)，后4核的负载为每个核占4/((3+4)*4)
// 我们修改get_thread_range函数，按照这个比例来计算，前4核的负载为每个核占3/((3+4)*4)，后4核的负载为每个核占4/((3+4)*4)
// 并且根据ith来计算，前4核的ith为0～3，后4核的ith为4～7，把每个线程绑定到对应的核心上

static std::pair<int64_t, int64_t> get_thread_range(const struct ggml_compute_params * params, const struct ggml_tensor * src0) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;
    const int64_t nr  = ggml_nrows(src0);

    // 硬编码CPU频率权重：前4核权重3，后4核权重4
    // 0-3核：1800000 kHz (权重3)
    // 4-7核：2400000 kHz (权重4)
    int weight_low = 3;
    int weight_high = 4;
    
    // 预先计算每个线程的行范围，避免浮点数舍入误差
    std::vector<int64_t> thread_starts(nth + 1, 0);
    
    // 计算总权重
    int64_t total_weight = 0;
    for (int i = 0; i < nth; ++i) {
        if (i < 4) {
            total_weight += weight_low;  // 前4核权重
        } else if (i < 8) {
            total_weight += weight_high;  // 后4核权重
        } else {
            total_weight += weight_high;  // 超过8核的话，默认使用大核权重
        }
    }
    
    // 逐个计算每个线程的起始行
    int64_t allocated_rows = 0;
    for (int i = 0; i < nth; ++i) {
        thread_starts[i] = allocated_rows;
        
        if (i == nth - 1) {
            // 最后一个线程处理所有剩余行，避免舍入误差
            thread_starts[i + 1] = nr;
        } else {
            // 计算该线程应该处理的行数
            int64_t weight = (i < 4) ? weight_low : ((i < 8) ? weight_high : weight_high);
            int64_t rows_for_thread = (nr * weight) / total_weight;
            allocated_rows += rows_for_thread;
            thread_starts[i + 1] = allocated_rows;
        }
    }
    
    const int64_t ir0 = thread_starts[ith];
    const int64_t ir1 = thread_starts[ith + 1];

    // 绑定线程到对应的cpu核心上
    // 前4核的ith为0～3，后4核的ith为4～7，把每个线程绑定到对应的核心上
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    // 根据ith值绑定到对应的CPU核心
    if (ith < 8) {  // 只处理前8个核心
        if(nth>4)
        {
            CPU_SET(ith, &cpuset);
        }
        else if(nth==4)
        {
            CPU_SET(ith+4, &cpuset);
        }
        int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (result == 0) {
            // 绑定成功，可以记录日志
            if(ir0!=0 || ir1!=0)
            {
                // GGML_LOG_INFO("Thread start row: %ld, end row: %ld, ith: %ld, nth: %ld, nr: %ld, num_rows: %ld\n", ir0, ir1, ith, nth, nr, ir1-ir0);
            }
        }
    }
#endif
    
    return {ir0, ir1};
}

#endif
