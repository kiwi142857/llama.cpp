#pragma once

#include "ggml.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// 启用性能监控的宏定义
#ifndef GGML_CPU_PERF_MONITOR
#define GGML_CPU_PERF_MONITOR 0
#endif

#if GGML_CPU_PERF_MONITOR

// 单个操作的性能记录
struct ggml_perf_op_record {
    int64_t total_time_us;        // 总执行时间（微秒）
    int64_t count;                // 执行次数
    int64_t min_time_us;          // 最小执行时间
    int64_t max_time_us;          // 最大执行时间
};

// 自定义函数类型枚举
enum ggml_perf_custom_func {
    GGML_PERF_FUNC_MUL_MAT_ONE_CHUNK = 0,
    GGML_PERF_FUNC_MUL_MAT_ID_ONE_CHUNK,
    GGML_PERF_FUNC_COUNT
};

// 单个自定义函数的性能记录
struct ggml_perf_custom_record {
    int64_t total_time_us;        // 总执行时间（微秒）
    int64_t count;                // 执行次数
    int64_t min_time_us;          // 最小执行时间
    int64_t max_time_us;          // 最大执行时间
    double avg_time_us;           // 平均执行时间（缓存计算结果）
};

// 每个线程的性能统计
struct ggml_perf_thread_stats {
    int thread_id;                                          // 线程ID
    struct ggml_perf_op_record ops[GGML_OP_COUNT];         // 每个操作类型的统计
    struct ggml_perf_custom_record custom_funcs[GGML_PERF_FUNC_COUNT]; // 自定义函数统计
    int64_t total_compute_time_us;                         // 该线程总计算时间
    int64_t total_custom_time_us;                         // 该线程自定义函数总时间
    bool active;                                           // 线程是否活跃
};

// 全局性能监控器
struct ggml_perf_monitor {
    struct ggml_perf_thread_stats threads[GGML_MAX_N_THREADS]; // 每个线程的统计
    int max_threads;                                            // 最大线程数
    bool enabled;                                               // 是否启用监控
    int64_t monitor_start_time_us;                             // 监控开始时间
};

// 性能监控接口
void ggml_perf_monitor_init(void);
void ggml_perf_monitor_free(void);
void ggml_perf_monitor_enable(bool enable);
void ggml_perf_monitor_reset(void);

// 操作时间记录接口
void ggml_perf_op_start(int thread_id, enum ggml_op op_type);
void ggml_perf_op_end(int thread_id, enum ggml_op op_type);

// 自定义函数时间记录接口
void ggml_perf_custom_func_start(int thread_id, enum ggml_perf_custom_func func_type);
void ggml_perf_custom_func_end(int thread_id, enum ggml_perf_custom_func func_type);

// 统计结果输出接口
void ggml_perf_monitor_print_summary(void);
void ggml_perf_monitor_print_detailed(void);
void ggml_perf_monitor_print_matmul_chunks(void);  // 专门打印matmul chunk函数的统计
void ggml_perf_monitor_export_csv(const char* filename);
void ggml_perf_monitor_export_json(const char* filename);
void ggml_perf_monitor_export_matmul_chunks_csv(const char* filename);  // 导出matmul chunk函数的CSV

// 便利宏定义
#define GGML_PERF_OP_START(params, op) \
    ggml_perf_op_start((params)->ith, (op))

#define GGML_PERF_OP_END(params, op) \
    ggml_perf_op_end((params)->ith, (op))

// 自定义函数监控宏
#define GGML_PERF_CUSTOM_FUNC_START(thread_id, func) \
    ggml_perf_custom_func_start((thread_id), (func))

#define GGML_PERF_CUSTOM_FUNC_END(thread_id, func) \
    ggml_perf_custom_func_end((thread_id), (func))

// 自动计时的RAII风格宏（需要C++支持）
#ifdef __cplusplus
class ggml_perf_timer {
private:
    int thread_id_;
    enum ggml_op op_;
public:
    ggml_perf_timer(int thread_id, enum ggml_op op) : thread_id_(thread_id), op_(op) {
        ggml_perf_op_start(thread_id_, op_);
    }
    ~ggml_perf_timer() {
        ggml_perf_op_end(thread_id_, op_);
    }
};

class ggml_perf_custom_timer {
private:
    int thread_id_;
    enum ggml_perf_custom_func func_;
public:
    ggml_perf_custom_timer(int thread_id, enum ggml_perf_custom_func func) : thread_id_(thread_id), func_(func) {
        ggml_perf_custom_func_start(thread_id_, func_);
    }
    ~ggml_perf_custom_timer() {
        ggml_perf_custom_func_end(thread_id_, func_);
    }
};

#define GGML_PERF_AUTO_TIMER(params, op) \
    ggml_perf_timer _perf_timer((params)->ith, (op))

#define GGML_PERF_AUTO_CUSTOM_TIMER(thread_id, func) \
    ggml_perf_custom_timer _perf_custom_timer((thread_id), (func))
#endif

#else // GGML_CPU_PERF_MONITOR

// 禁用监控时的空宏定义
#define ggml_perf_monitor_init()
#define ggml_perf_monitor_free()
#define ggml_perf_monitor_enable(enable)
#define ggml_perf_monitor_reset()
#define ggml_perf_op_start(thread_id, op_type)
#define ggml_perf_op_end(thread_id, op_type)
#define ggml_perf_custom_func_start(thread_id, func_type)
#define ggml_perf_custom_func_end(thread_id, func_type)
#define ggml_perf_monitor_print_summary()
#define ggml_perf_monitor_print_detailed()
#define ggml_perf_monitor_print_matmul_chunks()
#define ggml_perf_monitor_export_csv(filename)
#define ggml_perf_monitor_export_json(filename)
#define ggml_perf_monitor_export_matmul_chunks_csv(filename)
#define GGML_PERF_OP_START(params, op)
#define GGML_PERF_OP_END(params, op)
#define GGML_PERF_CUSTOM_FUNC_START(thread_id, func)
#define GGML_PERF_CUSTOM_FUNC_END(thread_id, func)

#ifdef __cplusplus
#define GGML_PERF_AUTO_TIMER(params, op)
#define GGML_PERF_AUTO_CUSTOM_TIMER(thread_id, func)
#endif

#endif // GGML_CPU_PERF_MONITOR

#ifdef __cplusplus
}
#endif 