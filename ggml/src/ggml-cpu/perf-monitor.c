#include "perf-monitor.h"
#include "ggml.h"

#if GGML_CPU_PERF_MONITOR

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

// 全局性能监控器实例
static struct ggml_perf_monitor g_perf_monitor = {0};

// 每个线程当前正在执行的操作开始时间栈（支持嵌套调用）
#define MAX_OP_STACK_DEPTH 64
static __thread struct {
    int64_t start_time;
    enum ggml_op op_type;
} g_op_stack[MAX_OP_STACK_DEPTH] = {0};
static __thread int g_op_stack_top = 0;

// 每个线程自定义函数的开始时间栈
static __thread struct {
    int64_t start_time;
    enum ggml_perf_custom_func func_type;
} g_custom_func_stack[MAX_OP_STACK_DEPTH] = {0};
static __thread int g_custom_func_stack_top = 0;

// 自定义函数名称映射
static const char* ggml_perf_custom_func_name(enum ggml_perf_custom_func func) {
    switch (func) {
        case GGML_PERF_FUNC_MUL_MAT_ONE_CHUNK:    return "mul_mat_one_chunk";
        case GGML_PERF_FUNC_MUL_MAT_ID_ONE_CHUNK: return "mul_mat_id_one_chunk";
        case GGML_PERF_FUNC_MUL_MAT_PRE_CHUNK: return "mul_mat_pre_chunk";
        case GGML_PERF_FUNC_MUL_MAT_ID_PRE_CHUNK: return "mul_mat_id_pre_chunk";
        default: return "unknown";
    }
}

// 初始化性能监控器
void ggml_perf_monitor_init(void) {
    memset(&g_perf_monitor, 0, sizeof(g_perf_monitor));
    g_perf_monitor.enabled = false;
    g_perf_monitor.max_threads = GGML_MAX_N_THREADS;
    g_perf_monitor.monitor_start_time_us = ggml_time_us();
    
    printf("DEBUG: 性能监控器已初始化，最大线程数: %d\n", GGML_MAX_N_THREADS);
    
    // 初始化每个线程的统计数据
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        g_perf_monitor.threads[t].thread_id = t;
        g_perf_monitor.threads[t].active = false;
        g_perf_monitor.threads[t].total_compute_time_us = 0;
        g_perf_monitor.threads[t].total_custom_time_us = 0;
        g_perf_monitor.threads[t].chunk_acquisitions_count = 0;
        
        for (int op = 0; op < GGML_OP_COUNT; op++) {
            g_perf_monitor.threads[t].ops[op].total_time_us = 0;
            g_perf_monitor.threads[t].ops[op].count = 0;
            g_perf_monitor.threads[t].ops[op].min_time_us = INT64_MAX;
            g_perf_monitor.threads[t].ops[op].max_time_us = 0;
        }
        
        // 初始化自定义函数统计
        for (int func = 0; func < GGML_PERF_FUNC_COUNT; func++) {
            g_perf_monitor.threads[t].custom_funcs[func].total_time_us = 0;
            g_perf_monitor.threads[t].custom_funcs[func].count = 0;
            g_perf_monitor.threads[t].custom_funcs[func].min_time_us = INT64_MAX;
            g_perf_monitor.threads[t].custom_funcs[func].max_time_us = 0;
            g_perf_monitor.threads[t].custom_funcs[func].avg_time_us = 0.0;
        }
    }
}

// 释放性能监控器
void ggml_perf_monitor_free(void) {
    // 当前实现中没有需要释放的动态内存
}

// 启用/禁用性能监控
void ggml_perf_monitor_enable(bool enable) {
    g_perf_monitor.enabled = enable;
    printf("DEBUG: 性能监控 %s\n", enable ? "已启用" : "已禁用");
    if (enable) {
        g_perf_monitor.monitor_start_time_us = ggml_time_us();
    }
}

// 重置性能统计
void ggml_perf_monitor_reset(void) {
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        g_perf_monitor.threads[t].total_compute_time_us = 0;
        g_perf_monitor.threads[t].total_custom_time_us = 0;
        g_perf_monitor.threads[t].active = false;
        g_perf_monitor.threads[t].chunk_acquisitions_count = 0;
        
        for (int op = 0; op < GGML_OP_COUNT; op++) {
            g_perf_monitor.threads[t].ops[op].total_time_us = 0;
            g_perf_monitor.threads[t].ops[op].count = 0;
            g_perf_monitor.threads[t].ops[op].min_time_us = INT64_MAX;
            g_perf_monitor.threads[t].ops[op].max_time_us = 0;
        }
        
        // 重置自定义函数统计
        for (int func = 0; func < GGML_PERF_FUNC_COUNT; func++) {
            g_perf_monitor.threads[t].custom_funcs[func].total_time_us = 0;
            g_perf_monitor.threads[t].custom_funcs[func].count = 0;
            g_perf_monitor.threads[t].custom_funcs[func].min_time_us = INT64_MAX;
            g_perf_monitor.threads[t].custom_funcs[func].max_time_us = 0;
            g_perf_monitor.threads[t].custom_funcs[func].avg_time_us = 0.0;
        }
    }
    g_perf_monitor.monitor_start_time_us = ggml_time_us();
}

// 开始记录操作时间
void ggml_perf_op_start(int thread_id, enum ggml_op op_type) {
    if (!g_perf_monitor.enabled || thread_id >= GGML_MAX_N_THREADS || op_type >= GGML_OP_COUNT) {
        return;
    }
    
    g_perf_monitor.threads[thread_id].active = true;
    
    // 将操作压入栈
    if (g_op_stack_top < MAX_OP_STACK_DEPTH) {
        g_op_stack[g_op_stack_top].start_time = ggml_time_us();
        g_op_stack[g_op_stack_top].op_type = op_type;
        g_op_stack_top++;
        
        // 调试信息
        static int call_count = 0;
        if (call_count < 5) {
            printf("DEBUG: 开始监控操作 %s (线程 %d, 栈深度 %d)\n", 
                   ggml_op_name(op_type), thread_id, g_op_stack_top);
            call_count++;
        }
    }
}

// 结束记录操作时间
void ggml_perf_op_end(int thread_id, enum ggml_op op_type) {
    if (!g_perf_monitor.enabled || thread_id >= GGML_MAX_N_THREADS || op_type >= GGML_OP_COUNT) {
        return;
    }
    
    // 从栈中弹出操作
    if (g_op_stack_top <= 0) {
        return; // 栈为空
    }
    
    g_op_stack_top--;
    
    // 检查操作类型是否匹配
    if (g_op_stack[g_op_stack_top].op_type != op_type) {
        // 操作类型不匹配，可能是嵌套调用的问题
        // 尝试在栈中查找匹配的操作
        int found = -1;
        for (int i = g_op_stack_top; i >= 0; i--) {
            if (g_op_stack[i].op_type == op_type) {
                found = i;
                break;
            }
        }
        
        if (found == -1) {
            printf("警告：找不到匹配的操作开始时间 %s (线程 %d)\n", 
                   ggml_op_name(op_type), thread_id);
            return;
        }
        
        // 使用找到的操作，并调整栈顶
        g_op_stack_top = found;
    }
    
    int64_t end_time = ggml_time_us();
    int64_t start_time = g_op_stack[g_op_stack_top].start_time;
    int64_t duration = end_time - start_time;
    
    struct ggml_perf_op_record* record = &g_perf_monitor.threads[thread_id].ops[op_type];
    
    // 更新统计信息
    record->total_time_us += duration;
    record->count++;
    
    if (record->count == 1 || duration < record->min_time_us) {
        record->min_time_us = duration;
    }
    if (duration > record->max_time_us) {
        record->max_time_us = duration;
    }
    
    g_perf_monitor.threads[thread_id].total_compute_time_us += duration;
    
    // 调试信息
    static int end_call_count = 0;
    if (end_call_count < 5) {
        printf("DEBUG: 结束监控操作 %s (线程 %d, 耗时 %.2f ms)\n", 
               ggml_op_name(op_type), thread_id, duration / 1000.0);
        end_call_count++;
    }
}

// 开始记录自定义函数时间
void ggml_perf_custom_func_start(int thread_id, enum ggml_perf_custom_func func_type) {
    if (!g_perf_monitor.enabled || thread_id >= GGML_MAX_N_THREADS || func_type >= GGML_PERF_FUNC_COUNT) {
        return;
    }
    
    g_perf_monitor.threads[thread_id].active = true;
    
    // 将自定义函数压入栈
    if (g_custom_func_stack_top < MAX_OP_STACK_DEPTH) {
        g_custom_func_stack[g_custom_func_stack_top].start_time = ggml_time_us();
        g_custom_func_stack[g_custom_func_stack_top].func_type = func_type;
        g_custom_func_stack_top++;
        
        // 调试信息
        static int call_count = 0;
        if (call_count < 5) {
            printf("DEBUG: 开始监控自定义函数 %s (线程 %d, 栈深度 %d)\n", 
                   ggml_perf_custom_func_name(func_type), thread_id, g_custom_func_stack_top);
            call_count++;
        }
    }
}

// 结束记录自定义函数时间
void ggml_perf_custom_func_end(int thread_id, enum ggml_perf_custom_func func_type) {
    if (!g_perf_monitor.enabled || thread_id >= GGML_MAX_N_THREADS || func_type >= GGML_PERF_FUNC_COUNT) {
        return;
    }
    
    // 从栈中弹出自定义函数
    if (g_custom_func_stack_top <= 0) {
        return; // 栈为空
    }
    
    g_custom_func_stack_top--;
    
    // 检查函数类型是否匹配
    if (g_custom_func_stack[g_custom_func_stack_top].func_type != func_type) {
        // 函数类型不匹配，尝试在栈中查找匹配的函数
        int found = -1;
        for (int i = g_custom_func_stack_top; i >= 0; i--) {
            if (g_custom_func_stack[i].func_type == func_type) {
                found = i;
                break;
            }
        }
        
        if (found == -1) {
            printf("警告：找不到匹配的自定义函数开始时间 %s (线程 %d)\n", 
                   ggml_perf_custom_func_name(func_type), thread_id);
            return;
        }
        
        // 使用找到的函数，并调整栈顶
        g_custom_func_stack_top = found;
    }
    
    int64_t end_time = ggml_time_us();
    int64_t start_time = g_custom_func_stack[g_custom_func_stack_top].start_time;
    int64_t duration = end_time - start_time;
    
    struct ggml_perf_custom_record* record = &g_perf_monitor.threads[thread_id].custom_funcs[func_type];
    
    // 更新统计信息
    record->total_time_us += duration;
    record->count++;
    
    if (record->count == 1 || duration < record->min_time_us) {
        record->min_time_us = duration;
    }
    if (duration > record->max_time_us) {
        record->max_time_us = duration;
    }
    
    // 更新平均时间（缓存计算结果）
    record->avg_time_us = (double)record->total_time_us / record->count;
    
    g_perf_monitor.threads[thread_id].total_custom_time_us += duration;
    
    // 调试信息
    static int end_call_count = 0;
    if (end_call_count < 5) {
        printf("DEBUG: 结束监控自定义函数 %s (线程 %d, 耗时 %.2f ms)\n", 
               ggml_perf_custom_func_name(func_type), thread_id, duration / 1000.0);
        end_call_count++;
    }
}

// 记录chunk抢占
void ggml_perf_record_chunk_acquisition(int thread_id) {
    if (!g_perf_monitor.enabled || thread_id >= GGML_MAX_N_THREADS) {
        return;
    }
    
    g_perf_monitor.threads[thread_id].active = true;
    g_perf_monitor.threads[thread_id].chunk_acquisitions_count++;
    
    // 调试信息
    static int chunk_acq_debug_count = 0;
    if (chunk_acq_debug_count < 5) {
        printf("DEBUG: 线程 %d 抢占chunk (总计: %ld)\n", 
               thread_id, g_perf_monitor.threads[thread_id].chunk_acquisitions_count);
        chunk_acq_debug_count++;
    }
}

// 打印简要统计信息
void ggml_perf_monitor_print_summary(void) {
    if (!g_perf_monitor.enabled) {
        printf("性能监控未启用\n");
        return;
    }
    
    int64_t total_time = ggml_time_us() - g_perf_monitor.monitor_start_time_us;
    
    printf("\n=== CPU 性能监控摘要 ===\n");
    printf("监控总时间: %.2f ms\n", total_time / 1000.0);
    printf("活跃线程数: ");
    
    int active_threads = 0;
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        if (g_perf_monitor.threads[t].active) {
            active_threads++;
        }
    }
    printf("%d\n", active_threads);
    
    printf("\n各线程计算时间:\n");
    printf("线程ID | 总计算时间(ms) | 自定义函数时间(ms) | Chunk抢占次数 | 利用率(%%)\n");
    printf("-------|---------------|------------------|--------------|----------\n");
    
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        if (g_perf_monitor.threads[t].active) {
            double compute_time_ms = g_perf_monitor.threads[t].total_compute_time_us / 1000.0;
            double custom_time_ms = g_perf_monitor.threads[t].total_custom_time_us / 1000.0;
            double utilization = (double)g_perf_monitor.threads[t].total_compute_time_us / total_time * 100.0;
            printf("%6d | %13.2f | %16.2f | %12ld | %8.1f\n", 
                   t, compute_time_ms, custom_time_ms, 
                   g_perf_monitor.threads[t].chunk_acquisitions_count, utilization);
        }
    }
    
    printf("\n热点操作类型 (所有线程汇总):\n");
    printf("操作类型 | 总时间(ms) | 调用次数 | 平均时间(us)\n");
    printf("---------|-----------|----------|-------------\n");
    
    // 汇总所有线程的操作统计
    struct ggml_perf_op_record total_ops[GGML_OP_COUNT] = {0};
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        if (g_perf_monitor.threads[t].active) {
            for (int op = 0; op < GGML_OP_COUNT; op++) {
                total_ops[op].total_time_us += g_perf_monitor.threads[t].ops[op].total_time_us;
                total_ops[op].count += g_perf_monitor.threads[t].ops[op].count;
            }
        }
    }
    
    // 只显示有执行的操作，按时间排序
    for (int op = 0; op < GGML_OP_COUNT; op++) {
        if (total_ops[op].count > 0) {
            double avg_time = (double)total_ops[op].total_time_us / total_ops[op].count;
            printf("%8s | %9.2f | %8ld | %11.1f\n", 
                   ggml_op_name((enum ggml_op)op),
                   total_ops[op].total_time_us / 1000.0,
                   total_ops[op].count,
                   avg_time);
        }
    }
    printf("\n");
}

// 打印MatMul Chunk函数的专门统计
void ggml_perf_monitor_print_matmul_chunks(void) {
    if (!g_perf_monitor.enabled) {
        printf("性能监控未启用\n");
        return;
    }
    
    printf("\n=== MatMul Chunk 函数性能分析 ===\n");
    
    // 汇总所有线程的自定义函数统计
    struct ggml_perf_custom_record total_funcs[GGML_PERF_FUNC_COUNT] = {0};
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        if (g_perf_monitor.threads[t].active) {
            for (int func = 0; func < GGML_PERF_FUNC_COUNT; func++) {
                total_funcs[func].total_time_us += g_perf_monitor.threads[t].custom_funcs[func].total_time_us;
                total_funcs[func].count += g_perf_monitor.threads[t].custom_funcs[func].count;
                if (g_perf_monitor.threads[t].custom_funcs[func].count > 0) {
                    if (total_funcs[func].min_time_us == 0 || 
                        g_perf_monitor.threads[t].custom_funcs[func].min_time_us < total_funcs[func].min_time_us) {
                        total_funcs[func].min_time_us = g_perf_monitor.threads[t].custom_funcs[func].min_time_us;
                    }
                    if (g_perf_monitor.threads[t].custom_funcs[func].max_time_us > total_funcs[func].max_time_us) {
                        total_funcs[func].max_time_us = g_perf_monitor.threads[t].custom_funcs[func].max_time_us;
                    }
                }
            }
        }
    }
    
    printf("\n汇总统计 (所有线程):\n");
    printf("函数名称                | 总时间(ms) | 调用次数 | 平均(us) | 最小(us) | 最大(us)\n");
    printf("------------------------|-----------|----------|----------|----------|----------\n");
    
    for (int func = 0; func < GGML_PERF_FUNC_COUNT; func++) {
        if (total_funcs[func].count > 0) {
            double avg_time = (double)total_funcs[func].total_time_us / total_funcs[func].count;
            printf("%-22s | %9.2f | %8ld | %8.1f | %8ld | %8ld\n",
                   ggml_perf_custom_func_name((enum ggml_perf_custom_func)func),
                   total_funcs[func].total_time_us / 1000.0,
                   total_funcs[func].count,
                   avg_time,
                   total_funcs[func].min_time_us,
                   total_funcs[func].max_time_us);
        }
    }
    
    printf("\n按线程详细统计:\n");
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        if (!g_perf_monitor.threads[t].active) continue;
        
        printf("\n--- 线程 %d ---\n", t);
        printf("函数名称                | 总时间(ms) | 调用次数 | 平均(us) | 最小(us) | 最大(us)\n");
        printf("------------------------|-----------|----------|----------|----------|----------\n");
        
        for (int func = 0; func < GGML_PERF_FUNC_COUNT; func++) {
            struct ggml_perf_custom_record* record = &g_perf_monitor.threads[t].custom_funcs[func];
            if (record->count > 0) {
                printf("%-22s | %9.2f | %8ld | %8.1f | %8ld | %8ld\n",
                       ggml_perf_custom_func_name((enum ggml_perf_custom_func)func),
                       record->total_time_us / 1000.0,
                       record->count,
                       record->avg_time_us,
                       record->min_time_us,
                       record->max_time_us);
            }
        }
    }
    printf("\n");
}

// 打印详细统计信息
void ggml_perf_monitor_print_detailed(void) {
    if (!g_perf_monitor.enabled) {
        printf("性能监控未启用\n");
        return;
    }
    
    printf("\n=== CPU 性能监控详细报告 ===\n");
    
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        if (!g_perf_monitor.threads[t].active) continue;
        
        printf("\n--- 线程 %d ---\n", t);
        printf("总计算时间: %.2f ms\n", g_perf_monitor.threads[t].total_compute_time_us / 1000.0);
        printf("自定义函数总时间: %.2f ms\n", g_perf_monitor.threads[t].total_custom_time_us / 1000.0);
        printf("Chunk抢占次数: %ld\n", g_perf_monitor.threads[t].chunk_acquisitions_count);
        printf("\n操作详情:\n");
        printf("操作类型 | 总时间(ms) | 调用次数 | 平均(us) | 最小(us) | 最大(us)\n");
        printf("---------|-----------|----------|----------|----------|----------\n");
        
        for (int op = 0; op < GGML_OP_COUNT; op++) {
            struct ggml_perf_op_record* record = &g_perf_monitor.threads[t].ops[op];
            if (record->count > 0) {
                double avg_time = (double)record->total_time_us / record->count;
                printf("%8s | %9.2f | %8ld | %8.1f | %8ld | %8ld\n",
                       ggml_op_name((enum ggml_op)op),
                       record->total_time_us / 1000.0,
                       record->count,
                       avg_time,
                       record->min_time_us,
                       record->max_time_us);
            }
        }
        
        printf("\n自定义函数详情:\n");
        printf("函数名称                | 总时间(ms) | 调用次数 | 平均(us) | 最小(us) | 最大(us)\n");
        printf("------------------------|-----------|----------|----------|----------|----------\n");
        
        for (int func = 0; func < GGML_PERF_FUNC_COUNT; func++) {
            struct ggml_perf_custom_record* record = &g_perf_monitor.threads[t].custom_funcs[func];
            if (record->count > 0) {
                printf("%-22s | %9.2f | %8ld | %8.1f | %8ld | %8ld\n",
                       ggml_perf_custom_func_name((enum ggml_perf_custom_func)func),
                       record->total_time_us / 1000.0,
                       record->count,
                       record->avg_time_us,
                       record->min_time_us,
                       record->max_time_us);
            }
        }
    }
    printf("\n");
}

// 导出MatMul Chunk函数的CSV
void ggml_perf_monitor_export_matmul_chunks_csv(const char* filename) {
    if (!g_perf_monitor.enabled) {
        printf("性能监控未启用，无法导出数据\n");
        return;
    }
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("无法创建文件: %s\n", filename);
        return;
    }
    
    // 写入CSV头部
    fprintf(fp, "线程ID,函数名称,总时间(ms),调用次数,平均时间(us),最小时间(us),最大时间(us)\n");
    
    // 写入数据
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        if (!g_perf_monitor.threads[t].active) continue;
        
        for (int func = 0; func < GGML_PERF_FUNC_COUNT; func++) {
            struct ggml_perf_custom_record* record = &g_perf_monitor.threads[t].custom_funcs[func];
            if (record->count > 0) {
                fprintf(fp, "%d,%s,%.3f,%ld,%.1f,%ld,%ld\n",
                       t,
                       ggml_perf_custom_func_name((enum ggml_perf_custom_func)func),
                       record->total_time_us / 1000.0,
                       record->count,
                       record->avg_time_us,
                       record->min_time_us,
                       record->max_time_us);
            }
        }
    }
    
    fclose(fp);
    printf("MatMul Chunk函数性能数据已导出到: %s\n", filename);
}

// 导出CSV格式
void ggml_perf_monitor_export_csv(const char* filename) {
    if (!g_perf_monitor.enabled) {
        printf("性能监控未启用，无法导出数据\n");
        return;
    }
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("无法创建文件: %s\n", filename);
        return;
    }
    
    // 写入CSV头部
    fprintf(fp, "线程ID,操作类型,总时间(ms),调用次数,平均时间(us),最小时间(us),最大时间(us)\n");
    
    // 写入数据
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        if (!g_perf_monitor.threads[t].active) continue;
        
        for (int op = 0; op < GGML_OP_COUNT; op++) {
            struct ggml_perf_op_record* record = &g_perf_monitor.threads[t].ops[op];
            if (record->count > 0) {
                double avg_time = (double)record->total_time_us / record->count;
                fprintf(fp, "%d,%s,%.3f,%ld,%.1f,%ld,%ld\n",
                       t,
                       ggml_op_name((enum ggml_op)op),
                       record->total_time_us / 1000.0,
                       record->count,
                       avg_time,
                       record->min_time_us,
                       record->max_time_us);
            }
        }
    }
    
    fclose(fp);
    printf("性能数据已导出到: %s\n", filename);
}

// 导出JSON格式
void ggml_perf_monitor_export_json(const char* filename) {
    if (!g_perf_monitor.enabled) {
        printf("性能监控未启用，无法导出数据\n");
        return;
    }
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("无法创建文件: %s\n", filename);
        return;
    }
    
    int64_t total_time = ggml_time_us() - g_perf_monitor.monitor_start_time_us;
    
    fprintf(fp, "{\n");
    fprintf(fp, "  \"monitor_duration_us\": %ld,\n", total_time);
    fprintf(fp, "  \"threads\": [\n");
    
    bool first_thread = true;
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        if (!g_perf_monitor.threads[t].active) continue;
        
        if (!first_thread) fprintf(fp, ",\n");
        first_thread = false;
        
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"thread_id\": %d,\n", t);
        fprintf(fp, "      \"total_compute_time_us\": %ld,\n", g_perf_monitor.threads[t].total_compute_time_us);
        fprintf(fp, "      \"total_custom_time_us\": %ld,\n", g_perf_monitor.threads[t].total_custom_time_us);
        fprintf(fp, "      \"chunk_acquisitions_count\": %ld,\n", g_perf_monitor.threads[t].chunk_acquisitions_count);
        fprintf(fp, "      \"operations\": [\n");
        
        bool first_op = true;
        for (int op = 0; op < GGML_OP_COUNT; op++) {
            struct ggml_perf_op_record* record = &g_perf_monitor.threads[t].ops[op];
            if (record->count > 0) {
                if (!first_op) fprintf(fp, ",\n");
                first_op = false;
                
                double avg_time = (double)record->total_time_us / record->count;
                fprintf(fp, "        {\n");
                fprintf(fp, "          \"op_type\": \"%s\",\n", ggml_op_name((enum ggml_op)op));
                fprintf(fp, "          \"total_time_us\": %ld,\n", record->total_time_us);
                fprintf(fp, "          \"count\": %ld,\n", record->count);
                fprintf(fp, "          \"avg_time_us\": %.1f,\n", avg_time);
                fprintf(fp, "          \"min_time_us\": %ld,\n", record->min_time_us);
                fprintf(fp, "          \"max_time_us\": %ld\n", record->max_time_us);
                fprintf(fp, "        }");
            }
        }
        
        fprintf(fp, "\n      ],\n");
        fprintf(fp, "      \"custom_functions\": [\n");
        
        bool first_func = true;
        for (int func = 0; func < GGML_PERF_FUNC_COUNT; func++) {
            struct ggml_perf_custom_record* record = &g_perf_monitor.threads[t].custom_funcs[func];
            if (record->count > 0) {
                if (!first_func) fprintf(fp, ",\n");
                first_func = false;
                
                fprintf(fp, "        {\n");
                fprintf(fp, "          \"func_name\": \"%s\",\n", ggml_perf_custom_func_name((enum ggml_perf_custom_func)func));
                fprintf(fp, "          \"total_time_us\": %ld,\n", record->total_time_us);
                fprintf(fp, "          \"count\": %ld,\n", record->count);
                fprintf(fp, "          \"avg_time_us\": %.1f,\n", record->avg_time_us);
                fprintf(fp, "          \"min_time_us\": %ld,\n", record->min_time_us);
                fprintf(fp, "          \"max_time_us\": %ld\n", record->max_time_us);
                fprintf(fp, "        }");
            }
        }
        
        fprintf(fp, "\n      ]\n");
        fprintf(fp, "    }");
    }
    
    fprintf(fp, "\n  ]\n");
    fprintf(fp, "}\n");
    
    fclose(fp);
    printf("性能数据已导出到: %s\n", filename);
}

#endif // GGML_CPU_PERF_MONITOR 