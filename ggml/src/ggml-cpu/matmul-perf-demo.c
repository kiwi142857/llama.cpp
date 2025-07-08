#include "perf-monitor.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/*
 * MatMul Chunk 性能监控演示程序
 * 
 * 此程序演示如何使用新添加的性能监控功能来分析 matmul 操作中的
 * ggml_compute_forward_mul_mat_one_chunk 和 ggml_compute_forward_mul_mat_id_one_chunk
 * 函数的执行性能。
 * 
 * 编译时需要启用性能监控：
 * gcc -DGGML_CPU_PERF_MONITOR=1 -o matmul-perf-demo matmul-perf-demo.c perf-monitor.c -lm
 */

// 模拟函数执行来演示性能监控
void simulate_matmul_chunk_execution() {
    printf("模拟 MatMul Chunk 函数执行...\n");
    
    // 模拟不同线程执行 mul_mat_one_chunk
    for (int thread = 0; thread < 4; thread++) {
        for (int i = 0; i < 10 + thread * 5; i++) {
            GGML_PERF_CUSTOM_FUNC_START(thread, GGML_PERF_FUNC_MUL_MAT_ONE_CHUNK);
            
            // 模拟不同的执行时间
            usleep(1000 + (rand() % 2000)); // 1-3毫秒
            
            GGML_PERF_CUSTOM_FUNC_END(thread, GGML_PERF_FUNC_MUL_MAT_ONE_CHUNK);
        }
    }
    
    // 模拟不同线程执行 mul_mat_id_one_chunk
    for (int thread = 0; thread < 3; thread++) {
        for (int i = 0; i < 5 + thread * 3; i++) {
            GGML_PERF_CUSTOM_FUNC_START(thread, GGML_PERF_FUNC_MUL_MAT_ID_ONE_CHUNK);
            
            // 模拟不同的执行时间
            usleep(1500 + (rand() % 3000)); // 1.5-4.5毫秒
            
            GGML_PERF_CUSTOM_FUNC_END(thread, GGML_PERF_FUNC_MUL_MAT_ID_ONE_CHUNK);
        }
    }
}

int main(int argc, char** argv) {
    printf("=== MatMul Chunk 性能监控演示 ===\n\n");
    
    // 检查是否启用了性能监控
#if !GGML_CPU_PERF_MONITOR
    printf("错误: 性能监控未启用！\n");
    printf("请使用 -DGGML_CPU_PERF_MONITOR=1 编译选项重新编译。\n");
    return 1;
#endif

    // 初始化性能监控器
    printf("初始化性能监控器...\n");
    ggml_perf_monitor_init();
    ggml_perf_monitor_enable(true);
    
    // 重置统计数据
    ggml_perf_monitor_reset();
    
    printf("执行模拟的 MatMul Chunk 函数调用...\n");
    
    // 执行模拟的函数调用
    simulate_matmul_chunk_execution();
    
    printf("模拟执行完成！\n\n");
    
    // 输出性能分析结果
    printf("=== 性能监控结果 ===\n");
    
    // 打印总体摘要
    ggml_perf_monitor_print_summary();
    
    // 打印MatMul Chunk函数的专门分析
    ggml_perf_monitor_print_matmul_chunks();
    
    // 导出详细数据到文件
    printf("导出性能数据到文件...\n");
    ggml_perf_monitor_export_matmul_chunks_csv("matmul_chunks_perf.csv");
    ggml_perf_monitor_export_json("matmul_perf_detailed.json");
    
    printf("\n=== 性能分析完成 ===\n");
    printf("详细数据已保存到:\n");
    printf("- matmul_chunks_perf.csv (MatMul Chunk函数专门分析)\n");
    printf("- matmul_perf_detailed.json (完整性能数据)\n\n");
    
    printf("性能监控 API 使用示例:\n");
    printf("1. 初始化: ggml_perf_monitor_init()\n");
    printf("2. 启用: ggml_perf_monitor_enable(true)\n");
    printf("3. 开始计时: GGML_PERF_CUSTOM_FUNC_START(thread_id, func_type)\n");
    printf("4. 结束计时: GGML_PERF_CUSTOM_FUNC_END(thread_id, func_type)\n");
    printf("5. 打印结果: ggml_perf_monitor_print_matmul_chunks()\n");
    printf("6. 导出数据: ggml_perf_monitor_export_matmul_chunks_csv()\n");
    
    // 清理资源
    ggml_perf_monitor_free();
    
    return 0;
}

/*
 * 使用说明:
 * 
 * 1. 编译启用性能监控:
 *    gcc -DGGML_CPU_PERF_MONITOR=1 -I../../../ggml/src -o matmul-perf-demo matmul-perf-demo.c perf-monitor.c ggml.c -lm
 * 
 * 2. 运行程序:
 *    ./matmul-perf-demo
 * 
 * 3. 查看输出:
 *    - 控制台显示性能摘要和详细的MatMul chunk分析
 *    - matmul_chunks_perf.csv: 专门的chunk函数性能数据
 *    - matmul_perf_detailed.json: 完整的性能监控数据
 * 
 * 在实际的 GGML 代码中的集成:
 * 
 * 在 ggml-cpu.c 中的函数调用处已经添加了监控代码:
 * 
 * ```c
 * // 在 ggml_compute_forward_mul_mat 函数中:
 * GGML_PERF_CUSTOM_FUNC_START(params->ith, GGML_PERF_FUNC_MUL_MAT_ONE_CHUNK);
 * ggml_compute_forward_mul_mat_one_chunk(...);
 * GGML_PERF_CUSTOM_FUNC_END(params->ith, GGML_PERF_FUNC_MUL_MAT_ONE_CHUNK);
 * 
 * // 在 ggml_compute_forward_mul_mat_id 函数中:
 * GGML_PERF_CUSTOM_FUNC_START(ith, GGML_PERF_FUNC_MUL_MAT_ID_ONE_CHUNK);
 * ggml_compute_forward_mul_mat_id_one_chunk(...);
 * GGML_PERF_CUSTOM_FUNC_END(ith, GGML_PERF_FUNC_MUL_MAT_ID_ONE_CHUNK);
 * ```
 * 
 * 性能指标解释:
 * - mul_mat_one_chunk: 标准矩阵乘法chunk函数的性能
 * - mul_mat_id_one_chunk: 专家混合(MoE)矩阵乘法chunk函数的性能
 * - 平均时间(us): 单次函数调用的平均微秒数
 * - 最小/最大时间: 帮助识别性能变化和瓶颈
 * - 调用次数: 显示函数被调用的频率
 * - 线程分布: 显示每个线程的工作负载
 */ 