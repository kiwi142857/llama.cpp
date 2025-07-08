// 编译时启用性能监控
#define GGML_CPU_PERF_MONITOR 1

#include "ggml.h"
#include "ggml-cpu.h"  
#include "perf-monitor.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 1. 初始化 GGML 和性能监控
    ggml_cpu_init();
    ggml_perf_monitor_init();
    
    // 2. 启用性能监控
    ggml_perf_monitor_enable(true);
    
    // 3. 创建简单的计算图进行测试
    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,  // 128 MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "无法初始化 GGML 上下文\n");
        return 1;
    }
    
    // 创建一些张量和操作
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 1000);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 1000);
    
    // 初始化数据
    for (int i = 0; i < 1000 * 1000; i++) {
        ((float*)a->data)[i] = 1.0f + i * 0.001f;
        ((float*)b->data)[i] = 0.5f + i * 0.0001f;
    }
    
    // 执行一些计算操作
    printf("开始执行计算...\n");
    
    // 加法操作
    struct ggml_tensor * c = ggml_add(ctx, a, b);
    
    // 乘法操作
    struct ggml_tensor * d = ggml_mul(ctx, a, b);
    
    // 矩阵乘法操作
    struct ggml_tensor * e = ggml_mul_mat(ctx, a, b);
    
    // 标准化操作
    struct ggml_tensor * f = ggml_rms_norm(ctx, e, 1e-6f);
    
    // 构建计算图
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, f);
    
    // 执行计算图
    int n_threads = 4;  // 使用4个线程
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);
    
    // 重复执行几次来收集足够的统计数据
    printf("开始性能测试（执行10次迭代）...\n");
    for (int iter = 0; iter < 10; iter++) {
        printf("迭代 %d/10\n", iter + 1);
        enum ggml_status status = ggml_graph_compute(cgraph, &cplan);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "计算失败，状态: %d\n", status);
            break;
        }
    }
    
    // 4. 输出性能统计结果
    printf("\n=== 性能统计结果 ===\n");
    
    // 简要统计
    ggml_perf_monitor_print_summary();
    
    // 详细统计
    ggml_perf_monitor_print_detailed();
    
    // 导出CSV格式数据
    ggml_perf_monitor_export_csv("cpu_perf_stats.csv");
    
    // 导出JSON格式数据
    ggml_perf_monitor_export_json("cpu_perf_stats.json");
    
    // 5. 清理资源
    ggml_free(ctx);
    ggml_perf_monitor_free();
    
    printf("性能监控测试完成！\n");
    printf("性能数据已保存到 cpu_perf_stats.csv 和 cpu_perf_stats.json\n");
    
    return 0;
}

// 辅助函数：创建性能监控的热图可视化
void create_perf_heatmap(const char* output_file) {
    FILE* fp = fopen(output_file, "w");
    if (!fp) return;
    
    // 生成Python脚本来创建热图
    fprintf(fp, "#!/usr/bin/env python3\n");
    fprintf(fp, "import pandas as pd\n");
    fprintf(fp, "import matplotlib.pyplot as plt\n");
    fprintf(fp, "import seaborn as sns\n");
    fprintf(fp, "import numpy as np\n");
    fprintf(fp, "\n");
    fprintf(fp, "# 读取性能数据\n");
    fprintf(fp, "df = pd.read_csv('cpu_perf_stats.csv')\n");
    fprintf(fp, "\n");
    fprintf(fp, "# 创建热图数据\n");
    fprintf(fp, "heatmap_data = df.pivot_table(index='线程ID', columns='操作类型', values='平均时间(us)', fill_value=0)\n");
    fprintf(fp, "\n");
    fprintf(fp, "# 绘制热图\n");
    fprintf(fp, "plt.figure(figsize=(15, 8))\n");
    fprintf(fp, "sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd')\n");
    fprintf(fp, "plt.title('CPU 操作性能热图 (微秒)')\n");
    fprintf(fp, "plt.xlabel('操作类型')\n");
    fprintf(fp, "plt.ylabel('线程ID')\n");
    fprintf(fp, "plt.tight_layout()\n");
    fprintf(fp, "plt.savefig('cpu_perf_heatmap.png', dpi=300)\n");
    fprintf(fp, "plt.show()\n");
    fprintf(fp, "\n");
    fprintf(fp, "# 创建操作总时间条形图\n");
    fprintf(fp, "plt.figure(figsize=(12, 6))\n");
    fprintf(fp, "op_totals = df.groupby('操作类型')['总时间(ms)'].sum().sort_values(ascending=False)\n");
    fprintf(fp, "op_totals.plot(kind='bar')\n");
    fprintf(fp, "plt.title('各操作类型总执行时间')\n");
    fprintf(fp, "plt.xlabel('操作类型')\n");
    fprintf(fp, "plt.ylabel('总时间 (ms)')\n");
    fprintf(fp, "plt.xticks(rotation=45)\n");
    fprintf(fp, "plt.tight_layout()\n");
    fprintf(fp, "plt.savefig('cpu_op_totals.png', dpi=300)\n");
    fprintf(fp, "plt.show()\n");
    
    fclose(fp);
    printf("性能可视化脚本已生成: %s\n", output_file);
    printf("运行 'python3 %s' 来生成性能图表\n", output_file);
} 