# GGML CPU 性能监控系统

## 📊 概述

这个性能监控系统为 GGML CPU 后端提供了细粒度的性能分析能力，可以统计每个线程执行每种操作类型的耗时，帮助您：

- 🔍 识别性能瓶颈
- ⚡ 优化多线程负载均衡
- 📈 分析操作执行效率
- 🎯 指导性能调优

## 🗂️ 文件结构

```
ggml/src/ggml-cpu/
├── perf-monitor.h          # 性能监控头文件
├── perf-monitor.c          # 性能监控实现
├── perf-example.c          # 使用示例
├── Makefile.perf           # 编译配置
└── README-perf.md          # 说明文档
```

## 🚀 快速开始

### 1. 编译时启用性能监控

```c
// 在编译时定义宏
#define GGML_CPU_PERF_MONITOR 1

// 或者在编译命令中添加
gcc -DGGML_CPU_PERF_MONITOR=1 ...
```

### 2. 基本使用

```c
#include "perf-monitor.h"

int main() {
    // 初始化性能监控
    ggml_perf_monitor_init();
    
    // 启用监控
    ggml_perf_monitor_enable(true);
    
    // 执行您的GGML计算
    // ... 计算代码 ...
    
    // 输出性能统计
    ggml_perf_monitor_print_summary();
    
    // 导出数据
    ggml_perf_monitor_export_csv("perf_stats.csv");
    
    // 清理
    ggml_perf_monitor_free();
    return 0;
}
```

## 🔧 API 参考

### 初始化和控制

```c
// 初始化性能监控器
void ggml_perf_monitor_init(void);

// 启用/禁用监控
void ggml_perf_monitor_enable(bool enable);

// 重置统计数据
void ggml_perf_monitor_reset(void);

// 释放资源
void ggml_perf_monitor_free(void);
```

### 输出和导出

```c
// 打印简要统计
void ggml_perf_monitor_print_summary(void);

// 打印详细统计
void ggml_perf_monitor_print_detailed(void);

// 导出CSV格式
void ggml_perf_monitor_export_csv(const char* filename);

// 导出JSON格式
void ggml_perf_monitor_export_json(const char* filename);
```

### 手动计时（高级用法）

```c
// 开始计时
void ggml_perf_op_start(int thread_id, enum ggml_op op_type);

// 结束计时
void ggml_perf_op_end(int thread_id, enum ggml_op op_type);

// 便利宏
#define GGML_PERF_OP_START(params, op)
#define GGML_PERF_OP_END(params, op)
```

## 📈 数据格式

### CSV 输出格式

```csv
线程ID,操作类型,总时间(ms),调用次数,平均时间(us),最小时间(us),最大时间(us)
0,ADD,12.34,1000,12.34,8.2,45.6
1,MUL_MAT,156.78,50,3135.6,2890.1,3456.2
```

### JSON 输出格式

```json
{
  "monitor_duration_us": 1500000,
  "threads": [
    {
      "thread_id": 0,
      "total_compute_time_us": 123456,
      "operations": [
        {
          "op_type": "ADD",
          "total_time_us": 12340,
          "count": 1000,
          "avg_time_us": 12.34,
          "min_time_us": 8,
          "max_time_us": 45
        }
      ]
    }
  ]
}
```

## 🛠️ 编译和运行

### 使用提供的 Makefile

```bash
# 编译示例
make -f Makefile.perf all

# 运行示例
make -f Makefile.perf run

# 生成性能报告
make -f Makefile.perf report

# 运行基准测试
make -f Makefile.perf benchmark

# 清理
make -f Makefile.perf clean
```

### 手动编译

```bash
# 编译性能监控库
gcc -DGGML_CPU_PERF_MONITOR=1 -c perf-monitor.c -o perf-monitor.o

# 编译示例程序
gcc -DGGML_CPU_PERF_MONITOR=1 -o perf-example perf-example.c perf-monitor.o -lm -pthread
```

## 📊 性能分析示例

### 控制台输出示例

```
=== CPU 性能监控摘要 ===
监控总时间: 1500.00 ms
活跃线程数: 4

各线程计算时间:
线程ID | 总计算时间(ms) | 利用率(%)
-------|---------------|----------
     0 |        450.23 |     30.0
     1 |        425.67 |     28.4
     2 |        398.45 |     26.6
     3 |        375.89 |     25.1

热点操作类型 (所有线程汇总):
操作类型 | 总时间(ms) | 调用次数 | 平均时间(us)
---------|-----------|----------|-------------
 MUL_MAT |     856.3 |       40 |      21407.5
 RMS_NORM|     234.5 |      120 |       1954.2
     ADD |     156.8 |      800 |        196.0
     MUL |      89.2 |      400 |        223.0
```

### 详细线程统计

```
=== CPU 性能监控详细报告 ===

--- 线程 0 ---
总计算时间: 450.23 ms

操作详情:
操作类型 | 总时间(ms) | 调用次数 | 平均(us) | 最小(us) | 最大(us)
---------|-----------|----------|----------|----------|----------
 MUL_MAT |     215.6 |       10 |  21560.0 |  18234.5 |  25678.3
 RMS_NORM|      58.7 |       30 |   1956.7 |   1234.5 |   2678.9
     ADD |      39.2 |      200 |    196.0 |    145.2 |    278.4
```

## 🎯 性能优化建议

### 1. 识别热点操作

```python
# 分析CSV数据
import pandas as pd
df = pd.read_csv('cpu_perf_stats.csv')

# 找出最耗时的操作
hot_ops = df.groupby('操作类型')['总时间(ms)'].sum().sort_values(ascending=False)
print("热点操作:", hot_ops.head(5))
```

### 2. 线程负载均衡分析

```python
# 检查线程负载分布
thread_loads = df.groupby('线程ID')['总时间(ms)'].sum()
load_variance = thread_loads.var()
print(f"负载方差: {load_variance:.2f}")

# 负载方差越小，负载越均衡
```

### 3. 操作效率分析

```python
# 分析每种操作的效率
efficiency = df.groupby('操作类型')['平均时间(us)'].mean()
print("操作效率排名:")
print(efficiency.sort_values())
```

## 🔍 故障排除

### 常见问题

1. **编译错误: 'GGML_CPU_PERF_MONITOR' 未定义**
   - 解决: 确保在编译时定义了 `GGML_CPU_PERF_MONITOR=1`

2. **运行时无输出**
   - 解决: 检查是否调用了 `ggml_perf_monitor_enable(true)`

3. **线程安全问题**
   - 解决: 确保每个线程使用正确的线程ID

4. **内存泄漏**
   - 解决: 确保调用 `ggml_perf_monitor_free()` 清理资源

### 调试技巧

```c
// 添加调试输出
#if GGML_CPU_PERF_MONITOR
    printf("性能监控已启用，线程ID: %d\n", params->ith);
#endif
```

## 📚 高级用法

### 自定义性能分析

```c
// 自定义操作计时
void my_custom_operation(struct ggml_compute_params * params) {
    GGML_PERF_OP_START(params, GGML_OP_ADD);  // 使用现有操作类型
    
    // 执行自定义操作
    // ...
    
    GGML_PERF_OP_END(params, GGML_OP_ADD);
}
```

### 条件性能监控

```c
// 只在调试模式下启用
#ifdef DEBUG
    ggml_perf_monitor_enable(true);
#endif
```

### 性能数据处理

```c
// 导出后处理数据
ggml_perf_monitor_export_json("raw_perf.json");

// 然后用Python或其他工具分析
// python analyze_perf.py raw_perf.json
```

## ⚡ 性能开销

性能监控系统的开销极小：

- **内存开销**: 每个线程约 8KB
- **CPU开销**: 每次操作约 0.1-0.5 微秒
- **编译开销**: 可通过宏完全禁用

## 🤝 贡献指南

1. 添加新的操作类型支持
2. 优化性能统计算法
3. 增加可视化功能
4. 完善文档和示例

## 📞 联系和支持

如果您在使用过程中遇到问题或有改进建议，欢迎：

- 提交 Issue
- 发送 Pull Request
- 参与讨论

---

**注意**: 性能监控功能默认关闭，需要在编译时显式启用。在生产环境中，建议禁用监控以获得最佳性能。 