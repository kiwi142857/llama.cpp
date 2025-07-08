set(TARGET llama-cli)

# 添加性能监控源文件
set(PERF_MONITOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-cpu/perf-monitor.c
)

add_executable(${TARGET} main.cpp ${PERF_MONITOR_SOURCES})
install(TARGETS ${TARGET} RUNTIME)

# 添加性能监控编译标志
target_compile_definitions(${TARGET} PRIVATE GGML_CPU_PERF_MONITOR=1)

# 添加包含路径
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-cpu)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include)

target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
