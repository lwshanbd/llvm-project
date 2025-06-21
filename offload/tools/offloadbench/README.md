# LLVM Offload Device Bandwidth Benchmark Tool

A comprehensive benchmark utility for measuring memory bandwidth and latency between host and offload devices in LLVM/OpenMP environments. This tool provides detailed performance analysis for device selection, data placement optimization, and runtime performance tuning.

## Features

### Core Benchmarking
- **Host-to-Device (H2D)** transfer performance measurement
- **Device-to-Host (D2H)** transfer performance measurement  
- **Device-to-Device (D2D)** transfer performance measurement
- Multi-threaded benchmarking support
- Configurable data sizes and iteration counts

### Advanced Testing Modes
- **NUMA-Aware Testing**: Measures performance from specific NUMA nodes to devices
- **Matrix Mode**: Comprehensive testing of all device pairs
- **Multi-Device Support**: Specify individual devices or device ranges

### Output Formats
- **Human-Readable**: Formatted output with units and descriptions
- **CSV Format**: Pure CSV data for scripting and analysis
- **Matrix-Only**: Clean matrix output without headers
- **C/C++ Header Generation**: Generate header files with performance data

### Device Support
- Automatic device detection and enumeration
- Support for multiple OpenMP offload targets
- Cross-platform compatibility (Linux with enhanced NUMA support)

## Building

### Prerequisites
- LLVM/Clang compiler with OpenMP support
- CMake build system
- libnuma library (Linux only, for NUMA-aware features)
- libomptarget library

### Build Instructions
```bash
# From LLVM project root
mkdir build && cd build
cmake -DLLVM_ENABLE_PROJECTS="clang;openmp;offload" ..
make llvm-offload-device-bench
```

The binary will be generated at: `build/bin/llvm-offload-device-bench`

## Usage

### Basic Usage
```bash
# Run with default settings (all tests, all devices)
./llvm-offload-device-bench

# Test specific devices
./llvm-offload-device-bench -d 0,1,2

# Test with larger data size
./llvm-offload-device-bench -s 512  # 512 MB
```

### Advanced Usage
```bash
# NUMA-aware testing
./llvm-offload-device-bench --numa-aware --numa-nodes 0,1

# Generate performance header file
./llvm-offload-device-bench -H performance.h --header-format cpp-constexpr

# CSV output for analysis
./llvm-offload-device-bench --csv --matrix-only > results.csv
```

## Command Line Options

### Basic Options
- `-s, --size SIZE`: Data size in MB (default: 128)
- `-d, --device ID`: Device ID(s) to test (default: all)
  - Examples: `-d 0`, `-d 0,1,2`, `-d 0-3`
- `-t, --threads NUM`: Number of threads (default: 1)
- `-i, --iterations NUM`: Number of iterations (default: 10)
- `-m, --mode MODE`: Test mode - h2d, d2h, d2d, all (default: all)
- `-h, --help`: Show help message
- `-I, --info`: Show device information

### Output Control
- `-u, --unit UNIT`: Output unit - auto, GB/s, MB/s, KB/s (default: auto)
- `-r, --raw`: Raw output (disable human-readable formatting)
- `--matrix-only`: Show only matrix data, no descriptions
- `--csv`: Output in pure CSV format

### NUMA Options
- `--numa-aware`: Enable NUMA-aware H2D/D2H testing
- `--numa-nodes NODES`: Specify NUMA nodes to test
  - Examples: `--numa-nodes 0`, `--numa-nodes 0,1,2`, `--numa-nodes 0-3`

### Header Generation
- `-H, --generate-header FILE`: Generate C/C++ header file
- `--header-namespace NAME`: Header namespace/prefix (default: LLVM_OFFLOAD)
- `--header-format FORMAT`: Header format - c-array, cpp-constexpr, macros (default: c-array)

## Test Modes

### Host-to-Device (H2D)
Measures bandwidth and latency when transferring data from host memory to device memory.

### Device-to-Host (D2H)  
Measures bandwidth and latency when transferring data from device memory to host memory.

### Device-to-Device (D2D)
Measures bandwidth and latency for direct device-to-device memory transfers. Automatically generates a full matrix showing performance between all device pairs.

### NUMA-Aware Testing
When enabled with `--numa-aware`, the tool measures:
- **NUMA-to-Device**: Performance from each NUMA node to each device
- **Device-to-NUMA**: Performance from each device to each NUMA node

## Output Examples

### Standard Output
```
=== Benchmark Configuration ===
Data size: 128 MB
Devices: 0,1,2,3
Threads: 1
Iterations: 10
Tests: H2D D2H D2D

=== Device to Device Transfer Matrix ===
     Dev0    Dev1    Dev2    Dev3
Dev0  450.50  125.30   98.90  112.20
Dev1  124.10  465.80   94.40  105.50
Dev2   97.20   95.60  440.20  108.90
Dev3  110.30  104.70  107.80  455.60
```

### CSV Output
```bash
./llvm-offload-device-bench --csv --matrix-only
```
```
450.50,125.30,98.90,112.20
124.10,465.80,94.40,105.50
97.20,95.60,440.20,108.90
110.30,104.70,107.80,455.60
```

### Generated Header File
```cpp
#ifndef LLVM_OFFLOAD_DEVICE_PERFORMANCE_H
#define LLVM_OFFLOAD_DEVICE_PERFORMANCE_H

/*
 * LLVM Offload Device Performance Data
 * Generated on: Mon Dec 16 10:30:45 2024
 * Test configuration:
 *   Data size: 128 MB
 *   Devices: 0,1,2,3
 *   Iterations: 10
 *   Threads: 1
 */

// Basic device information
#define LLVM_OFFLOAD_DEVICE_COUNT 4
#define LLVM_OFFLOAD_NUMA_NODE_COUNT 1

// Performance matrices
static const double LLVM_OFFLOAD_DEVICE_TO_DEVICE_BANDWIDTH_GBPS[4][4] = {
  {450.50, 125.30, 98.90, 112.20},
  {124.10, 465.80, 94.40, 105.50},
  {97.20, 95.60, 440.20, 108.90},
  {110.30, 104.70, 107.80, 455.60}
};

static const double LLVM_OFFLOAD_DEVICE_TO_DEVICE_LATENCY_US[4][4] = {
  {2.15, 8.45, 10.20, 9.80},
  {8.30, 2.10, 10.50, 9.90},
  {10.40, 10.60, 2.25, 9.70},
  {9.85, 9.95, 9.75, 2.20}
};

#endif // LLVM_OFFLOAD_DEVICE_PERFORMANCE_H
```

## Header File Formats

### c-array (default)
```cpp
static const double ARRAY_NAME[rows][cols] = { /* data */ };
```

### cpp-constexpr
```cpp
static constexpr double ARRAY_NAME[rows][cols] = { /* data */ };
```

### macros
```cpp
#define ARRAY_NAME_ROWS 4
#define ARRAY_NAME_COLS 4
#define ARRAY_NAME_0_0 450.50
#define ARRAY_NAME_0_1 125.30
// ... individual macros for each value
```

## Use Cases

### Compiler Integration
- LLVM device selection optimization
- Compile-time performance-aware transformations
- Static data placement decisions

### Runtime Optimization
- Dynamic device selection based on data size and transfer patterns
- NUMA-aware memory allocation strategies
- Load balancing across multiple devices

### Performance Analysis
- Identifying optimal device configurations
- Understanding memory hierarchy performance
- Benchmarking different hardware setups

### Build System Integration
- Automated performance tuning during build
- Configuration-dependent optimization flags
- Cross-platform performance characterization

## Examples

### Basic Performance Testing
```bash
# Test all devices with default settings
./llvm-offload-device-bench

# Test specific devices with larger data
./llvm-offload-device-bench -d 0,1 -s 256

# High-precision testing
./llvm-offload-device-bench -i 50 -t 4
```

### NUMA-Aware Analysis
```bash
# Enable NUMA testing with automatic node detection
./llvm-offload-device-bench --numa-aware

# Test specific NUMA nodes
./llvm-offload-device-bench --numa-aware --numa-nodes 0,1 -d 0-3

# Generate NUMA-aware header file
./llvm-offload-device-bench --numa-aware -H numa_perf.h
```

### Integration Examples
```bash
# Generate header for compiler integration
./llvm-offload-device-bench -H perf_data.hpp \
  --header-namespace CUDA_PERF \
  --header-format cpp-constexpr

# CSV output for analysis scripts
./llvm-offload-device-bench --csv --matrix-only > device_matrix.csv

# Combined testing with multiple outputs
./llvm-offload-device-bench --numa-aware --csv -H results.h \
  --header-format macros --matrix-only
```

### Scripting Integration
```bash
#!/bin/bash
# Automated performance characterization
for size in 64 128 256 512; do
  ./llvm-offload-device-bench -s $size --csv --matrix-only \
    > results_${size}MB.csv
done

# Generate optimized configuration header
./llvm-offload-device-bench -s 1024 -i 20 \
  -H optimal_config.h --header-format cpp-constexpr
```

### Performance Considerations

- **Data Size**: Larger data sizes generally provide more accurate bandwidth measurements
- **Iterations**: More iterations improve statistical accuracy but increase test time
- **System Load**: Run benchmarks on idle systems for consistent results
- **Thermal Effects**: Long-running tests may be affected by thermal throttling
