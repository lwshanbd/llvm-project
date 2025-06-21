//===- llvm-offload-device-bench.cpp - Device info as seen by LLVM/Offload -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that, by using LLVM/Offload, and the device
// plugins, evaluate the bandwidth and latency of the device to host memory.
//
//===----------------------------------------------------------------------===//

#include "omptarget.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#ifdef __linux__
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#include <numaif.h>
#endif
#include <fstream>
#include <ctime>

using namespace std;

// NUMA helper functions (declared before DeviceBenchmark class)
#ifdef __linux__
vector<int> getAvailableNumaNodes() {
  vector<int> NumaNodes;
  if (numa_available() == -1) {
    return NumaNodes; // NUMA not available
  }
  
  int MaxNode = numa_max_node();
  for (int I = 0; I <= MaxNode; I++) {
    if (numa_bitmask_isbitset(numa_get_mems_allowed(), I)) {
      NumaNodes.push_back(I);
    }
  }
  return NumaNodes;
}

bool bindToNumaNode(int NumaNode) {
  if (numa_available() == -1) {
    return false;
  }
  
  struct bitmask *NodeMask = numa_allocate_nodemask();
  numa_bitmask_setbit(NodeMask, NumaNode);
  
  numa_bind(NodeMask);
  numa_free_nodemask(NodeMask);
  
  return true;
}

bool bindThreadToNumaNode(int NumaNode) {
  if (numa_available() == -1) {
    return false;
  }
  
  // Use numa_set_preferred to prefer allocation on the specified node
  // This is a soft binding that should work across libnuma versions
  numa_set_preferred(NumaNode);
  
  return true;
}
#else
vector<int> getAvailableNumaNodes() {
  return vector<int>{0}; // Assume single NUMA node on non-Linux systems
}

bool bindToNumaNode(int NumaNode) {
  return true; // No-op on non-Linux systems
}

bool bindThreadToNumaNode(int NumaNode) {
  return true; // No-op on non-Linux systems
}
#endif

struct BenchmarkConfiguration {
  size_t DataSize = 128 * 1024 * 1024; // 128MB default
  vector<int> DeviceIds; // Changed from single DeviceId to list
  int NumberOfThreads = 1;
  int NumberOfIterations = 10;
  bool TestHostToDevice = true;
  bool TestDeviceToHost = true;
  bool TestDeviceToDevice = true; // Changed default to true
  bool HumanReadable = true;
  string OutputUnit = "auto";
  bool ShowHelp = false;
  bool ShowDeviceInfo = false;
  bool TestAllDevices = false; // New: test all available devices if no specific devices specified
  bool NumaAware = false; // New: NUMA-aware testing
  bool MatrixOnly = false; // New: matrix-only output
  bool CsvOutput = false; // New: CSV format output
  vector<int> NumaNodes; // New: specific NUMA nodes to test
  
  // Header generation options
  string HeaderFilePath = ""; // New: path to generate header file
  string HeaderNamespace = "LLVM_OFFLOAD"; // New: namespace/prefix for generated code
  string HeaderFormat = "c-array"; // New: output format (c-array, cpp-constexpr, macros)
};

struct BenchmarkResult {
  double BandwidthGigabytesPerSecond;
  double LatencyMicroseconds;
  double MinimumTime;
  double MaximumTime;
  double AverageTime;
  double StandardDeviation;
  vector<double> Times;
};

// Structure to store all benchmark results for header generation
struct AllBenchmarkResults {
  vector<vector<double>> DeviceToDeviceBandwidth;
  vector<vector<double>> DeviceToDeviceLatency;
  vector<vector<double>> HostToDeviceBandwidth; // NUMA to Device
  vector<vector<double>> DeviceToHostBandwidth; // Device to NUMA
  vector<vector<double>> HostToDeviceLatency; // NUMA to Device
  vector<vector<double>> DeviceToHostLatency; // Device to NUMA
  vector<int> DeviceIds;
  vector<int> NumaNodes;
  int DeviceCount = 0;
  int NumaNodeCount = 0;
  
  AllBenchmarkResults() = default;
};

class DeviceBenchmark {
private:
  BenchmarkConfiguration Configuration;
  int NumberOfDevices;
  AllBenchmarkResults Results; // Store all benchmark results for header generation

public:
  DeviceBenchmark(BenchmarkConfiguration &Config) : Configuration(Config) {
    __tgt_bin_desc EmptyDesc = {0, nullptr, nullptr, nullptr};
    __tgt_register_lib(&EmptyDesc);
    __tgt_init_all_rtls();
    NumberOfDevices = omp_get_num_devices();
    
    // Set default devices if none specified
    if (Configuration.DeviceIds.empty()) {
      Configuration.TestAllDevices = true;
      for (int I = 0; I < NumberOfDevices; I++) {
        Configuration.DeviceIds.push_back(I);
      }
    }
    
    // Set default NUMA nodes if NUMA-aware testing is enabled but no nodes specified
    if (Configuration.NumaAware && Configuration.NumaNodes.empty()) {
      Configuration.NumaNodes = getAvailableNumaNodes();
    }
  }

  void printDeviceInfo() {
    if (Configuration.MatrixOnly || Configuration.CsvOutput) return;
    
    cout << "=== Device Information ===" << endl;
    cout << "Number of devices: " << NumberOfDevices << endl;
    for (int I = 0; I < NumberOfDevices; I++) {
      cout << "\nDevice " << I << ":" << endl;
      __tgt_print_device_info(I);
    }
    cout << endl;
  }

  bool validateConfiguration() {
    if (Configuration.DeviceIds.empty()) {
      cerr << "Error: No devices specified" << endl;
      return false;
    }

    for (int DeviceId : Configuration.DeviceIds) {
      if (DeviceId >= NumberOfDevices || DeviceId < 0) {
        cerr << "Error: Invalid device ID " << DeviceId << ". Available devices: 0-" << (NumberOfDevices - 1) << endl;
        return false;
      }
    }

    if (Configuration.DataSize == 0) {
      cerr << "Error: Data size must be greater than 0" << endl;
      return false;
    }

    if (Configuration.NumberOfThreads <= 0) {
      cerr << "Error: Number of threads must be greater than 0" << endl;
      return false;
    }

    return true;
  }

  double calculateStandardDeviation(const vector<double> &Times, double Mean) {
    double SumSquaredDifference = 0.0;
    for (double Time : Times) {
      SumSquaredDifference += (Time - Mean) * (Time - Mean);
    }
    return sqrt(SumSquaredDifference / Times.size());
  }

  void formatBandwidth(double BandwidthGigabytesPerSecond, ostream &OutputStream) {
    if (Configuration.HumanReadable && Configuration.OutputUnit == "auto") {
      if (BandwidthGigabytesPerSecond >= 1.0) {
        OutputStream << fixed << setprecision(2) << BandwidthGigabytesPerSecond << " GB/s";
      } else if (BandwidthGigabytesPerSecond >= 0.001) {
        OutputStream << fixed << setprecision(2) << (BandwidthGigabytesPerSecond * 1000) << " MB/s";
      } else {
        OutputStream << fixed << setprecision(2) << (BandwidthGigabytesPerSecond * 1000000) << " KB/s";
      }
    } else if (Configuration.OutputUnit == "MB/s") {
      OutputStream << fixed << setprecision(2) << (BandwidthGigabytesPerSecond * 1000) << " MB/s";
    } else if (Configuration.OutputUnit == "KB/s") {
      OutputStream << fixed << setprecision(2) << (BandwidthGigabytesPerSecond * 1000000) << " KB/s";
    } else {
      OutputStream << fixed << setprecision(2) << BandwidthGigabytesPerSecond << " GB/s";
    }
  }

  BenchmarkResult benchmarkHostToDevice(int ThreadId, size_t ThreadDataSize) {
    BenchmarkResult Result;
    vector<double> Times;

    unique_ptr<char[]> HostData(new char[ThreadDataSize]);
    
    // Initialize with random data
    random_device RandomDevice;
    mt19937 Generator(RandomDevice() + ThreadId);
    uniform_int_distribution<> Distribution(0, 255);
    for (size_t I = 0; I < ThreadDataSize; I++) {
      HostData[I] = static_cast<char>(Distribution(Generator));
    }

    // Use first device if only testing single device type operations
    int DeviceId = Configuration.DeviceIds.empty() ? 0 : Configuration.DeviceIds[0];
    void *DevicePointer = omp_target_alloc(ThreadDataSize, DeviceId);
    if (!DevicePointer) {
      cerr << "Failed to allocate device memory for thread " << ThreadId << endl;
      return Result;
    }

    // Warmup
    for (int I = 0; I < 3; I++) {
      omp_target_memcpy(DevicePointer, HostData.get(), ThreadDataSize, 0, 0,
                       DeviceId, omp_get_initial_device());
    }

    // Actual benchmark
    for (int I = 0; I < Configuration.NumberOfIterations; I++) {
      auto Start = chrono::high_resolution_clock::now();
      
      int ReturnValue = omp_target_memcpy(DevicePointer, HostData.get(), ThreadDataSize,
                                 0, 0, DeviceId, omp_get_initial_device());
      
      auto End = chrono::high_resolution_clock::now();
      
      if (ReturnValue != 0) {
        cerr << "Memory copy failed in iteration " << I << " for thread " 
             << ThreadId << endl;
        continue;
      }

      double TimeMilliseconds = chrono::duration<double, milli>(End - Start).count();
      Times.push_back(TimeMilliseconds);
    }

    omp_target_free(DevicePointer, DeviceId);

    if (!Times.empty()) {
      Result.Times = Times;
      Result.MinimumTime = *min_element(Times.begin(), Times.end());
      Result.MaximumTime = *max_element(Times.begin(), Times.end());
      Result.AverageTime = accumulate(Times.begin(), Times.end(), 0.0) / Times.size();
      Result.StandardDeviation = calculateStandardDeviation(Times, Result.AverageTime);
      
      double DataGigabytes = static_cast<double>(ThreadDataSize) / (1024 * 1024 * 1024);
      Result.BandwidthGigabytesPerSecond = DataGigabytes / (Result.AverageTime / 1000.0);
      Result.LatencyMicroseconds = Result.AverageTime * 1000.0;
    }

    return Result;
  }

  BenchmarkResult benchmarkDeviceToHost(int ThreadId, size_t ThreadDataSize) {
    BenchmarkResult Result;
    vector<double> Times;

    unique_ptr<char[]> HostData(new char[ThreadDataSize]);
    int DeviceId = Configuration.DeviceIds.empty() ? 0 : Configuration.DeviceIds[0];
    void *DevicePointer = omp_target_alloc(ThreadDataSize, DeviceId);
    
    if (!DevicePointer) {
      cerr << "Failed to allocate device memory for thread " << ThreadId << endl;
      return Result;
    }

    // Initialize device memory with some data
    memset(HostData.get(), 0xAA, ThreadDataSize);
    omp_target_memcpy(DevicePointer, HostData.get(), ThreadDataSize, 0, 0,
                     DeviceId, omp_get_initial_device());

    // Warmup
    for (int I = 0; I < 3; I++) {
      omp_target_memcpy(HostData.get(), DevicePointer, ThreadDataSize, 0, 0,
                       omp_get_initial_device(), DeviceId);
    }

    // Actual benchmark
    for (int I = 0; I < Configuration.NumberOfIterations; I++) {
      auto Start = chrono::high_resolution_clock::now();
      
      int ReturnValue = omp_target_memcpy(HostData.get(), DevicePointer, ThreadDataSize,
                                 0, 0, omp_get_initial_device(), DeviceId);
      
      auto End = chrono::high_resolution_clock::now();
      
      if (ReturnValue != 0) {
        cerr << "Memory copy failed in iteration " << I << " for thread " 
             << ThreadId << endl;
        continue;
      }

      double TimeMilliseconds = chrono::duration<double, milli>(End - Start).count();
      Times.push_back(TimeMilliseconds);
    }

    omp_target_free(DevicePointer, DeviceId);

    if (!Times.empty()) {
      Result.Times = Times;
      Result.MinimumTime = *min_element(Times.begin(), Times.end());
      Result.MaximumTime = *max_element(Times.begin(), Times.end());
      Result.AverageTime = accumulate(Times.begin(), Times.end(), 0.0) / Times.size();
      Result.StandardDeviation = calculateStandardDeviation(Times, Result.AverageTime);
      
      double DataGigabytes = static_cast<double>(ThreadDataSize) / (1024 * 1024 * 1024);
      Result.BandwidthGigabytesPerSecond = DataGigabytes / (Result.AverageTime / 1000.0);
      Result.LatencyMicroseconds = Result.AverageTime * 1000.0;
    }

    return Result;
  }

  // Modified to accept specific source and destination devices
  BenchmarkResult benchmarkDeviceToDeviceSpecific(int SourceDevice, int DestinationDevice, 
                                                  size_t ThreadDataSize) {
    BenchmarkResult Result;
    vector<double> Times;

    void *SourcePointer = omp_target_alloc(ThreadDataSize, SourceDevice);
    void *DestinationPointer = omp_target_alloc(ThreadDataSize, DestinationDevice);
    
    if (!SourcePointer || !DestinationPointer) {
      cerr << "Failed to allocate device memory for D2D test (" << SourceDevice 
           << " -> " << DestinationDevice << ")" << endl;
      if (SourcePointer) omp_target_free(SourcePointer, SourceDevice);
      if (DestinationPointer) omp_target_free(DestinationPointer, DestinationDevice);
      return Result;
    }

    // Initialize source device memory
    unique_ptr<char[]> TemporaryData(new char[ThreadDataSize]);
    memset(TemporaryData.get(), 0xBB, ThreadDataSize);
    omp_target_memcpy(SourcePointer, TemporaryData.get(), ThreadDataSize, 0, 0,
                     SourceDevice, omp_get_initial_device());

    // Warmup
    for (int I = 0; I < 3; I++) {
      omp_target_memcpy(DestinationPointer, SourcePointer, ThreadDataSize, 0, 0,
                       DestinationDevice, SourceDevice);
    }

    // Actual benchmark
    for (int I = 0; I < Configuration.NumberOfIterations; I++) {
      auto Start = chrono::high_resolution_clock::now();
      
      int ReturnValue = omp_target_memcpy(DestinationPointer, SourcePointer, ThreadDataSize, 0, 0,
                                 DestinationDevice, SourceDevice);
      
      auto End = chrono::high_resolution_clock::now();
      
      if (ReturnValue != 0) {
        cerr << "D2D memory copy failed in iteration " << I << " (" << SourceDevice
             << " -> " << DestinationDevice << ")" << endl;
        continue;
      }

      double TimeMilliseconds = chrono::duration<double, milli>(End - Start).count();
      Times.push_back(TimeMilliseconds);
    }

    omp_target_free(SourcePointer, SourceDevice);
    omp_target_free(DestinationPointer, DestinationDevice);

    if (!Times.empty()) {
      Result.Times = Times;
      Result.MinimumTime = *min_element(Times.begin(), Times.end());
      Result.MaximumTime = *max_element(Times.begin(), Times.end());
      Result.AverageTime = accumulate(Times.begin(), Times.end(), 0.0) / Times.size();
      Result.StandardDeviation = calculateStandardDeviation(Times, Result.AverageTime);
      
      double DataGigabytes = static_cast<double>(ThreadDataSize) / (1024 * 1024 * 1024);
      Result.BandwidthGigabytesPerSecond = DataGigabytes / (Result.AverageTime / 1000.0);
      Result.LatencyMicroseconds = Result.AverageTime * 1000.0;
    }

    return Result;
  }

  BenchmarkResult benchmarkDeviceToDevice(int ThreadId, size_t ThreadDataSize) {
    BenchmarkResult Result;
    vector<double> Times;

    if (NumberOfDevices < 2) {
      cerr << "Device-to-device test requires at least 2 devices" << endl;
      return Result;
    }

    int SourceDevice = Configuration.DeviceIds.empty() ? 0 : Configuration.DeviceIds[0];
    int DestinationDevice = (SourceDevice + 1) % NumberOfDevices;

    return benchmarkDeviceToDeviceSpecific(SourceDevice, DestinationDevice, ThreadDataSize);
  }

  // New function for comprehensive D2D matrix testing
  void runDeviceToDeviceMatrix() {
    if (NumberOfDevices < 2) {
      if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
        cout << "=== Device to Device Matrix ===" << endl;
        cout << "Requires at least 2 devices (found " << NumberOfDevices << ")" << endl;
      }
      return;
    }

    vector<int> TestDevices = Configuration.DeviceIds;
    if (Configuration.TestAllDevices || TestDevices.empty()) {
      TestDevices.clear();
      for (int I = 0; I < NumberOfDevices; I++) {
        TestDevices.push_back(I);
      }
    }

    size_t TestDataSize = Configuration.DataSize / Configuration.NumberOfThreads;
    
    // Create matrices for results
    vector<vector<double>> BandwidthMatrix(TestDevices.size(), vector<double>(TestDevices.size(), 0.0));
    vector<vector<double>> LatencyMatrix(TestDevices.size(), vector<double>(TestDevices.size(), 0.0));
    
    if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
      cout << "=== Device to Device Transfer Matrix ===" << endl;
      cout << "Testing " << TestDevices.size() << " devices..." << endl;
    }
    
    for (size_t I = 0; I < TestDevices.size(); I++) {
      for (size_t J = 0; J < TestDevices.size(); J++) {
        int SourceDevice = TestDevices[I];
        int DestinationDevice = TestDevices[J];
        
        BenchmarkResult Result = benchmarkDeviceToDeviceSpecific(SourceDevice, DestinationDevice, TestDataSize);
        
        if (!Result.Times.empty()) {
          BandwidthMatrix[I][J] = Result.BandwidthGigabytesPerSecond;
          LatencyMatrix[I][J] = Result.LatencyMicroseconds;
        }
      }
    }
    
    // Store results for header generation
    Results.DeviceToDeviceBandwidth = BandwidthMatrix;
    Results.DeviceToDeviceLatency = LatencyMatrix;
    Results.DeviceIds = TestDevices;
    Results.DeviceCount = TestDevices.size();
    
    // Print bandwidth matrix
    printMatrixHeader("Device to Device Bandwidth Matrix", "GB/s", TestDevices);
    for (size_t I = 0; I < TestDevices.size(); I++) {
      printMatrixRow("Dev", TestDevices[I], BandwidthMatrix[I]);
    }
    
    if (!Configuration.CsvOutput) {
      // Print latency matrix
      printMatrixHeader("Device to Device Latency Matrix", "μs", TestDevices);
      for (size_t I = 0; I < TestDevices.size(); I++) {
        printMatrixRow("Dev", TestDevices[I], LatencyMatrix[I]);
      }
    }
    
    if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
      cout << endl;
    }
  }

  // NUMA-aware H2D matrix testing
  void runHostToDeviceNumaMatrix() {
    if (Configuration.NumaNodes.empty()) {
      return;
    }

    size_t TestDataSize = Configuration.DataSize / Configuration.NumberOfThreads;
    vector<int> &TestDevices = Configuration.DeviceIds;
    
    // Create matrices for results
    vector<vector<double>> BandwidthMatrix(Configuration.NumaNodes.size(), 
                                          vector<double>(TestDevices.size(), 0.0));
    vector<vector<double>> LatencyMatrix(Configuration.NumaNodes.size(), 
                                        vector<double>(TestDevices.size(), 0.0));
    
    if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
      cout << "=== NUMA to Device Transfer Matrix ===" << endl;
      cout << "Testing " << Configuration.NumaNodes.size() << " NUMA nodes to " 
           << TestDevices.size() << " devices..." << endl;
    }
    
    for (size_t I = 0; I < Configuration.NumaNodes.size(); I++) {
      for (size_t J = 0; J < TestDevices.size(); J++) {
        int NumaNode = Configuration.NumaNodes[I];
        int DeviceId = TestDevices[J];
        
        BenchmarkResult Result = benchmarkHostToDeviceNuma(NumaNode, DeviceId, TestDataSize);
        
        if (!Result.Times.empty()) {
          BandwidthMatrix[I][J] = Result.BandwidthGigabytesPerSecond;
          LatencyMatrix[I][J] = Result.LatencyMicroseconds;
        }
      }
    }
    
    // Store results for header generation
    Results.HostToDeviceBandwidth = BandwidthMatrix;
    Results.HostToDeviceLatency = LatencyMatrix;
    Results.NumaNodes = Configuration.NumaNodes;
    Results.NumaNodeCount = Configuration.NumaNodes.size();
    
    // Print bandwidth matrix
    printMatrixHeader("NUMA to Device Bandwidth Matrix", "GB/s", TestDevices);
    for (size_t I = 0; I < Configuration.NumaNodes.size(); I++) {
      printMatrixRow("N", Configuration.NumaNodes[I], BandwidthMatrix[I]);
    }
    
    if (!Configuration.CsvOutput) {
      // Print latency matrix
      printMatrixHeader("NUMA to Device Latency Matrix", "μs", TestDevices);
      for (size_t I = 0; I < Configuration.NumaNodes.size(); I++) {
        printMatrixRow("N", Configuration.NumaNodes[I], LatencyMatrix[I]);
      }
    }
    
    if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
      cout << endl;
    }
  }

  // NUMA-aware D2H matrix testing
  void runDeviceToHostNumaMatrix() {
    if (Configuration.NumaNodes.empty()) {
      return;
    }

    size_t TestDataSize = Configuration.DataSize / Configuration.NumberOfThreads;
    vector<int> &TestDevices = Configuration.DeviceIds;
    
    // Create matrices for results
    vector<vector<double>> BandwidthMatrix(TestDevices.size(), 
                                          vector<double>(Configuration.NumaNodes.size(), 0.0));
    vector<vector<double>> LatencyMatrix(TestDevices.size(), 
                                        vector<double>(Configuration.NumaNodes.size(), 0.0));
    
    if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
      cout << "=== Device to NUMA Transfer Matrix ===" << endl;
      cout << "Testing " << TestDevices.size() << " devices to " 
           << Configuration.NumaNodes.size() << " NUMA nodes..." << endl;
    }
    
    for (size_t I = 0; I < TestDevices.size(); I++) {
      for (size_t J = 0; J < Configuration.NumaNodes.size(); J++) {
        int DeviceId = TestDevices[I];
        int NumaNode = Configuration.NumaNodes[J];
        
        BenchmarkResult Result = benchmarkDeviceToHostNuma(NumaNode, DeviceId, TestDataSize);
        
        if (!Result.Times.empty()) {
          BandwidthMatrix[I][J] = Result.BandwidthGigabytesPerSecond;
          LatencyMatrix[I][J] = Result.LatencyMicroseconds;
        }
      }
    }
    
    // Store results for header generation
    Results.DeviceToHostBandwidth = BandwidthMatrix;
    Results.DeviceToHostLatency = LatencyMatrix;
    
    // Print bandwidth matrix
    printMatrixHeader("Device to NUMA Bandwidth Matrix", "GB/s", Configuration.NumaNodes, "N");
    for (size_t I = 0; I < TestDevices.size(); I++) {
      printMatrixRow("Dev", TestDevices[I], BandwidthMatrix[I]);
    }
    
    if (!Configuration.CsvOutput) {
      // Print latency matrix
      printMatrixHeader("Device to NUMA Latency Matrix", "μs", Configuration.NumaNodes, "N");
      for (size_t I = 0; I < TestDevices.size(); I++) {
        printMatrixRow("Dev", TestDevices[I], LatencyMatrix[I]);
      }
    }
    
    if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
      cout << endl;
    }
  }

  // NUMA-aware H2D benchmark
  BenchmarkResult benchmarkHostToDeviceNuma(int NumaNode, int DeviceId, size_t DataSize) {
    BenchmarkResult Result;
    vector<double> Times;

    // Bind to NUMA node
    if (!bindThreadToNumaNode(NumaNode)) {
      cerr << "Warning: Failed to bind to NUMA node " << NumaNode << endl;
    }

    unique_ptr<char[]> HostData(new char[DataSize]);
    
    // Initialize with random data
    random_device RandomDevice;
    mt19937 Generator(RandomDevice() + NumaNode * 1000 + DeviceId);
    uniform_int_distribution<> Distribution(0, 255);
    for (size_t I = 0; I < DataSize; I++) {
      HostData[I] = static_cast<char>(Distribution(Generator));
    }

    void *DevicePointer = omp_target_alloc(DataSize, DeviceId);
    if (!DevicePointer) {
      cerr << "Failed to allocate device memory (NUMA " << NumaNode << " -> Device " << DeviceId << ")" << endl;
      return Result;
    }

    // Warmup
    for (int I = 0; I < 3; I++) {
      omp_target_memcpy(DevicePointer, HostData.get(), DataSize, 0, 0,
                       DeviceId, omp_get_initial_device());
    }

    // Actual benchmark
    for (int I = 0; I < Configuration.NumberOfIterations; I++) {
      auto Start = chrono::high_resolution_clock::now();
      
      int ReturnValue = omp_target_memcpy(DevicePointer, HostData.get(), DataSize,
                                 0, 0, DeviceId, omp_get_initial_device());
      
      auto End = chrono::high_resolution_clock::now();
      
      if (ReturnValue != 0) {
        continue;
      }

      double TimeMilliseconds = chrono::duration<double, milli>(End - Start).count();
      Times.push_back(TimeMilliseconds);
    }

    omp_target_free(DevicePointer, DeviceId);

    if (!Times.empty()) {
      Result.Times = Times;
      Result.MinimumTime = *min_element(Times.begin(), Times.end());
      Result.MaximumTime = *max_element(Times.begin(), Times.end());
      Result.AverageTime = accumulate(Times.begin(), Times.end(), 0.0) / Times.size();
      Result.StandardDeviation = calculateStandardDeviation(Times, Result.AverageTime);
      
      double DataGigabytes = static_cast<double>(DataSize) / (1024 * 1024 * 1024);
      Result.BandwidthGigabytesPerSecond = DataGigabytes / (Result.AverageTime / 1000.0);
      Result.LatencyMicroseconds = Result.AverageTime * 1000.0;
    }

    return Result;
  }

  // NUMA-aware D2H benchmark
  BenchmarkResult benchmarkDeviceToHostNuma(int NumaNode, int DeviceId, size_t DataSize) {
    BenchmarkResult Result;
    vector<double> Times;

    // Bind to NUMA node
    if (!bindThreadToNumaNode(NumaNode)) {
      cerr << "Warning: Failed to bind to NUMA node " << NumaNode << endl;
    }

    unique_ptr<char[]> HostData(new char[DataSize]);
    void *DevicePointer = omp_target_alloc(DataSize, DeviceId);
    
    if (!DevicePointer) {
      cerr << "Failed to allocate device memory (Device " << DeviceId << " -> NUMA " << NumaNode << ")" << endl;
      return Result;
    }

    // Initialize device memory with some data
    memset(HostData.get(), 0xAA, DataSize);
    omp_target_memcpy(DevicePointer, HostData.get(), DataSize, 0, 0,
                     DeviceId, omp_get_initial_device());

    // Warmup
    for (int I = 0; I < 3; I++) {
      omp_target_memcpy(HostData.get(), DevicePointer, DataSize, 0, 0,
                       omp_get_initial_device(), DeviceId);
    }

    // Actual benchmark
    for (int I = 0; I < Configuration.NumberOfIterations; I++) {
      auto Start = chrono::high_resolution_clock::now();
      
      int ReturnValue = omp_target_memcpy(HostData.get(), DevicePointer, DataSize,
                                 0, 0, omp_get_initial_device(), DeviceId);
      
      auto End = chrono::high_resolution_clock::now();
      
      if (ReturnValue != 0) {
        continue;
      }

      double TimeMilliseconds = chrono::duration<double, milli>(End - Start).count();
      Times.push_back(TimeMilliseconds);
    }

    omp_target_free(DevicePointer, DeviceId);

    if (!Times.empty()) {
      Result.Times = Times;
      Result.MinimumTime = *min_element(Times.begin(), Times.end());
      Result.MaximumTime = *max_element(Times.begin(), Times.end());
      Result.AverageTime = accumulate(Times.begin(), Times.end(), 0.0) / Times.size();
      Result.StandardDeviation = calculateStandardDeviation(Times, Result.AverageTime);
      
      double DataGigabytes = static_cast<double>(DataSize) / (1024 * 1024 * 1024);
      Result.BandwidthGigabytesPerSecond = DataGigabytes / (Result.AverageTime / 1000.0);
      Result.LatencyMicroseconds = Result.AverageTime * 1000.0;
    }

    return Result;
  }

  // Output helper functions
  void printMatrixHeader(const string &Title, const string &Unit, const vector<int> &ColIds, const string &ColPrefix = "Dev") {
    if (Configuration.CsvOutput) return;
    
    if (!Configuration.MatrixOnly) {
      cout << "\n=== " << Title << " (" << Unit << ") ===" << endl;
    }
    
    cout << "     ";
    for (int Id : ColIds) {
      cout << setw(8) << (ColPrefix + to_string(Id));
    }
    cout << endl;
  }

  void printMatrixRow(const string &RowPrefix, int RowId, const vector<double> &Values) {
    if (Configuration.CsvOutput) {
      for (size_t I = 0; I < Values.size(); I++) {
        if (Values[I] > 0) {
          cout << fixed << setprecision(2) << Values[I];
        }
        if (I < Values.size() - 1) cout << ",";
      }
      cout << endl;
    } else {
      cout << RowPrefix << setw(2) << RowId;
      for (double Value : Values) {
        if (Value > 0) {
          cout << setw(8) << fixed << setprecision(2) << Value;
        } else {
          cout << setw(8) << "N/A";
        }
      }
      cout << endl;
    }
  }

  void runBenchmark() {
    if (Configuration.ShowDeviceInfo) {
      printDeviceInfo();
    }

    if (!validateConfiguration()) {
      return;
    }

    // Show configuration only if not in matrix-only or CSV mode
    if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
      cout << "=== Benchmark Configuration ===" << endl;
      cout << "Data size: " << (Configuration.DataSize / (1024 * 1024)) << " MB" << endl;
      cout << "Devices: ";
      for (size_t I = 0; I < Configuration.DeviceIds.size(); I++) {
        cout << Configuration.DeviceIds[I];
        if (I < Configuration.DeviceIds.size() - 1) cout << ",";
      }
      cout << endl;
      cout << "Threads: " << Configuration.NumberOfThreads << endl;
      cout << "Iterations: " << Configuration.NumberOfIterations << endl;
      cout << "Tests: ";
      if (Configuration.TestHostToDevice) cout << "H2D ";
      if (Configuration.TestDeviceToHost) cout << "D2H ";
      if (Configuration.TestDeviceToDevice) cout << "D2D ";
      if (Configuration.NumaAware) cout << "NUMA ";
      cout << endl << endl;
    }

    size_t PerThreadSize = Configuration.DataSize / Configuration.NumberOfThreads;

    // NUMA-aware testing takes precedence over traditional testing
    if (Configuration.NumaAware) {
      if (Configuration.TestHostToDevice) {
        runHostToDeviceNumaMatrix();
      }
      if (Configuration.TestDeviceToHost) {
        runDeviceToHostNumaMatrix();
      }
    } else {
      // Traditional H2D and D2H tests
      if (Configuration.TestHostToDevice) {
        if (Configuration.MatrixOnly || Configuration.CsvOutput || Configuration.DeviceIds.size() == 1) {
          // For matrix output or single device, skip traditional threaded approach
          if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
            cout << "=== Host to Device Transfer ===" << endl;
          }
        } else {
          cout << "=== Host to Device Transfer ===" << endl;
          vector<thread> Threads;
          vector<BenchmarkResult> Results(Configuration.NumberOfThreads);

          for (int ThreadIndex = 0; ThreadIndex < Configuration.NumberOfThreads; ThreadIndex++) {
            Threads.emplace_back([this, ThreadIndex, PerThreadSize, &Results]() {
              Results[ThreadIndex] = benchmarkHostToDevice(ThreadIndex, PerThreadSize);
            });
          }

          for (auto &ThreadHandle : Threads) {
            ThreadHandle.join();
          }

          // Aggregate results
          double TotalBandwidth = 0;
          double TotalLatency = 0;
          int ValidResults = 0;

          for (const auto &Result : Results) {
            if (!Result.Times.empty()) {
              TotalBandwidth += Result.BandwidthGigabytesPerSecond;
              TotalLatency += Result.LatencyMicroseconds;
              ValidResults++;
            }
          }

          if (ValidResults > 0) {
            cout << "Aggregate bandwidth: ";
            formatBandwidth(TotalBandwidth, cout);
            cout << endl;
            cout << "Average latency: " << fixed << setprecision(2) 
                 << (TotalLatency / ValidResults) << " μs" << endl;
            
            if (Configuration.NumberOfThreads > 1) {
              cout << "Per-thread bandwidth: ";
              formatBandwidth(TotalBandwidth / ValidResults, cout);
              cout << endl;
            }
          }
          cout << endl;
        }
      }

      if (Configuration.TestDeviceToHost) {
        if (Configuration.MatrixOnly || Configuration.CsvOutput || Configuration.DeviceIds.size() == 1) {
          // For matrix output or single device, skip traditional threaded approach
          if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
            cout << "=== Device to Host Transfer ===" << endl;
          }
        } else {
          cout << "=== Device to Host Transfer ===" << endl;
          vector<thread> Threads;
          vector<BenchmarkResult> Results(Configuration.NumberOfThreads);

          for (int ThreadIndex = 0; ThreadIndex < Configuration.NumberOfThreads; ThreadIndex++) {
            Threads.emplace_back([this, ThreadIndex, PerThreadSize, &Results]() {
              Results[ThreadIndex] = benchmarkDeviceToHost(ThreadIndex, PerThreadSize);
            });
          }

          for (auto &ThreadHandle : Threads) {
            ThreadHandle.join();
          }

          // Aggregate results
          double TotalBandwidth = 0;
          double TotalLatency = 0;
          int ValidResults = 0;

          for (const auto &Result : Results) {
            if (!Result.Times.empty()) {
              TotalBandwidth += Result.BandwidthGigabytesPerSecond;
              TotalLatency += Result.LatencyMicroseconds;
              ValidResults++;
            }
          }

          if (ValidResults > 0) {
            cout << "Aggregate bandwidth: ";
            formatBandwidth(TotalBandwidth, cout);
            cout << endl;
            cout << "Average latency: " << fixed << setprecision(2) 
                 << (TotalLatency / ValidResults) << " μs" << endl;
            
            if (Configuration.NumberOfThreads > 1) {
              cout << "Per-thread bandwidth: ";
              formatBandwidth(TotalBandwidth / ValidResults, cout);
              cout << endl;
            }
          }
          cout << endl;
        }
      }
    }

    if (Configuration.TestDeviceToDevice) {
      // If we have multiple devices or want comprehensive testing, use matrix mode
      if (Configuration.DeviceIds.size() > 1 || Configuration.TestAllDevices) {
        runDeviceToDeviceMatrix();
      } else if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
        // Traditional single-pair D2D test only if not in special output modes
        cout << "=== Device to Device Transfer ===" << endl;
        vector<thread> Threads;
        vector<BenchmarkResult> Results(Configuration.NumberOfThreads);

        for (int ThreadIndex = 0; ThreadIndex < Configuration.NumberOfThreads; ThreadIndex++) {
          Threads.emplace_back([this, ThreadIndex, PerThreadSize, &Results]() {
            Results[ThreadIndex] = benchmarkDeviceToDevice(ThreadIndex, PerThreadSize);
          });
        }

        for (auto &ThreadHandle : Threads) {
          ThreadHandle.join();
        }

        // Aggregate results
        double TotalBandwidth = 0;
        double TotalLatency = 0;
        int ValidResults = 0;

        for (const auto &Result : Results) {
          if (!Result.Times.empty()) {
            TotalBandwidth += Result.BandwidthGigabytesPerSecond;
            TotalLatency += Result.LatencyMicroseconds;
            ValidResults++;
          }
        }

        if (ValidResults > 0) {
          cout << "Aggregate bandwidth: ";
          formatBandwidth(TotalBandwidth, cout);
          cout << endl;
          cout << "Average latency: " << fixed << setprecision(2) 
               << (TotalLatency / ValidResults) << " μs" << endl;
          
          if (Configuration.NumberOfThreads > 1) {
            cout << "Per-thread bandwidth: ";
            formatBandwidth(TotalBandwidth / ValidResults, cout);
            cout << endl;
          }
        }
        cout << endl;
      }
    }

    // Generate header file if requested
    if (!Configuration.HeaderFilePath.empty()) {
      if (!generateHeaderFile()) {
        cerr << "Warning: Header file generation failed" << endl;
      }
    }
  }

  // Header file generation functions
  string getCurrentTimestamp() {
    time_t now = time(0);
    char* timeStr = ctime(&now);
    string timestamp(timeStr);
    // Remove newline character
    if (!timestamp.empty() && timestamp.back() == '\n') {
      timestamp.pop_back();
    }
    return timestamp;
  }

  string generateIncludeGuard(const string& name) {
    string guard = name + "_DEVICE_PERFORMANCE_H";
    // Convert to uppercase and replace non-alphanumeric with underscore
    for (char& c : guard) {
      if (isalnum(c)) {
        c = toupper(c);
      } else {
        c = '_';
      }
    }
    return guard;
  }

  void writeMatrixAsArray(ofstream& file, const string& name, const vector<vector<double>>& matrix, 
                         const string& type_qualifier = "static const") {
    if (matrix.empty() || matrix[0].empty()) {
      file << type_qualifier << " double " << name << "[1][1] = {{0.0}};\n";
      return;
    }
    
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    file << type_qualifier << " double " << name << "[" << rows << "][" << cols << "] = {\n";
    for (size_t i = 0; i < rows; i++) {
      file << "  {";
      for (size_t j = 0; j < cols; j++) {
        file << fixed << setprecision(2) << matrix[i][j];
        if (j < cols - 1) file << ", ";
      }
      file << "}";
      if (i < rows - 1) file << ",";
      file << "\n";
    }
    file << "};\n\n";
  }

  void writeMatrixAsMacros(ofstream& file, const string& prefix, const vector<vector<double>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) {
      file << "#define " << prefix << "_ROWS 1\n";
      file << "#define " << prefix << "_COLS 1\n";
      file << "#define " << prefix << "(r,c) 0.0\n\n";
      return;
    }
    
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    file << "#define " << prefix << "_ROWS " << rows << "\n";
    file << "#define " << prefix << "_COLS " << cols << "\n";
    
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        file << "#define " << prefix << "_" << i << "_" << j << " " 
             << fixed << setprecision(2) << matrix[i][j] << "\n";
      }
    }
    file << "\n";
  }

  bool generateHeaderFile() {
    if (Configuration.HeaderFilePath.empty()) {
      return true; // No header generation requested
    }

    ofstream file(Configuration.HeaderFilePath);
    if (!file.is_open()) {
      cerr << "Error: Cannot create header file: " << Configuration.HeaderFilePath << endl;
      return false;
    }

    string includeGuard = generateIncludeGuard(Configuration.HeaderNamespace);
    string timestamp = getCurrentTimestamp();
    
    // Write header
    file << "/*\n";
    file << " * LLVM Offload Device Performance Data\n";
    file << " * Generated on: " << timestamp << "\n";
    file << " * Test configuration:\n";
    file << " *   Data size: " << (Configuration.DataSize / (1024 * 1024)) << " MB\n";
    file << " *   Devices: ";
    for (size_t i = 0; i < Results.DeviceIds.size(); i++) {
      file << Results.DeviceIds[i];
      if (i < Results.DeviceIds.size() - 1) file << ",";
    }
    file << "\n";
    file << " *   Iterations: " << Configuration.NumberOfIterations << "\n";
    file << " *   Threads: " << Configuration.NumberOfThreads << "\n";
    if (Configuration.NumaAware && !Results.NumaNodes.empty()) {
      file << " *   NUMA nodes: ";
      for (size_t i = 0; i < Results.NumaNodes.size(); i++) {
        file << Results.NumaNodes[i];
        if (i < Results.NumaNodes.size() - 1) file << ",";
      }
      file << "\n";
    }
    file << " */\n\n";

    file << "#ifndef " << includeGuard << "\n";
    file << "#define " << includeGuard << "\n\n";

    // Basic definitions
    file << "// Basic device information\n";
    file << "#define " << Configuration.HeaderNamespace << "_DEVICE_COUNT " << Results.DeviceCount << "\n";
    file << "#define " << Configuration.HeaderNamespace << "_NUMA_NODE_COUNT " << Results.NumaNodeCount << "\n\n";

    // Generate matrices based on format
    if (Configuration.HeaderFormat == "macros") {
      file << "// Performance matrices as macros\n";
      writeMatrixAsMacros(file, Configuration.HeaderNamespace + "_DEVICE_TO_DEVICE_BANDWIDTH_GBPS", 
                         Results.DeviceToDeviceBandwidth);
      writeMatrixAsMacros(file, Configuration.HeaderNamespace + "_DEVICE_TO_DEVICE_LATENCY_US", 
                         Results.DeviceToDeviceLatency);
      
      if (Configuration.NumaAware) {
        writeMatrixAsMacros(file, Configuration.HeaderNamespace + "_HOST_TO_DEVICE_BANDWIDTH_GBPS", 
                           Results.HostToDeviceBandwidth);
        writeMatrixAsMacros(file, Configuration.HeaderNamespace + "_DEVICE_TO_HOST_BANDWIDTH_GBPS", 
                           Results.DeviceToHostBandwidth);
      }
    } else {
      string typeQualifier = "static const";
      if (Configuration.HeaderFormat == "cpp-constexpr") {
        typeQualifier = "static constexpr";
      }
      
      file << "// Performance matrices\n";
      writeMatrixAsArray(file, Configuration.HeaderNamespace + "_DEVICE_TO_DEVICE_BANDWIDTH_GBPS", 
                        Results.DeviceToDeviceBandwidth, typeQualifier);
      writeMatrixAsArray(file, Configuration.HeaderNamespace + "_DEVICE_TO_DEVICE_LATENCY_US", 
                        Results.DeviceToDeviceLatency, typeQualifier);
      
      if (Configuration.NumaAware) {
        writeMatrixAsArray(file, Configuration.HeaderNamespace + "_HOST_TO_DEVICE_BANDWIDTH_GBPS", 
                          Results.HostToDeviceBandwidth, typeQualifier);
        writeMatrixAsArray(file, Configuration.HeaderNamespace + "_DEVICE_TO_HOST_BANDWIDTH_GBPS", 
                          Results.DeviceToHostBandwidth, typeQualifier);
      }
    }

    file << "#endif // " << includeGuard << "\n";
    file.close();

    if (!Configuration.MatrixOnly && !Configuration.CsvOutput) {
      cout << "Header file generated: " << Configuration.HeaderFilePath << endl;
    }

    return true;
  }
};

// Function to parse device ID ranges
vector<int> parseDeviceIds(const string &DeviceString) {
  vector<int> DeviceIds;
  
  if (DeviceString.empty()) {
    return DeviceIds;
  }
  
  istringstream Stream(DeviceString);
  string Token;
  
  while (getline(Stream, Token, ',')) {
    // Remove whitespace
    Token.erase(remove_if(Token.begin(), Token.end(), ::isspace), Token.end());
    
    // Check for range notation (e.g., 0-3)
    size_t DashPos = Token.find('-');
    if (DashPos != string::npos) {
      int StartDevice = stoi(Token.substr(0, DashPos));
      int EndDevice = stoi(Token.substr(DashPos + 1));
      
      if (StartDevice <= EndDevice) {
        for (int I = StartDevice; I <= EndDevice; I++) {
          DeviceIds.push_back(I);
        }
      } else {
        cerr << "Invalid device range: " << Token << endl;
      }
    } else {
      // Single device ID
      DeviceIds.push_back(stoi(Token));
    }
  }
  
  return DeviceIds;
}

void printUsage(const char *ProgramName) {
  cout << "Usage: " << ProgramName << " [OPTIONS]" << endl;
  cout << "OpenMP Offload Device Bandwidth Benchmark" << endl << endl;
  cout << "Options:" << endl;
  cout << "  -s, --size SIZE        Data size in MB (default: 128)" << endl;
  cout << "  -d, --device ID        Device ID(s) to use (default: all)" << endl;
  cout << "                         Examples: -d 0, -d 0,1,2, -d 0-3" << endl;
  cout << "  -t, --threads NUM      Number of threads (default: 1)" << endl;
  cout << "  -i, --iterations NUM   Number of iterations (default: 10)" << endl;
  cout << "  -m, --mode MODE        Test mode: h2d, d2h, d2d, all (default: all)" << endl;
  cout << "  -u, --unit UNIT        Output unit: auto, GB/s, MB/s, KB/s (default: auto)" << endl;
  cout << "  -r, --raw              Raw output (not human readable)" << endl;
  cout << "  -I, --info             Show device information" << endl;
  cout << "  -h, --help             Show this help message" << endl;
  cout << "      --numa-aware       Enable NUMA-aware testing" << endl;
  cout << "      --numa-nodes NODES NUMA node(s) to test (default: all)" << endl;
  cout << "                         Examples: --numa-nodes 0, --numa-nodes 0,1,2" << endl;
  cout << "      --matrix-only      Output only matrix results (no descriptions)" << endl;
  cout << "      --csv              Output in CSV format (pure data)" << endl;
  cout << "  -H, --generate-header FILE  Generate C/C++ header file" << endl;
  cout << "      --header-namespace NAME  Header namespace/prefix (default: LLVM_OFFLOAD)" << endl;
  cout << "      --header-format FORMAT   Header format: c-array, cpp-constexpr, macros" << endl;
  cout << "                               (default: c-array)" << endl << endl;
  cout << "Test modes:" << endl;
  cout << "  h2d                    Host to Device transfer" << endl;
  cout << "  d2h                    Device to Host transfer" << endl;
  cout << "  d2d                    Device to Device transfer (default: matrix mode)" << endl;
  cout << "  all                    All transfer types (default)" << endl << endl;
  cout << "Device specification:" << endl;
  cout << "  Single device:         -d 0" << endl;
  cout << "  Multiple devices:      -d 0,1,2" << endl;
  cout << "  Device range:          -d 0-3" << endl;
  cout << "  All devices:           (default if -d not specified)" << endl << endl;
  cout << "NUMA options:" << endl;
  cout << "  --numa-aware           Enable NUMA-aware H2D/D2H testing" << endl;
  cout << "  --numa-nodes 0,1       Test specific NUMA nodes" << endl;
  cout << "  --numa-nodes 0-3       Test NUMA node range" << endl << endl;
  cout << "Output options:" << endl;
  cout << "  --matrix-only          Show only matrix data, no headers/descriptions" << endl;
  cout << "  --csv                  Output pure CSV format (data only, no headers)" << endl << endl;
  cout << "Header generation:" << endl;
  cout << "  -H performance.h       Generate C/C++ header with performance data" << endl;
  cout << "  --header-namespace NS  Customize namespace/prefix (default: LLVM_OFFLOAD)" << endl;
  cout << "  --header-format FORMAT:" << endl;
  cout << "    c-array              Static const arrays (default)" << endl;
  cout << "    cpp-constexpr        Static constexpr arrays" << endl;
  cout << "    macros               Preprocessor macros" << endl << endl;
  cout << "Examples:" << endl;
  cout << "  " << ProgramName << " --numa-aware -H perf.h" << endl;
  cout << "  " << ProgramName << " -d 0,1 --header-format cpp-constexpr -H data.hpp" << endl;
  cout << "  " << ProgramName << " --csv --generate-header results.h" << endl;
}

size_t parseSizeString(const string &SizeString) {
  size_t Multiplier = 1024 * 1024; // Default MB
  string NumberString = SizeString;
  
  if (SizeString.back() == 'B' || SizeString.back() == 'b') {
    NumberString = SizeString.substr(0, SizeString.length() - 1);
    if (NumberString.back() == 'K' || NumberString.back() == 'k') {
      Multiplier = 1024;
      NumberString = NumberString.substr(0, NumberString.length() - 1);
    } else if (NumberString.back() == 'M' || NumberString.back() == 'm') {
      Multiplier = 1024 * 1024;
      NumberString = NumberString.substr(0, NumberString.length() - 1);
    } else if (NumberString.back() == 'G' || NumberString.back() == 'g') {
      Multiplier = 1024 * 1024 * 1024;
      NumberString = NumberString.substr(0, NumberString.length() - 1);
    }
  }
  
  return static_cast<size_t>(stod(NumberString) * Multiplier);
}

// Function to parse NUMA node IDs (similar to device IDs)
vector<int> parseNumaNodes(const string &NodeString) {
  vector<int> NumaNodes;
  
  if (NodeString.empty()) {
    return NumaNodes;
  }
  
  istringstream Stream(NodeString);
  string Token;
  
  while (getline(Stream, Token, ',')) {
    // Remove whitespace
    Token.erase(remove_if(Token.begin(), Token.end(), ::isspace), Token.end());
    
    // Check for range notation (e.g., 0-3)
    size_t DashPos = Token.find('-');
    if (DashPos != string::npos) {
      int StartNode = stoi(Token.substr(0, DashPos));
      int EndNode = stoi(Token.substr(DashPos + 1));
      
      if (StartNode <= EndNode) {
        for (int I = StartNode; I <= EndNode; I++) {
          NumaNodes.push_back(I);
        }
      } else {
        cerr << "Invalid NUMA node range: " << Token << endl;
      }
    } else {
      // Single NUMA node ID
      NumaNodes.push_back(stoi(Token));
    }
  }
  
  return NumaNodes;
}

int main(int argc, char **argv) {
  BenchmarkConfiguration Configuration;

  static struct option LongOptions[] = {
    {"size", required_argument, 0, 's'},
    {"device", required_argument, 0, 'd'},
    {"threads", required_argument, 0, 't'},
    {"iterations", required_argument, 0, 'i'},
    {"mode", required_argument, 0, 'm'},
    {"unit", required_argument, 0, 'u'},
    {"raw", no_argument, 0, 'r'},
    {"info", no_argument, 0, 'I'},
    {"help", no_argument, 0, 'h'},
    {"numa-aware", no_argument, 0, 1000},
    {"numa-nodes", required_argument, 0, 1001},
    {"matrix-only", no_argument, 0, 1002},
    {"csv", no_argument, 0, 1003},
    {"generate-header", required_argument, 0, 'H'},
    {"header-namespace", required_argument, 0, 1005},
    {"header-format", required_argument, 0, 1006},
    {0, 0, 0, 0}
  };

  int OptionCharacter;
  while ((OptionCharacter = getopt_long(argc, argv, "s:d:t:i:m:u:rIhH:", LongOptions, nullptr)) != -1) {
    switch (OptionCharacter) {
    case 's':
      Configuration.DataSize = parseSizeString(optarg);
      break;
    case 'd':
      Configuration.DeviceIds = parseDeviceIds(optarg);
      break;
    case 't':
      Configuration.NumberOfThreads = atoi(optarg);
      break;
    case 'i':
      Configuration.NumberOfIterations = atoi(optarg);
      break;
    case 'm':
      Configuration.TestHostToDevice = Configuration.TestDeviceToHost = Configuration.TestDeviceToDevice = false;
      if (strcmp(optarg, "h2d") == 0) {
        Configuration.TestHostToDevice = true;
      } else if (strcmp(optarg, "d2h") == 0) {
        Configuration.TestDeviceToHost = true;
      } else if (strcmp(optarg, "d2d") == 0) {
        Configuration.TestDeviceToDevice = true;
      } else if (strcmp(optarg, "all") == 0) {
        Configuration.TestHostToDevice = Configuration.TestDeviceToHost = Configuration.TestDeviceToDevice = true;
      } else {
        cerr << "Invalid mode: " << optarg << endl;
        return 1;
      }
      break;
    case 'u':
      Configuration.OutputUnit = optarg;
      break;
    case 'r':
      Configuration.HumanReadable = false;
      break;
    case 'I':
      Configuration.ShowDeviceInfo = true;
      break;
    case 'h':
      Configuration.ShowHelp = true;
      break;
    case 'H':
      Configuration.HeaderFilePath = optarg;
      break;
    case 1000: // numa-aware (using unique ID)
      Configuration.NumaAware = true;
      break;
    case 1001: // numa-nodes (using unique ID) 
      Configuration.NumaNodes = parseNumaNodes(optarg);
      break;
    case 1002: // matrix-only (using unique ID)
      Configuration.MatrixOnly = true;
      break;
    case 1003: // csv (using unique ID)
      Configuration.CsvOutput = true;
      break;
    case 1005: // header-namespace (using unique ID)
      Configuration.HeaderNamespace = optarg;
      break;
    case 1006: // header-format (using unique ID)
      Configuration.HeaderFormat = optarg;
      // Validate header format
      if (Configuration.HeaderFormat != "c-array" && 
          Configuration.HeaderFormat != "cpp-constexpr" && 
          Configuration.HeaderFormat != "macros") {
        cerr << "Invalid header format: " << Configuration.HeaderFormat << endl;
        cerr << "Valid options: c-array, cpp-constexpr, macros" << endl;
        return 1;
      }
      break;
    default:
      printUsage(argv[0]);
      return 1;
    }
  }

  if (Configuration.ShowHelp) {
    printUsage(argv[0]);
    return 0;
  }
  DeviceBenchmark Benchmark(Configuration);
  Benchmark.runBenchmark();

  return 0;
}