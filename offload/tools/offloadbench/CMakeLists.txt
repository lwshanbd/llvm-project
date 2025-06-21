message(STATUS "Building the llvm-offload-device-bench tool")

add_openmp_tool(llvm-offload-device-bench llvm-offload-device-bench.cpp)

llvm_update_compile_flags(llvm-offload-device-bench)

target_include_directories(llvm-offload-device-bench PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
)

# Check for NUMA library on Linux systems
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_library(NUMA_LIBRARY NAMES numa)
  if(NUMA_LIBRARY)
    message(STATUS "Found NUMA library: ${NUMA_LIBRARY}")
    target_link_libraries(llvm-offload-device-bench PRIVATE ${NUMA_LIBRARY})
  else()
    message(WARNING "NUMA library not found - NUMA-aware features will be disabled")
  endif()
endif()

target_link_libraries(llvm-offload-device-bench PRIVATE
  omp
  omptarget
)
