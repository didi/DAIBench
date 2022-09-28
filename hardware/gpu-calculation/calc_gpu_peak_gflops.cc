#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>


#define CHECK_CUDA(x, str) \
  if((x) != cudaSuccess) \
  { \
    fprintf(stderr, str); \
    exit(EXIT_FAILURE); \
  }
#define CHECK_CUBLAS(x, str) \
  if((x) != CUBLAS_STATUS_SUCCESS) \
  { \
    fprintf(stderr, str); \
    exit(EXIT_FAILURE); \
  }

double second (void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

int cc2cores(int major, int minor)
{
  typedef struct
  {
    int SM;
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] =
  {
    {0x30, 192},
    {0x32, 192},
    {0x35, 192},
    {0x37, 192},
    {0x50, 128},
    {0x52, 128},
    {0x53, 128},
    {0x60,  64},
    {0x61, 128},
    {0x62, 128},
    {0x70,  64},
    {0x72,  64},
    {0x75,  64},
    {0x80,  64},
    {0x86, 128},
    {0x87, 128},
    {-1, -1}
  };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1)
  {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

bool has_fp16(int major, int minor)
{
  int cc = major * 10 + minor;
  return ((cc == 60) || (cc == 62) || (cc == 70) || (cc == 75));
}
bool has_fp16_hfma2(int major, int minor)
{
  int cc = major * 10 + minor;
  return (cc == 80);
}
bool has_bf16(int major, int minor)
{
  int cc = major * 10 + minor;
  return (cc == 80);
}
bool has_int8(int major, int minor)
{
  int cc = major * 10 + minor;
  return ((cc == 61) || (cc == 70) || (cc == 75) || (cc == 80));
}
bool has_tensor_core_v1(int major, int minor)
{
  int cc = major * 10 + minor;
  return ((cc == 70) || (cc == 72) );
}
bool has_tensor_core_v2(int major, int minor)
{
  int cc = major * 10 + minor;
  return (cc == 75);
}
bool has_tensor_core_v3(int major, int minor)
{
  int cc = major * 10 + minor;
  return (cc == 80);
}

typedef float Dtype;
int run_sgemm(const int M, const int N, const int K)
{
  cublasStatus_t status;
  Dtype *h_A;
  Dtype *h_B;
  Dtype *h_C, *h_gC;
  Dtype *d_A = 0;
  Dtype *d_B = 0;
  Dtype *d_C = 0;
  Dtype alpha = 1;
  Dtype beta = 0;

  const int TOTAL_ITER = 100;
  int total_iter = TOTAL_ITER;
  cublasHandle_t handle;
  double start, stop;
  /* Initialize CUBLAS */
  fprintf(stdout, "CUBLAS benchmark running...");
  CHECK_CUBLAS(cublasCreate(&handle), "**** CUBLAS handle create error\n");
  // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION), "**** CUBLAS SetMathMode failure\n"); // Very Important! Or you cannot use TensorCores
  // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "**** CUBLAS SetMathMode failure\n");
  // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "**** CUBLAS SetMathMode failure\n");
  // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "**** CUBLAS SetMathMode failure\n");
  // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "**** CUBLAS SetMathMode failure\n");
  /* Allocate host memory for the matrices */
  h_A = (Dtype *)malloc(M * K * sizeof(h_A[0]));
  if (h_A == 0)
  {
    fprintf(stderr, "!!!! host memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  h_B = (Dtype *)malloc(N * K * sizeof(h_B[0]));
  if (h_B == 0)
  {
    fprintf(stderr, "!!!! host memory allocation error (B)\n");
    return EXIT_FAILURE;
  }
  h_C = (Dtype *)malloc(M * N * sizeof(h_C[0]));
  if (h_C == 0)
  {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }
  h_gC = (Dtype *)malloc(M * N * sizeof(h_gC[0]));
  if (h_gC == 0)
  {
    fprintf(stderr, "!!!! host memory allocation error (gC)\n");
    return EXIT_FAILURE;
  }

  /* Allocate device memory for the matrices */
  CHECK_CUDA(cudaMalloc((void **)&d_A, M * K * sizeof(d_A[0])), "**** CUDA Malloc failure(d_A)\n");
  CHECK_CUDA(cudaMalloc((void **)&d_B, N * K * sizeof(d_B[0])), "**** CUDA Malloc failure(d_B)\n");
  CHECK_CUDA(cudaMalloc((void **)&d_C, M * N * sizeof(d_C[0])), "**** CUDA Malloc failure(d_C)\n");

  /* Initialize the device matrices with the host matrices */
  CHECK_CUBLAS(cublasSetVector(M * K, sizeof(h_A[0]), h_A, 1, d_A, 1), "**** CUBLAS SetVector error(A)");
  CHECK_CUBLAS(cublasSetVector(N * K, sizeof(h_B[0]), h_B, 1, d_B, 1), "**** CUBLAS SetVector error(B)");
  CHECK_CUBLAS(cublasSetVector(M * N, sizeof(h_C[0]), h_C, 1, d_C, 1), "**** CUBLAS SetVector error(C)");

  start = second();
  while(total_iter--)
  {
    // CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, M * K, d_B, K, N * K, &beta, d_C, M, M * N, 1), "CUBLAS SGEMM Failed!\n");
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M), "CUBLAS SGEMM Failed!\n");
  }
  cudaError_t cudaStatus = cudaDeviceSynchronize();
  stop = second();
  fprintf(stdout, "Done. Real performance = %g GFLOPS(M = %d, N = %d, K = %d)\n", (2.0 * M * N * K * 1e-9) / (stop - start) * TOTAL_ITER, M, N, K);
  CHECK_CUBLAS(cublasGetVector(M * N, sizeof(h_C[0]), d_C, 1, h_C, 1), "**** CUBLAS SetVector error(C)");

  /* Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_gC);
  CHECK_CUDA(cudaFree(d_A), "**** CUDA Free failure(d_A)");
  CHECK_CUDA(cudaFree(d_B), "**** CUDA Free failure(d_B)");
  CHECK_CUDA(cudaFree(d_C), "**** CUDA Free failure(d_C)");
  CHECK_CUBLAS(cublasDestroy(handle), "**** CUBLAS Destroy handle error\n");
  return 0;
}


int main(int argc, char **argv)
{
  cudaDeviceProp prop;
  int dc;
  CHECK_CUDA(cudaGetDeviceCount(&dc), "cudaGetDeviceCount error!");
  printf("GPU count = %d\n", dc);

  if(dc < 1)
  {
    fprintf(stderr, "No GPU device found. Please check hardware and driver setup.\n");
    exit(-1);
  }
  for(int i = 0; i < dc; i++)
  {
    printf("=================GPU #%d=================\n", i);
    CHECK_CUDA(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties error");
    printf("GPU Name = %s\n", prop.name);
    printf("Compute Capability = %d%d\n", prop.major, prop.minor);
    printf("GPU SMs = %d\n", prop.multiProcessorCount);
    printf("GPU CUDA cores = %d\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount);
    printf("GPU SM clock rate = %.3f GHz\n", prop.clockRate/1e6);
    printf("GPU Mem clock rate = %.3f GHz\n", prop.memoryClockRate/1e6);
    printf("FP32 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2);
    if(has_fp16(prop.major, prop.minor))
    {
      printf("FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 2);
    }
    if(has_fp16_hfma2(prop.major, prop.minor))
    {
      printf("FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 4);
    }
    if(has_bf16(prop.major, prop.minor))
    {
      printf("BF16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 2);
    }
    if(has_int8(prop.major, prop.minor))
    {
      printf("INT8 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 4);
    }
    if(has_tensor_core_v1(prop.major, prop.minor))
    {
      printf("Tensor Core FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 8);
    }
    if(has_tensor_core_v2(prop.major, prop.minor))
    {
      printf("Tensor Core FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 8);
      printf("Tensor Core INT8 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 16);
    }
    if(has_tensor_core_v3(prop.major, prop.minor))
    {
      printf("Tensor Core TF32 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 8);
      printf("Tensor Core FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 16);
      printf("Tensor Core BF16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 16);
      printf("Tensor Core INT8 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 32);
    }
    // cudaSetDevice(i);
    // run_sgemm(2048, 8192, 4096);
  }
  return 0;
}
