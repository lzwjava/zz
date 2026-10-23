// GPU benchmark for AMD Radeon Pro WX 9100 (gfx900) via HIP.
// Measures: FP32 GEMM throughput (TFLOPS), HBM bandwidth (GB/s), and an
// optional sustained "burn" loop.
//
// Build:  hipcc -O3 --offload-arch=gfx900 -o gpu_bench gpu_bench.hip
// Run:    ./gpu_bench [--size N] [--iters I] [--burn SECONDS]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t _e = (call);                                              \
        if (_e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s at %s:%d\n",                       \
                    hipGetErrorString(_e), __FILE__, __LINE__);              \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- FP32 GEMM
// Simple tiled SGEMM: each thread computes one C element.
template <int TS>
__global__ void gemm_f32(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int N) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < N; k0 += TS) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (k0 + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TS; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// -------------------------------------------------- FP32 GEMM (register tiled)
// Block tile 64x64, K-tile 16, 256 threads, each thread computes a 4x4 tile.
template <int TM, int TN, int BK>
__global__ void gemm_f32_reg(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
    constexpr int BM = 16 * TM;  // block rows
    constexpr int BN = 16 * TN;  // block cols
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Cooperative load: BK*BM (and BK*BN) elements over 256 threads.
#pragma unroll
        for (int e = tid; e < BK * BM; e += 256) {
            int k = e % BK;
            int r = e / BK;
            As[k][r] = A[(blockRow + r) * N + (k0 + k)];
        }
#pragma unroll
        for (int e = tid; e < BK * BN; e += 256) {
            int k = e % BK;
            int c = e / BK;
            Bs[k][c] = B[(k0 + k) * N + (blockCol + c)];
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][threadIdx.y * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][threadIdx.x * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(blockRow + threadIdx.y * TM + i) * N + blockCol + threadIdx.x * TN + j] = acc[i][j];
}

// ------------------------------------------------------- bandwidth copy kernel
__global__ void copy_kernel(const float4* __restrict__ in,
                            float4* __restrict__ out,
                            size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) out[i] = in[i];
}

// ------------------------------------------------------------- burn kernel
// Dense FP32 FMA loop to keep the CUs busy.
__global__ void burn_kernel(float* __restrict__ buf, int n, int reps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = buf[i];
    float y = 1.000001f;
    for (int r = 0; r < reps; ++r) {
        x = fmaf(x, y, 0.0000001f);
        y = fmaf(y, 1.0000001f, x * 1e-7f);
    }
    buf[i] = x + y;
}

static void print_device_info() {
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, dev));
    printf("Device           : %s\n", p.name);
    printf("GCN arch         : %s\n", p.gcnArchName);
    printf("Compute units    : %d\n", p.multiProcessorCount);
    printf("Clock (kHz)      : %d\n", p.clockRate);
    printf("Memory (GiB)     : %.2f\n", p.totalGlobalMem / 1073741824.0);
    printf("Memory clock kHz : %d\n", p.memoryClockRate);
    printf("Memory bus width : %d bits\n", p.memoryBusWidth);
    printf("Warp size        : %d\n", p.warpSize);
    printf("\n");
}

template <typename Launch>
static void timed_gemm(const char* label, int N, int iters, Launch launch) {
    // warmup
    for (int i = 0; i < 2; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());

    double t0 = now_s();
    for (int i = 0; i < iters; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    double t1 = now_s();

    double sec = (t1 - t0) / iters;
    double flops = 2.0 * (double)N * N * N;
    printf("%s %4d x %4d : %8.3f ms  %7.2f GFLOP/s  %6.3f TFLOPS\n",
           label, N, N, sec * 1e3, flops / sec / 1e9, flops / sec / 1e12);
}

static void bench_gemm(int N, int iters) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemset(dA, 0x3c, bytes));
    HIP_CHECK(hipMemset(dB, 0x3c, bytes));

    const int TS = 32;
    dim3 block(TS, TS);
    dim3 grid(N / TS, N / TS);
    timed_gemm("FP32 GEMM (naive)", N, iters, [&] {
        gemm_f32<TS><<<grid, block>>>(dA, dB, dC, N);
    });

    if (N % 64 == 0) {
        dim3 rblock(16, 16);
        dim3 rgrid(N / 64, N / 64);
        timed_gemm("FP32 GEMM (reg)  ", N, iters, [&] {
            gemm_f32_reg<4, 4, 16><<<rgrid, rblock>>>(dA, dB, dC, N);
        });
    }

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
}

static void bench_bandwidth(size_t mib, int iters) {
    size_t nbytes = mib * 1024 * 1024;
    size_t n4 = nbytes / sizeof(float4);
    float4 *a, *b;
    HIP_CHECK(hipMalloc(&a, n4 * sizeof(float4)));
    HIP_CHECK(hipMalloc(&b, n4 * sizeof(float4)));
    HIP_CHECK(hipMemset(a, 0x3c, n4 * sizeof(float4)));

    int block = 256;
    int grid = 4096;

    for (int i = 0; i < 3; ++i)
        copy_kernel<<<grid, block>>>(a, b, n4);
    HIP_CHECK(hipDeviceSynchronize());

    double t0 = now_s();
    for (int i = 0; i < iters; ++i)
        copy_kernel<<<grid, block>>>(a, b, n4);
    HIP_CHECK(hipDeviceSynchronize());
    double t1 = now_s();

    double sec = (t1 - t0) / iters;
    double gb = 2.0 * n4 * sizeof(float4) / 1e9;  // read + write
    printf("Copy %zu MiB      : %8.3f ms  %7.2f GB/s (r+w)\n",
           mib, sec * 1e3, gb / sec);

    HIP_CHECK(hipFree(a)); HIP_CHECK(hipFree(b));
}

static void run_burn(double seconds) {
    size_t n = 64ull * 1024 * 1024;  // 256 MiB of floats
    float* buf;
    HIP_CHECK(hipMalloc(&buf, n * sizeof(float)));
    HIP_CHECK(hipMemset(buf, 0x3c, n * sizeof(float)));

    int block = 256;
    int grid = 4096;
    const int reps = 200;  // ~ tens of ms per launch on this GPU

    printf("Burning for %.0f s (Ctrl-C to stop early)...\n", seconds);
    double t0 = now_s();
    long long launches = 0;
    while (now_s() - t0 < seconds) {
        burn_kernel<<<grid, block>>>(buf, n, reps);
        HIP_CHECK(hipDeviceSynchronize());  // keep the host and GPU in step
        launches++;
    }
    double elapsed = now_s() - t0;
    HIP_CHECK(hipFree(buf));
    // Only grid*block threads are active; each rep does ~2 FMA (4 flops).
    double active = (double)grid * block;
    if (active > (double)n) active = (double)n;
    double flops = (double)launches * active * reps * 4.0;
    printf("Burn done: %lld launches in %.1f s (%.1f GFLOPS sustained, %ld active threads)\n",
           launches, elapsed, flops / elapsed / 1e9, (long)active);
}

int main(int argc, char** argv) {
    int max_size = 4096;
    int iters = 20;
    double burn_sec = 0.0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--size") && i + 1 < argc) max_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--burn") && i + 1 < argc) burn_sec = atof(argv[++i]);
        else { printf("usage: %s [--size N] [--iters I] [--burn SECONDS]\n", argv[0]); return 1; }
    }

    print_device_info();

    bench_bandwidth(512, 50);
    printf("\n");
    for (int n = 256; n <= max_size; n *= 2)
        bench_gemm(n, iters);

    if (burn_sec > 0.0) {
        printf("\n");
        run_burn(burn_sec);
    }
    return 0;
}
