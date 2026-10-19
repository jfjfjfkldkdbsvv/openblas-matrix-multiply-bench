/*
 * sgemm_bench.c — SGEMM benchmark (OpenBLAS). Reports GFLOPS for square sizes.
 * Build:  gcc -O3 -march=native -o sgemm_bench sgemm_bench.c -lopenblas -lpthread -lm
 * Run:    ./sgemm_bench [max_N]   (default 8192)
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int max_n = (argc > 1) ? atoi(argv[1]) : 8192;
    printf("===================================================\n");
    printf("SGEMM Benchmark (OpenBLAS, single precision)\n");
    printf("===================================================\n");
#ifdef OPENBLAS_VERSION
    printf("OpenBLAS: %s\n", OPENBLAS_VERSION);
#endif
    printf("%-8s %-10s %-12s %s\n", "N", "GFLOPS", "Time(ms)", "Check");
    printf("---------------------------------------------------\n");

    for (int n = 256; n <= max_n; n *= 2) {
        size_t bytes = (size_t)n * n * sizeof(float);
        float *A = (float *)malloc(bytes);
        float *B = (float *)malloc(bytes);
        float *C = (float *)malloc(bytes);
        if (!A || !B || !C) {
            fprintf(stderr, "alloc failed at N=%d\n", n);
            break;
        }
        for (size_t i = 0; i < (size_t)n * n; i++) {
            A[i] = (float)(rand() % 100) / 100.0f;
            B[i] = (float)(rand() % 100) / 100.0f;
            C[i] = 0.0f;
        }
        double t0 = now_sec();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);
        double dt = now_sec() - t0;
        double flops = 2.0 * (double)n * n * n;
        double gflops = (flops / dt) / 1e9;
        double ref = 0.0;
        for (int k = 0; k < n; k++) ref += (double)A[k] * (double)B[k * n];
        double diff = C[0] - ref; if (diff < 0) diff = -diff;
        printf("%-8d %-10.1f %-12.2f %s\n",
               n, gflops, dt * 1e3, diff < 1e-2 ? "OK" : "FAIL");
        free(A); free(B); free(C);
    }
    printf("===================================================\n");
    return 0;
}
