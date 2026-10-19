/*
 * avx512_bench.c - Max-throughput GEMM benchmark.
 *   (1) FP32 hand-tuned AVX-512 microkernel with OpenMP (configurable threads)
 *   (2) int8 GEMM via AVX-512 VNNI (peak inference throughput)
 * Build: gcc -O3 -march=native -fopenmp -o avx512_bench avx512_bench.c -lm
 * Run:   ./avx512_bench [N] [threads]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

static double now_sec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9; }

#define MR 16
#define NR 16

static void sgemm_avx512(float *A, float *B, float *C, int M, int N, int K, int nthr){
  omp_set_num_threads(nthr);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<M;i+=MR){
    for(int j=0;j<N;j+=NR){
      __m512 c[MR];
      for(int ii=0;ii<MR;ii++) c[ii]=_mm512_setzero_ps();
      for(int k=0;k<K;k++){
        __m512 b=_mm512_load_ps(&B[(long)k*N+j]);
        for(int ii=0;ii<MR;ii++){
          __m512 a=_mm512_set1_ps(A[(long)(i+ii)*K+k]);
          c[ii]=_mm512_fmadd_ps(a,b,c[ii]);
        }
      }
      for(int ii=0;ii<MR;ii++) _mm512_store_ps(&C[(long)(i+ii)*N+j], c[ii]);
    }
  }
}

static void igemm_vnni(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K, int nthr){
  omp_set_num_threads(nthr);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j+=16){
      __m512i acc=_mm512_setzero_si512();
      for(int k=0;k<K;k+=4){
        int a=*(int32_t*)&A[(long)i*K+k];
        __m512i av=_mm512_set1_epi32(a);
        __m512i bv=_mm512_load_si512(&B[(long)k*N+j]);
        acc=_mm512_dpbusd_epi32(acc, av, bv);
      }
      _mm512_store_si512(&C[(long)i*N+j], acc);
    }
  }
}

int main(int argc, char **argv){
  int N = (argc>1)?atoi(argv[1]):2048;
  int T = (argc>2)?atoi(argv[2]):8;
  printf("=== AVX-512 Benchmark | N=%d threads=%d ===\n", N, T);
  printf("CPU logical processors=%d\n", omp_get_num_procs());

  {
    float *A=aligned_alloc(64,(long)N*N*4);
    float *B=aligned_alloc(64,(long)N*N*4);
    float *C=aligned_alloc(64,(long)N*N*4);
    for(long i=0;i<(long)N*N;i++){ A[i]=(float)rand()/RAND_MAX; B[i]=(float)rand()/RAND_MAX; C[i]=0; }
    double t0=now_sec();
    sgemm_avx512(A,B,C,N,N,N,T);
    double dt=now_sec()-t0;
    printf("FP32 AVX-512 GEMM : %10.1f GFLOPS  (%.3fs)  C[0]=%.2f\n", 2.0*N*N*N/dt/1e9, dt, C[0]);
    free(A);free(B);free(C);
  }

  {
    int8_t *A=aligned_alloc(64,(long)N*N);
    int8_t *B=aligned_alloc(64,(long)N*N);
    int32_t *C=aligned_alloc(64,(long)N*N*4);
    for(long i=0;i<(long)N*N;i++){ A[i]=rand()%8-4; B[i]=rand()%8-4; C[i]=0; }
    double t0=now_sec();
    igemm_vnni(A,B,C,N,N,N,T);
    double dt=now_sec()-t0;
    printf("int8 VNNI GEMM   : %10.1f GOPS    (%.3fs)  C[0]=%d\n", 2.0*N*N*N/dt/1e9, dt, C[0]);
    free(A);free(B);free(C);
  }
  return 0;
}
