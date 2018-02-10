#include <time.h>
#include <stdio.h>
#include <stdlib.h>

long timediff(clock_t t1, clock_t t2) {
    long elapsed;
    elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
    return elapsed;
}

int main (int argc, char* argv[])
{
    clock_t t1, t2;
    long total_time;

    t1 = clock();

    int N = 1200;
    int *A, *B, *C;

    A = (int*)malloc(N*N*sizeof(int));
    B = (int*)malloc(N*N*sizeof(int));
    C = (int*)malloc(N*N*sizeof(int));

    int i, j, k;
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            A[j + i*N] = 2;//((i + 1) * 10) + j + 1;
            B[j + i*N] = 2;//((i + 1) * 10) + j + 1;
        }
    }

    for(i=0;i<N;i++)
        for(j=0;j<N;j++)        
            for(k=0;k<N;k++)   
                C[j + i*N] += A[i*N + k]*B[k*N + j];

    t2 = clock();

    total_time = timediff(t1, t2);
    printf("elapsed: %ld ms\n", total_time);

}
