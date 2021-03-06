#include <stdio.h>

#define Q 5
#define BLOCKS_NUM (Q*Q*Q)
#define N (Q * 240)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct {
    int width;
    int height;
    int stride;
    int* elements;
} Matrix;

//inizializza le matrici
void initMatrix(Matrix A, Matrix B) {
    for(int i=0; i< A.width; i++) {
        for(int j=0; j< A.width; j++) {
            A.elements[j + i*A.height] = 2;//((i + 1) * 10) + j + 1;
        }
    }
    for(int i=0; i< B.width; i++) {
        for(int j=0; j< B.width; j++) {
            B.elements[j + i*B.height] = 3;//((i + 1) * 10) + j + 1;
        }
    }
}

//inizializza le matrici
void printMatrix(Matrix A) {
    for(int i=0; i< A.width; i++) {
        for(int j=0; j< A.width; j++) {
            printf("%d\t", A.elements[j + i*A.height]);
        }
        printf("\n");
    }
}

//Ritorna la sottomatrice Asub di dimensione BLOCK_SIZExBLOCK_SIZE
//di A che è localizzata col sottomatrici verso destra
// e row sottomatrici verso il basso
//dall'angolo in alto a sinistra di A
__device__ Matrix getSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = blockDim.x; Asub.height = blockDim.x;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * blockDim.x * row
                                         + blockDim.x * col];
    return Asub;
}

//ritorna un elemento della matrice
__device__ int getElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void multiplyAndReduce(Matrix A, Matrix B, Matrix C)
{
    int col = threadIdx.x;
    for (int k=0; k < blockDim.x; k++) {
        int Cvalue = 0;
        for(int j=0; j < blockDim.x; j++) 
            Cvalue += getElement(A, k, j) * getElement(B, j, col);
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = k + blockIdx.y * blockDim.x;
        int offset = x + y * blockDim.x * gridDim.x;
        atomicAdd(&C.elements[offset], Cvalue);
    }
}

//Kernel per algoritmo DNS
__global__ void dnsKernel(Matrix A, Matrix B, Matrix C)
{
     //riga colonna e profondità del blocco
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int blockDeep = blockIdx.z;
    //Ricevi la sottomatrice di A in base all'asse z
    Matrix Asub = getSubMatrix(A, blockRow, blockDeep);
    //Ricevi la sottomatrice di B in base all'asse z
    Matrix Bsub = getSubMatrix(A, blockDeep, blockCol);
    //Moltiplica e riduci con una operazione di somma sull'asse z
    multiplyAndReduce(Asub, Bsub, C);
}

int main(void) {

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //matrici di partenza
    Matrix A, B, C;
    A.width = N; A.height = N; A.stride = N;
    B.width = N; B.height = N; B.stride = N;
    C.width = N; C.height = N; C.stride = N;
    //allocazione matrici
    size_t size = A.width * A.height * sizeof(int);
    A.elements = (int *)malloc(size);
    size = B.width * B.height * sizeof(int);
    B.elements = (int *)malloc(size);
    size = C.width * C.height * sizeof(int);
    C.elements = (int *)malloc(size);
    //inizializzazione matrici
    initMatrix(A, B);
    //matrici su GPU
    Matrix d_A, d_B, d_C;
    d_A.width = A.width; d_A.height = A.height; d_A.stride = N;
    d_B.width = B.width; d_B.height = B.height; d_B.stride = N;
    d_C.width = C.width; d_C.height = C.height; d_C.stride = N;
    //allocazione matrici su GPU
    size = d_A.width * d_A.height * sizeof(int);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    size = d_B.width * d_B.height * sizeof(int);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    size = d_C.width * d_C.height * sizeof(int);
    cudaMalloc(&d_C.elements, size);
    //invoca kernel
    dim3 dimBlock(N/Q);
    dim3 dimGrid(Q, Q, Q);
    dnsKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost) );
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed Time: %3.1f ms/n\n\n", elapsedTime);
    //libera memoria
    free(A.elements);
    free(B.elements);
    free(C.elements);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    return 0;
}