#include <stdio.h>

#define n 30
#define N 1200

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
            A.elements[j + i*A.height] = (((i + 1) * 10) + j + 1) % 4 + 1;
        }
    }
    for(int i=0; i< B.width; i++) {
        for(int j=0; j< B.width; j++) {
            B.elements[j + i*B.height] = (((i + 1) * 10) + j + 1) % 6 + 2;
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
    Asub.width = n; Asub.height = n;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * n * row
                                         + n * col];
    return Asub;
}

//ritorna un elemento della matrice
__device__ int getElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
//scrivi un elemento nella matrice
__device__ void setElement(Matrix A, int row, int col, int value)
{
    A.elements[row * A.stride + col] = value;
}

//Kernel per algoritmo di Cannon
__global__ void cannonKernel(Matrix A, Matrix B, Matrix C)
{   
    //riga e colonna del blocco
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    //Ogni blocco computa una sottomatrice Csub di C
    Matrix Csub = getSubMatrix(C, blockRow, blockCol);
    //Ogni thread computa una colonna di Csub
    //accumulando il risultato in Cvalue
    int Cvalue = 0;
    //riga del thread
    int col = threadIdx.x;
    int row =threadIdx.y;
    //in base alla riga di A eseguo lo shift a sx
    //(col + n_shift) MOD (n_col);
    int a_shift = (blockCol + blockRow) % (gridDim.y);
    Matrix Asub = getSubMatrix(A, blockRow, a_shift);
    //in base alla colonna di B eseguo lo shift in su
    //(col + n_shift) MOD (n_col);
    int b_shift = (blockCol + blockRow) % (gridDim.x);
    Matrix Bsub = getSubMatrix(B, b_shift, blockCol);
    //Utilizza memoria condivisa
    __shared__ int sA[n][n];
    __shared__ int sB[n][n];
    sA[row][col] = getElement(Asub, row, col);
    sB[row][col] = getElement(Bsub, row, col);
    __syncthreads();
    //Moltiplica ogni sottomatrice e accumula il risultato
    //Dopo ogni moltiplicazione esegui lo shift di uno
    //La moltiplicazione viene eseguito gridDim.x volte
    for (int i=0; i < gridDim.x; i++) {
        //moltiplica Asub e Bsub
        for(int j = 0; j < blockDim.x; j++) 
            Cvalue += sA[row][j] * sB[j][col];
        //in base alla riga di A eseguo lo shift a sx di 1
        //(col + n_shift) MOD (n_col);
        a_shift = (a_shift + 1)  % gridDim.x;
        Asub = getSubMatrix(A, blockRow, a_shift);
        //in base alla colonna di B eseguo lo shift in su di 1
        //(col + n_shift) MOD (n_col);
        b_shift = (b_shift + 1) % gridDim.x;
        Bsub = getSubMatrix(B, b_shift, blockCol);
        //setta memoria condivisa con i prossimi valori
        sA[row][col] = getElement(Asub, row, col);
        sB[row][col] = getElement(Bsub, row, col);
        __syncthreads();
    }
    setElement(Csub, row, col, Cvalue);
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
    dim3 dimBlock(n,n);
    dim3 dimGrid(N/n, N/n);
    printf("N: %d\n", N);
    printf("GRIGLIA: [%d,%d]\n", N/n, N/n);
    printf("THREADS: [%d,%d]\n", n, n);
    cannonKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    gpuErrchk( cudaPeekAtLastError() );
    //Copia C da device a host
    gpuErrchk( cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost) );
    //registra stop
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