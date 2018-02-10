#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> //mettere -lm alla fine del comando per compilare (per gcc)

//Funzione per Prodotto di matrici
int moltiplica(int *a, int *b, int *c, int n)  // prodotto delle matrici a,b in c
{
    int i,j,k;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)        
            for(k=0;k<n;k++)   
                c[i*n+j] += a[i*n+k]*b[k*n+j];    
} 

void dns(int *A, int *B, int *C, int *a, int *b, int *c, int n, int N)
{   
    int root = 0;
    int nproc, rank, rank_grid, rank_piano_ij, rank_piano_ik;
    int dim[3], period[3], coord[3];
    MPI_Comm grid, piano_ij, piano_ik, piano_k, piano_i, piano_j;

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    dim[0] = dim[1] = dim[2] = cbrt(nproc);  // dimensione della griglia logica n x n x n
    period[0] = period[1] = period[2] = 1;       

    // crea nuovo communicator, chiamato grid   
    MPI_Cart_create(
        MPI_COMM_WORLD, //da comm world
        3,              //griglia 3D
        dim,            //dimensioni nuovo comm
        period,    //periodicità sia su x che su y
        0,              //riordina rank
        &grid         //nuovo comunicator
    );

    // trova il rank nel communicator grid  
    MPI_Comm_rank(grid, &rank_grid);

    // trova le coordinate nel communicator grid
    MPI_Cart_coords(
        grid,        //da che comunicator vado a prenderle
        rank_grid,   //rank processo nella griglia
        3,              //dimensione vettore coordinate
        coord           //vettore coordinate
    );

    //dividi la griglia per i piani ij
    dim[0] = dim[1] = 1;
	dim[2] = 0;
	MPI_Cart_sub( grid, dim, &piano_ij);
    MPI_Comm_rank(piano_ij, &rank_piano_ij);

    //dividi la griglia per i piani ik
    dim[0] = dim[2] = 1;
	dim[1] = 0;
	MPI_Cart_sub( grid, dim, &piano_ik);

    //dividi la griglia per i piani k
    dim[2] = 1;
	dim[0] =  dim[1] = 0;
	MPI_Cart_sub( grid, dim, &piano_k);

    //dividi la griglia per i piani i
    dim[0] = 1;
	dim[1] =  dim[2] = 0;
	MPI_Cart_sub( grid, dim, &piano_i);

    //dividi la griglia per i piani j
    dim[1] = 1;
	dim[0] =  dim[2] = 0;
	MPI_Cart_sub( grid, dim, &piano_j);

    double startup_time = MPI_Wtime();

    //Invio matrici al piano k=0
    if(coord[2] == 0 ) {
		MPI_Scatter( A, n*n, MPI_INT, a, n*n, MPI_INT, root, piano_ij );
        MPI_Scatter( B, n*n, MPI_INT, b, n*n, MPI_INT, root, piano_ij );
	}

    //Invio matrici nei piani k
    MPI_Bcast( a, n*n, MPI_INT, root, piano_k);
    MPI_Bcast( b, n*n, MPI_INT, root, piano_k);

    double load_time = MPI_Wtime();

    //Bcast della riga su a in base a K
    MPI_Bcast( a, n*n, MPI_INT, coord[2], piano_j);
    //Bcast della colonna su b in base a K
    MPI_Bcast( b, n*n, MPI_INT, coord[2], piano_i);

    double bcast_time = MPI_Wtime();

    moltiplica(a, b, c, n);

    double mul_time = MPI_Wtime();

    int *c_ridotta;
    if( coord[2] == 0 )
        c_ridotta = (int*)malloc(n*n*sizeof(int));

    MPI_Reduce( c, c_ridotta, n*n, MPI_INT, MPI_SUM, root, piano_k);
    c = c_ridotta;
    
    double reduce_time = MPI_Wtime();

    if(coord[2] == 0) {
       MPI_Gather(c_ridotta, n*n, MPI_INT, C, n*n, MPI_INT, 0, piano_ij);
    }

    double gather_time = MPI_Wtime();

}

void initMatrici(int *A, int *B, int *C, int N, int n)
{
        int **tmpA = (int **)malloc(N * sizeof(int*));
        for(int i = 0; i < N; i++)
            tmpA[i] = (int *)malloc(N * sizeof(int));

        int **tmpB = (int **)malloc(N * sizeof(int*));
        for(int i = 0; i < N; i++)
            tmpB[i] = (int *)malloc(N * sizeof(int));

        //inizializzazione matrici
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                tmpA[i][j] = 2;//((i + 1) * 10) + j + 1;
                tmpB[i][j] = 3;//((i + 1) * 10) + j + 1;
            }
        }
        int count = 0;
        //creazione sottomatrici da passare ai vari processi
        for(int x = 0; x < N/n; x++) {
            for(int y = 0; y < N/n; y++) {
                for(int i = 0; i < n; i++) {
                    for(int j = 0; j < n; j++) {
                        A[count] = tmpA[i + x * n][j + y * n];
                        B[count] = tmpB[i + x * n][j + y * n];
                        count++;
                    }
                }
            }
        }
}

int main(int argc, char *argv[])
{
    double begin_time = MPI_Wtime();

    int N, n;           //Dimensione matrici e sottomatrici;
    int root = 0;       //Processo root
    int rank, nproc;    //variabili per memorizzare il rank del processo e il numero di processi
    int *A, *B, *C;     //Puntatori alle matrici;
    int *a, *b, *c;     //Puntatori alle sottomatrici;

    //Inizializzazione ambiente
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    if((float)ceilf(cbrt(nproc)) != (float)cbrt(nproc)) {
        if(rank == root) printf("ERRORE: Il numero dei processi deve essere una radice cubica\n\n");
        MPI_Finalize();
        return 0;
    }

    //dimensione matrici 
    N = (int)cbrt(nproc) * 400;
    //dimensione sottomatrici       
    n = (int)(N/cbrt(nproc));

    //Se root alloca e inizializza le matrici
    if(rank == root) {
        //Allocazione memoria per le matrici
        A = (int*)malloc(N*N*sizeof(int));
        B = (int*)malloc(N*N*sizeof(int));
        C = (int*)calloc(N*N, sizeof(int));
        initMatrici(A, B, C, N, n);
    }

    //allocazione sottomatrici
    a = (int *)calloc(n*n, sizeof(int));
    b = (int *)calloc(n*n, sizeof(int));
    c = (int *)calloc(n*n, sizeof(int));

    //Inizio algoritmo DNS
    dns(A, B, C, a, b, c, n, N);

    double total_time = MPI_Wtime();

    if(rank == root) {
        printf("- Total time: %f\n", total_time - begin_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
