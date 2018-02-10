#include <mpi.h>
#include <stdio.h>
#include <math.h> //mettere -lm alla fine del comando per compilare (per gcc)
#include <stdlib.h>


//Funzione per Prodotto scalare
int moltiplica(int *a, int *b, int *c, int n)  // prodotto delle matrici a,b in c
{                                              // occhio che le pensiamo come matrici
    int i,j,k;                                 // su una riga, cioe' le righe sono 
    for(i=0;i<n;i++)                           // una dietro l'altra
        for(j=0;j<n;j++)        
            for(k=0;k<n;k++)   
                c[i*n+j] += a[i*n+k]*b[k*n+j];    
}  


//FUNZIONE PER ALGORITMO CANNON
void cannon(int *a, int *b, int *c, int n)
{    
    int i;    
    int nproc, rank, rank_griglia;    
    int dim[2], periodicita[2], coord[2];    
    int rankDx, rankSx, rankSu, rankGiu;    
    int sorgente, destinazione; 
 
    MPI_Status status;                              // crea la variabile "status per la comunicazione"  
    MPI_Comm griglia;                               // variabile associata al nuovo comunicatore  
  
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

    dim[0] = dim[1] = sqrt(nproc);                  // dimensione della griglia logica n x n 
    periodicita[0] = periodicita[1] = 1;            // definisco cicliche sia sull'asse x che sull'asse y
    
    // crea nuovo communicator, chiamato griglia   
    MPI_Cart_create(
        MPI_COMM_WORLD, //da comm world
        2,              //griglia 2D
        dim,            //dimensioni nuovo comm
        periodicita,    //periodicità sia su x che su y
        0,              //riordina rank
        &griglia         //nuovo comunicator
    );

    
    // trova il mio rank nel communicator griglia  
    MPI_Comm_rank(griglia, &rank_griglia);

    // trova le mie coordinate in questo comm 
    MPI_Cart_coords(
        griglia,        //da che comunicator vado a prenderle
        rank_griglia,   //rank processo nella griglia
        2,              //dimensione vettore coordinate
        coord           //vettore coordinate
    );

    // trova rank dei miei vicini di dx e sx
    MPI_Cart_shift(
        griglia,     //comunicator griglia
        1,           //direzione sull'asse x
        coord[0],           //mi muovo di coord passi
        &rankSx,      //rank sorgente
        &rankDx       //rank destinazione
    );

    // trova rank dei miei vicini    su e giu
    MPI_Cart_shift(
        griglia,     //comunicator griglia
        0,           //direzione sull'asse x
        coord[1],           //mi muovo di coord passi
        &rankSu,      //rank sorgente
        &rankGiu       //rank destinazione
    );

    // sposta le sottomatrici "a"
    MPI_Sendrecv_replace(a, n*n, MPI_INT, rankSx, 1, rankDx, 1, griglia, &status);

    // sposta le sottomatrici "b"
    MPI_Sendrecv_replace(b, n*n, MPI_INT, rankSu, 1, rankGiu, 1, griglia, &status);

    // aspetta che tutti lo abbiano fatto
    MPI_Barrier(griglia);

    for (i = 0; i < dim[0]; i++) {      
        moltiplica(a, b, c, n);  // fai prodotto matriciale delle sottomatrici "a" e "b" all'interno del processo 

        // sposta le sottomatrici "a" a sx di una posizione (contention free formula) TUTTE LE RIGHE
        MPI_Cart_shift(
            griglia,     //comunicator griglia
            1,           //direzione sull'asse x
            1,           //mi muovo di 1 passo ora
            &rankSx,      //rank sorgente
            &rankDx       //rank destinazione
        );

        // sposta le sottomatrici "b" su   di una posizione(contention free formula)  
        MPI_Cart_shift(
            griglia,     //comunicator griglia
            0,           //direzione sull'asse x
            1,           //mi muovo di 1 passo ora
            &rankSu,      //rank sorgente
            &rankGiu       //rank destinazione
        );

        //invia nuova matrice
        MPI_Sendrecv_replace(a, n*n, MPI_INT, rankSx, 1, rankDx, 1, griglia, &status);
        MPI_Sendrecv_replace(b, n*n, MPI_INT, rankSu, 1, rankGiu, 1, griglia, &status);
    }

    MPI_Comm_free(&griglia);         // elimina la griglia virtuale
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
                tmpA[i][j] = (i % 4) + 1;//((i + 1) * 10) + j + 1;
                tmpB[i][j] = (j % 4) + 1;//((i + 1) * 10) + j + 1;
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

        for(int i = 0; i < N; i++) {
            free(tmpA[i]);
            free(tmpB[i]);
        }
        free(tmpA);
        free(tmpB);
}


//MAIN
int main(int argc, char* argv[]) {

    double begin_time = MPI_Wtime();

    //Inizializzazione variabili per ogni processo
    int N;              //NxN per le Matrici. N.B.: nproc deve essere = N
    int n;              //nxn per le sottomatrici
    int i, j;           //indici di spostamento
    int root = 0;       //root per confronto con rank = 0
    int *A, *B, *C;     //putantori alle Matrici
    int *a, *b, *c;     //puntatori alle sottomatrici
    int rank, nproc;    //rank e numero processi

    //Inizializzazione ambiente
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    //Inizializzazione dimensione matrici
    N = (int)sqrt(nproc) * 300;          //Dimensione Matrici
    n = N/sqrt(nproc);   //dimensioni sottomatrici

    //se sei root alloca e inizializza le Matrici
    if (rank == root) {
        //Allocazione memoria per le matrici
        A = (int *)malloc(N*N*sizeof(int));
        B = (int *)malloc(N*N*sizeof(int));
        C = (int *)calloc(N*N, sizeof(int)); //azzerare matrice con calloc
        initMatrici(A, B, C, N, n);
    }

    //Allocazione sottomatrici
    a = (int *)calloc(n*n, sizeof(int));
    b = (int *)calloc(n*n, sizeof(int));
    c = (int *)calloc(n*n, sizeof(int));

    // manda ad ogni processo una sottomatrice a  
    MPI_Scatter(
        A, //Buffer contenente sottomatrici
        n*n, //manda n*n dati
        MPI_INT,
        a, // sottomatrice a per ogni processo
        n*n, //ricevi n*n dati
        MPI_INT,
        root, //processo che invia
        MPI_COMM_WORLD
    );

    // manda ad ogni processo una sottomatrice b  
    MPI_Scatter(
        B, //Buffer contenente sottomatrici
        n*n, //manda n*n dati
        MPI_INT,
        b, // sottomatrice a per ogni processo
        n*n, //ricevi n*n dati
        MPI_INT,
        root, //processo che invia
        MPI_COMM_WORLD
    );
    
    // a questo punto ogni processo ha in memoria
    //una sottomatrice "a" e una "b"
    cannon(a, b, c, n);
    
    // aspetta che tutti i processi abbiano lavorato
    MPI_Barrier(MPI_COMM_WORLD);

    // raccogli tutte le sottomatrici c in C 
    MPI_Gather(c, n * n, MPI_INT, C, n*n, MPI_INT, 0, MPI_COMM_WORLD);

    double total_time = MPI_Wtime();

    if(rank == root) {
        printf("- Total time: %f\n", total_time - begin_time);
    }

    //Finalizza ambiente
    MPI_Finalize();
    return 0;
}
