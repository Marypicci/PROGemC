//código que mede tempo em multiplicação de matrizes da matéria de Computação Paralela


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N; // dimensão da matriz
    double st, end, tempo, tempo_global;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // cada processo terá N_local = N/size linhas
    if (N % size != 0) {
        if (rank == 0)
            printf("Erro: N deve ser divisível por size.\n");
        MPI_Finalize();
        return 0;
    }

    int N_local = N / size;

    // Alocar vetores
    double *A_local = (double*) malloc(N_local * N * sizeof(double));
    double *x = (double*) malloc(N * sizeof(double));
    double *y_local = (double*) malloc(N_local * sizeof(double));
    double *y = NULL;

    if (rank == 0) {
        srand(rank);
        double *A = (double*) malloc(N * N * sizeof(double));
        y = (double*) malloc(N * sizeof(double));

        for (int i = 0; i < N; i++) {
            x[i] = rand() % (10 * N * N);
            for (int j = 0; j < N; j++) {
                A[i*N + j] = rand() % (10 * N * N);
            }
        }

        MPI_Scatter(A, N_local*N, MPI_DOUBLE,
                    A_local, N_local*N, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        free(A);
    } else {
        MPI_Scatter(NULL, N_local*N, MPI_DOUBLE,
                    A_local, N_local*N, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // --- medir tempo a partir daqui ---
    st = MPI_Wtime();

    // broadcast do vetor x
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // cálculo local
    for (int i = 0; i < N_local; i++) {
        y_local[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y_local[i] += A_local[i*N + j] * x[j];
        }
    }

    // reunir resultados
    MPI_Gather(y_local, N_local, MPI_DOUBLE,
               y, N_local, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    end = MPI_Wtime();
    tempo = end - st;

    // pega o tempo máximo entre todos os processos (tempo real da execução)
    MPI_Reduce(&tempo, &tempo_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Tempo de execução: %f segundos\n", tempo_global);
        free(y);
    }

    free(A_local);
    free(x);
    free(y_local);

    MPI_Finalize();
    return 0;
}
