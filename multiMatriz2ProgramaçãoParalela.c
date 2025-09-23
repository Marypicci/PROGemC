// segundo código mais eficiente de programação paralela de matrizes

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N ; // dimensão da matriz
    int dims[2], periods[2] = {0, 0}, coords[2];
    double st, end, time;
    MPI_Comm cart_comm, row_comm, col_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) N = atoi(argv[1]);

    // Definir grade 2D de processos (quadrada, se possível)
    dims[0] = dims[1] = 0;
    MPI_Dims_create(size, 2, dims); // calcula fatores próximos
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int p_rows = dims[0], p_cols = dims[1];
    int row = coords[0], col = coords[1];

    if (N % p_rows != 0 || N % p_cols != 0) {
        if (rank == 0) {
            printf("Erro: N deve ser divisível por p_rows=%d e p_cols=%d\n", p_rows, p_cols);
        }
        MPI_Finalize();
        return 0;
    }

    int n_local_rows = N / p_rows;
    int n_local_cols = N / p_cols;

    // Aloca blocos locais
    double *A_local = (double*) malloc(n_local_rows * n_local_cols * sizeof(double));
    double *x_local = (double*) malloc(n_local_cols * sizeof(double));
    double *y_partial = (double*) calloc(n_local_rows, sizeof(double));
    double *y_local = (double*) calloc(n_local_rows, sizeof(double));
    double *y = NULL;

    if (rank == 0) {
        // Cria matriz e vetor completos
        double *A = (double*) malloc(N * N * sizeof(double));
        double *x = (double*) malloc(N * sizeof(double));
        srand(rank);	
        for (int i = 0; i < N; i++) {
            x[i] = rand()%(10*N*N); // vetor aleatório
            for (int j = 0; j < N; j++) {
                A[i*N + j] = rand()%(10*N*N);
            }
        }

        // Distribui blocos de A manualmente
        for (int pr = 0; pr < p_rows; pr++) {
            for (int pc = 0; pc < p_cols; pc++) {
                int dest_rank;
                int coords2[2] = {pr, pc};
                MPI_Cart_rank(cart_comm, coords2, &dest_rank);

                // Extrai submatriz
                double *tmp = (double*) malloc(n_local_rows * n_local_cols * sizeof(double));
                for (int ii = 0; ii < n_local_rows; ii++) {
                    for (int jj = 0; jj < n_local_cols; jj++) {
                        tmp[ii*n_local_cols + jj] =
                            A[(pr*n_local_rows + ii) * N + (pc*n_local_cols + jj)];
                    }
                }
                if (dest_rank == 0) {
                    for (int k = 0; k < n_local_rows*n_local_cols; k++)
                        A_local[k] = tmp[k];
                } else {
                    MPI_Send(tmp, n_local_rows*n_local_cols, MPI_DOUBLE,
                             dest_rank, 0, MPI_COMM_WORLD);
                }
                free(tmp);
            }
        }

        // Distribui blocos de x
        for (int pc = 0; pc < p_cols; pc++) {
            double *tmp = (double*) malloc(n_local_cols * sizeof(double));
            for (int jj = 0; jj < n_local_cols; jj++) {
                tmp[jj] = x[pc*n_local_cols + jj];
            }
            int coords2[2] = {0, pc};
            int dest_rank;
            MPI_Cart_rank(cart_comm, coords2, &dest_rank);
            if (dest_rank == 0) {
                for (int k = 0; k < n_local_cols; k++) x_local[k] = tmp[k];
            } else {
                MPI_Send(tmp, n_local_cols, MPI_DOUBLE, dest_rank, 1, MPI_COMM_WORLD);
            }
            free(tmp);
        }

        free(A);
        free(x);
    } else {
        MPI_Recv(A_local, n_local_rows*n_local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (row == 0) {
            MPI_Recv(x_local, n_local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Criar comunicadores de linha e coluna
    MPI_Comm_split(cart_comm, row, col, &row_comm);
    MPI_Comm_split(cart_comm, col, row, &col_comm);

    st=MPI_Wtime();

    // Broadcast do vetor x_local ao longo da coluna
    MPI_Bcast(x_local, n_local_cols, MPI_DOUBLE, 0, col_comm);

    // Cálculo local: y_partial = A_local * x_local
    for (int i = 0; i < n_local_rows; i++) {
        for (int j = 0; j < n_local_cols; j++) {
            y_partial[i] += A_local[i*n_local_cols + j] * x_local[j];
        }
    }

    // Reduzir soma ao longo da linha de processos
    MPI_Reduce(y_partial, y_local, n_local_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    // Reunir resultado em rank 0 global
    if (col == 0) {
        if (rank == 0) {
            y = (double*) malloc(N * sizeof(double));
        }
        MPI_Gather(y_local, n_local_rows, MPI_DOUBLE,
                   y, n_local_rows, MPI_DOUBLE,
                   0, col_comm);
    }

    end = MPI_Wtime();
    time = end - st;

    printf("%f \n", time);

    // Imprimir resultado
    //if (rank == 0) {
    //    printf("Resultado y = A * x:\n");
    //    for (int i = 0; i < N; i++) {
    //        printf("%6.2f \n", y[i] );
    //    }
    //    printf("\n");
    //    free(y);
    //}

    free(A_local);
    free(x_local);
    free(y_partial);
    free(y_local);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}
