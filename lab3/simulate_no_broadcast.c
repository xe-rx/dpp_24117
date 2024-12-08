/*
 * simulate.cu
 *
 * SSID: 15225054 - Boris Vukajlovic
 * SSID: 14675218 - Marouan Bellari
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi/mpi.h>
#include "simulate.h"

#define C 0.15

double *simulate(const int i_max, const int t_max, double *old_array,
                 double *current_array, double *next_array)
{
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = i_max - 2;
    int base_chunk = N / size;
    int remainder = N % size;
    int local_N = base_chunk + (rank < remainder ? 1 : 0);

    int *global_offsets = NULL;
    if (rank == 0) {
        global_offsets = (int *)malloc(size * sizeof(int));
        int offset = 1;
        for (int i = 0; i < size; i++) {
            global_offsets[i] = offset;
            offset += base_chunk + (i < remainder ? 1 : 0);
        }
    } else {
        global_offsets = (int *)malloc(size * sizeof(int));
    }

    MPI_Bcast(global_offsets, size, MPI_INT, 0, MPI_COMM_WORLD);

    int offset = global_offsets[rank];

    if (rank != 0) {
        old_array = (double *)malloc(i_max * sizeof(double));
        current_array = (double *)malloc(i_max * sizeof(double));
    }

    MPI_Bcast(old_array, i_max, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(current_array, i_max, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *old_local = (double *)malloc((local_N + 2) * sizeof(double));
    double *current_local = (double *)malloc((local_N + 2) * sizeof(double));
    double *next_local = (double *)malloc((local_N + 2) * sizeof(double));

    memcpy(&old_local[1], &old_array[offset], local_N * sizeof(double));
    memcpy(&current_local[1], &current_array[offset], local_N * sizeof(double));

    old_local[0] = 0.0;
    old_local[local_N + 1] = 0.0;
    current_local[0] = 0.0;
    current_local[local_N + 1] = 0.0;

    int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    for (int t = 0; t < t_max; t++) {
        MPI_Sendrecv(&current_local[1], 1, MPI_DOUBLE, left, 0,
                     &current_local[local_N + 1], 1, MPI_DOUBLE, right, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&current_local[local_N], 1, MPI_DOUBLE, right, 1,
                     &current_local[0], 1, MPI_DOUBLE, left, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 1; i <= local_N; i++) {
            next_local[i] = 2 * current_local[i] - old_local[i] +
                            C * (current_local[i - 1] -
                                 (2 * current_local[i] - current_local[i + 1]));
        }

        double *temp = old_local;
        old_local = current_local;
        current_local = next_local;
        next_local = temp;
    }

    int *counts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        counts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            counts[i] = base_chunk + (i < remainder ? 1 : 0);
        }
        displs[0] = 1;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + counts[i - 1];
        }
    }

    MPI_Gatherv(&current_local[1], local_N, MPI_DOUBLE, current_array, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(old_local);
    free(current_local);
    free(next_local);
    free(global_offsets);
    if (rank != 0) {
        free(old_array);
        free(current_array);
    }
    if (rank == 0) {
        free(counts);
        free(displs);
    }

    MPI_Finalize();

    if (rank == 0) {
        return current_array;
    } else {
        exit(0);
    }
}
