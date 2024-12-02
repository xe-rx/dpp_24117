/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "simulate.h"

#define C 0.15

/*
 * Executes the entire simulation.
 *
 * i_max: number of data points on a single wave
 * t_max: number of iterations the simulation should run
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max (to be filled with t+1)
 */
double *simulate(const int i_max, const int t_max, double *old_array,
                 double *current_array, double *next_array)
{
    int rank, size;
    MPI_Init(NULL, NULL); // initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = i_max - 2; // exclude boundary points
    int base_chunk = N / size;
    int remainder = N % size;

    // calc local domain size for each process
    int local_N = base_chunk + (rank < remainder ? 1 : 0);

    // calc start and end indices in the global array
    int offset = 1; // start from index 1 to exclude boundary
    for (int i = 0; i < rank; i++) {
        offset += base_chunk + (i < remainder ? 1 : 0);
    }

    // allocate local arrays with halo cells (extra two elements)
    double *old_local = (double *)malloc((local_N + 2) * sizeof(double));
    double *current_local = (double *)malloc((local_N + 2) * sizeof(double));
    double *next_local = (double *)malloc((local_N + 2) * sizeof(double));

    // scatter old_array and current_array to all processes
    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int displacement = 1; // start from index 1
        for (int i = 0; i < size; i++) {
            int count = base_chunk + (i < remainder ? 1 : 0);
            sendcounts[i] = count;
            displs[i] = displacement;
            displacement += count;
        }
    }

    // receive counts for local arrays (excluding halo cells)
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // scatter old_array
    MPI_Scatterv(old_array, sendcounts, displs, MPI_DOUBLE,
                 &old_local[1], local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // scatter current_array
    MPI_Scatterv(current_array, sendcounts, displs, MPI_DOUBLE,
                 &current_local[1], local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // initialize halo cells (boundary conditions)
    old_local[0] = 0.0;
    old_local[local_N + 1] = 0.0;
    current_local[0] = 0.0;
    current_local[local_N + 1] = 0.0;

    // identify neighboring ranks
    int left = rank - 1;
    int right = rank + 1;
    if (left < 0)
        left = MPI_PROC_NULL;
    if (right >= size)
        right = MPI_PROC_NULL;

    for (int t = 0; t < t_max; t++) {
        // exchange halo cells with neighbors
        // send left, receive from right
        MPI_Sendrecv(&current_local[1], 1, MPI_DOUBLE, left, 0,
                     &current_local[local_N + 1], 1, MPI_DOUBLE, right, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // send right, receive from left
        MPI_Sendrecv(&current_local[local_N], 1, MPI_DOUBLE, right, 1,
                     &current_local[0], 1, MPI_DOUBLE, left, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // update the local array
        for (int i = 1; i <= local_N; i++) {
            next_local[i] = 2 * current_local[i] - old_local[i] +
                            C * (current_local[i - 1] -
                                 (2 * current_local[i] - current_local[i + 1]));
        }

        // swap pointers for next iteration
        double *temp = old_local;
        old_local = current_local;
        current_local = next_local;
        next_local = temp;
    }

    // gather the results back to the root process
    MPI_Gatherv(&current_local[1], local_N, MPI_DOUBLE,
                current_array, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // free
    free(old_local);
    free(current_local);
    free(next_local);
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();

    if (rank == 0) {
        return current_array;
    } else {
        exit(0);
    }
}
