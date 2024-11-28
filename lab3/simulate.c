/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "simulate.h"

#define C 0.15  // Wave speed constant

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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // get number of processes

    int N = i_max - 2;  // number of inner points (excluding boundaries)
    int base = N / size;
    int extra = N % size;
    int local_N = base + (rank < extra ? 1 : 0);  // local number of points

    // allocate local arrays with halo cells
    double *local_old_array = malloc((local_N + 2) * sizeof(double));
    double *local_current_array = malloc((local_N + 2) * sizeof(double));
    double *local_next_array = malloc((local_N + 2) * sizeof(double));

    // set up counts and displacements for scatter and gather
    int *counts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int sum = 0;
        for (int i = 0; i < size; i++) {
            counts[i] = base + (i < extra ? 1 : 0);
            displs[i] = sum + 1;
            sum += counts[i];
        }
    }

    // scatter old_array and current_array to local arrays
    MPI_Scatterv(old_array + 1, counts, displs, MPI_DOUBLE,
                 local_old_array + 1, local_N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(current_array + 1, counts, displs, MPI_DOUBLE,
                 local_current_array + 1, local_N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // initialize halo cells
    local_old_array[0] = 0.0;
    local_old_array[local_N + 1] = 0.0;
    local_current_array[0] = 0.0;
    local_current_array[local_N + 1] = 0.0;

    for (int t = 0; t < t_max; t++) {
        // exchange halo cells with neighbors
        if (rank > 0) {
            MPI_Sendrecv(&local_current_array[1], 1, MPI_DOUBLE, rank - 1, 0,
                         &local_current_array[0], 1, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            local_current_array[0] = 0.0;  // left boundary condition
        }

        if (rank < size - 1) {
            MPI_Sendrecv(&local_current_array[local_N], 1, MPI_DOUBLE, rank + 1, 0,
                         &local_current_array[local_N + 1], 1, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            local_current_array[local_N + 1] = 0.0;  // right boundary condition
        }

        // compute next_array
        for (int i = 1; i <= local_N; i++) {
            local_next_array[i] = 2 * local_current_array[i] - local_old_array[i] +
                                  C * (local_current_array[i - 1] - 2 * local_current_array[i] + local_current_array[i + 1]);
        }

        // swap arrays
        double *temp = local_old_array;
        local_old_array = local_current_array;
        local_current_array = local_next_array;
        local_next_array = temp;
    }

    // gather results back to process 0
    double *final_array = NULL;
    if (rank == 0) {
        final_array = malloc(i_max * sizeof(double));
        final_array[0] = 0.0;             // left boundary
        final_array[i_max - 1] = 0.0;     // right boundary
    }

    MPI_Gatherv(local_current_array + 1, local_N, MPI_DOUBLE,
                final_array + 1, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // copy results to current_array in process 0
    if (rank == 0) {
        memcpy(current_array, final_array, i_max * sizeof(double));
        free(final_array);
    }

    free(local_old_array);
    free(local_current_array);
    free(local_next_array);
    if (rank == 0) {
        free(counts);
        free(displs);
    }

    // return the final array in process 0
    if (rank == 0) {
        return current_array;
    } else {
        return NULL;
    }
}
