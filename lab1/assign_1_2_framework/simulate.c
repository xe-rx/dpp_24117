/*
 * SSID: 15225054 - Boris Vukajlovic
 * SSID: 14675218 - Marouan Bellari
 *
 * Program Description:
 * This program simulates a wave equation or similar iterative calculation over a 1D array using OpenMP for parallel processing.
 * Given initial arrays and parameters, the program calculates new values for each element based on its neighboring elements.
 * The simulation runs for a specified number of time steps, with each thread working on a different chunk of the array.
 * OpenMP barriers synchronize threads between steps, and pointers are swapped to update arrays efficiently.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "simulate.h"

#define C 0.15 // Constant used for Wave Function

double *old_array_global;
double *current_array_global;
double *next_array_global;
int i_max_global;
int t_max_global;
int num_threads_global;

// Main simulation function: Computes the next state of a 1D array over multiple time steps using parallel processing.
// Each thread processes a specific range of elements in the array, updating each element based on its neighbors.
// OpenMP barriers synchronize threads between time steps, and single-threaded execution handles pointer swapping for array updates.
double *simulate(const int i_max, const int t_max, const int num_threads,
                 double *old_array, double *current_array, double *next_array)
{
    old_array_global = old_array;
    current_array_global = current_array;
    next_array_global = next_array;
    i_max_global = i_max;
    t_max_global = t_max;
    num_threads_global = num_threads;

    int N = i_max - 2;
    int chunk_size = N / num_threads;
    int remainder = N % num_threads;
    int i_start, i_end;

    omp_set_schedule(1, 1);

    omp_set_num_threads(num_threads);

    #pragma omp parallel private(i_start, i_end, old_array, current_array, next_array)
    {
        if (omp_get_thread_num() < remainder) {
            i_start = 1 + omp_get_thread_num() * (chunk_size + 1);
            i_end = i_start + chunk_size + 1;
        } else {
            i_start = 1 + omp_get_thread_num() * chunk_size + remainder;
            i_end = i_start + chunk_size;
        }

        for (int t = 0; t < t_max; t++) {
            for (int i = i_start; i < i_end; i++) {
                next_array_global[i] = 2 * current_array_global[i] - old_array_global[i] +
                            C * (current_array_global[i - 1] -
                            (2 * current_array_global[i] - current_array_global[i + 1]));
            }

            #pragma omp barrier

            #pragma omp single
            {
                double *temp = old_array_global;
                old_array_global = current_array_global;
                current_array_global = next_array_global;
                next_array_global = temp;
            }

            #pragma omp barrier
        }
    }
    return current_array_global;
}
