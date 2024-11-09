
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "simulate.h"

#define C 0.15

/* Global variables */
double *old_array_global;
double *current_array_global;
double *next_array_global;
int i_max_global;
int t_max_global;
int num_threads_global;

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

    #pragma omp parallel
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
                next_array[i] = 2 * current_array[i] - old_array[i] +
                                C * (current_array[i - 1] -
                                (2 * current_array[i] - current_array[i + 1]));
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

            old_array = old_array_global;
            current_array = current_array_global;
            next_array = next_array_global;
        }
    }

    return current_array_global;
}

