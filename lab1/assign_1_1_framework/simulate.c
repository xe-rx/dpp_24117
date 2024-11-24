/*
 * SSID: 15225054 - Boris Vukajlovic
 * SSID: 14675218 - Marouan Bellari
 *
 * Program Description:
 * This program simulates a wave equation or similar iterative calculation over a 1D array using pthreads for parallel processing.
 * The simulation runs for a specified number of time steps, with each thread working on a different chunk of the array.
 * The program uses a pthread barrier to synchronize threads between time steps, and the main thread manages pointer swapping to update arrays efficiently.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "simulate.h"

#define C 0.15

double *old_array_global;
double *current_array_global;
double *next_array_global;
int i_max_global;
int t_max_global;
int num_threads_global;
pthread_barrier_t barrier;

// Worker thread function: Computes the next state of a section of the array for each time step.
// Each thread updates elements within its assigned chunk, based on neighboring values in the current array.
// Synchronization barriers ensure threads proceed in unison after each update, and the main thread performs
// pointer swapping to update array references for the next iteration.
void *worker(void *arg)
{
    int thread_id = *(int *)arg;
    int num_threads = num_threads_global;
    int i_max = i_max_global;
    int t_max = t_max_global;

    int N = i_max - 2;
    int chunk_size = N / num_threads;
    int remainder = N % num_threads;
    int i_start, i_end;

    if (thread_id < remainder) {
        i_start = 1 + thread_id * (chunk_size + 1);
        i_end = i_start + chunk_size + 1;
    } else {
        i_start = 1 + thread_id * chunk_size + remainder;
        i_end = i_start + chunk_size;
    }

    for (int t = 0; t < t_max; t++) {
        for (int i = i_start; i < i_end; i++) {
            next_array_global[i] = 2 * current_array_global[i] - old_array_global[i] +
                            C * (current_array_global[i - 1] -
                            (2 * current_array_global[i] - current_array_global[i + 1]));
        }

        pthread_barrier_wait(&barrier);

        if (thread_id == 0) {
            double *temp = old_array_global;
            old_array_global = current_array_global;
            current_array_global = next_array_global;
            next_array_global = temp;
        }

        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

// Main simulation function: Sets up and manages threads for parallel simulation of array updates.
// Initializes the barrier and creates worker threads, each of which processes a chunk of the array.
// After all threads complete, cleans up allocated resources and returns the final state of the current array.
double *simulate(const int i_max, const int t_max, const int num_threads,
                 double *old_array, double *current_array, double *next_array)
{
    old_array_global = old_array;
    current_array_global = current_array;
    next_array_global = next_array;
    i_max_global = i_max;
    t_max_global = t_max;
    num_threads_global = num_threads;

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    int *thread_ids = malloc(num_threads * sizeof(int));

    pthread_barrier_init(&barrier, NULL, num_threads);

    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, worker, &thread_ids[i])) {
            fprintf(stderr, "Error creating thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);

    free(threads);
    free(thread_ids);

    return current_array_global;
}
