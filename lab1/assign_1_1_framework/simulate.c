/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "simulate.h"

#define C 0.15

/* Global variables */
double *old_array_global;
double *current_array_global;
double *next_array_global;
int i_max_global;
int t_max_global;
int num_threads_global;
pthread_barrier_t barrier;

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

    double *old_array = old_array_global;
    double *current_array = current_array_global;
    double *next_array = next_array_global;

    for (int t = 0; t < t_max; t++) {
        for (int i = i_start; i < i_end; i++) {
            next_array[i] = 2 * current_array[i] - old_array[i] +
                            C * (current_array[i - 1] -
                            (2 * current_array[i] - current_array[i + 1]));
        }

        pthread_barrier_wait(&barrier);

        if (thread_id == 0) {
            double *temp = old_array_global;
            old_array_global = current_array_global;
            current_array_global = next_array_global;
            next_array_global = temp;
        }

        pthread_barrier_wait(&barrier);

        old_array = old_array_global;
        current_array = current_array_global;
        next_array = next_array_global;
    }

    pthread_exit(NULL);
}

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
