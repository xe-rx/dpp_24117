#define _POSIX_SOURCE 200122L
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include "queue/queue.h"
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#define BUFFERSIZE (size_t)80

int num_threads = 0;
int flag = 0;

pthread_barrier_t barrier;
pthread_mutex_t mutex_primes;

void *filter_thread(void *arg) {
    struct queue *input_queue = (struct queue *)arg;
    struct queue *output_queue = queue_init(BUFFERSIZE);
    int prime;

    while (1){
        if (queue_empty(input_queue) == 1) {
            continue;
        }
        pthread_mutex_lock(&mutex_primes);
        prime = queue_pop(input_queue);
        pthread_mutex_unlock(&mutex_primes);

        printf("%d", prime);
        break;
    }

    pthread_t filter;
    pthread_create(&filter, NULL, filter_thread, output_queue);
    num_threads++;

    while(1) {
        if (queue_empty(input_queue) == 1) {
            continue;
        }
        int popped = queue_pop(input_queue);
        if ((popped % prime) == 1) {
            queue_push(output_queue, popped);
        }
        if (flag == 1) {
            break;
        }
    }

    pthread_barrier_wait(&barrier);
    queue_cleanup(output_queue);
    pthread_barrier_wait(&barrier);
    return NULL;
}

void *generator_thread(void *arg) {
    struct queue *output_queue = (struct queue *)arg;
    int number = 2;
    num_threads++;

    pthread_t filter;
    pthread_create(&filter, NULL, filter_thread, output_queue);

    while (1) {
        if (queue_push(output_queue, number++) == 1) {
            number--;
        }
        if (flag == 1) {
            break;
        }
    }
    pthread_barrier_wait(&barrier);
    queue_cleanup(output_queue);
    pthread_barrier_wait(&barrier);
    return NULL;
}

int main (void){
    struct queue *generator_queue = queue_init(BUFFERSIZE);
    if (generator_queue == NULL) {
        exit(0);
    }

    pthread_mutex_init(&mutex_primes, NULL);

    pthread_t generator;
    pthread_create(&generator, NULL, generator_thread, generator_queue);
    num_threads++;

    if (num_threads >= 5000) {
        flag = 1;
        pthread_barrier_init(&barrier, NULL, num_threads);
    }

    free(generator_queue);
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&mutex_primes);
}