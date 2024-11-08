#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "queue/queue.h"

#define QUEUE_SIZE 100
#define MAX_PRIMES 5000

volatile int primes_printed = 0;
volatile int done = 0;
pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;

void* generator_thread(void* arg) {
    struct queue* q = (struct queue*)arg;
    int num = 2;
    while (1) {
        if (done) {
            queue_terminate(q);
            break;
        }
        if (queue_push(q, num) == -1) {
            break;
        }
        num++;
    }
    return NULL;
}

void* filter_thread(void* arg) {
    struct queue* input_q = (struct queue*)arg;
    int prime;

    if (queue_pop(input_q, &prime) == -1) {
        return NULL;
    }

    pthread_mutex_lock(&count_mutex);
    if (primes_printed < MAX_PRIMES) {
        primes_printed++;
        printf("%d\n", prime);
        pthread_mutex_unlock(&count_mutex);
    } else {
        done = 1;
        pthread_mutex_unlock(&count_mutex);
        queue_terminate(input_q);
        return NULL;
    }

    struct queue* output_q = queue_init(QUEUE_SIZE);
    pthread_t next_filter_thread;
    pthread_create(&next_filter_thread, NULL, filter_thread, output_q);

    int num;
    while (1) {
        if (queue_pop(input_q, &num) == -1) {
            break;
        }
        if (done) {
            queue_terminate(output_q);
            break;
        }
        if (num % prime != 0) {
            if (queue_push(output_q, num) == -1) {
                break;
            }
        }
    }

    pthread_join(next_filter_thread, NULL);
    queue_cleanup(output_q);
    return NULL;
}

int main() {
    struct queue* q = queue_init(QUEUE_SIZE);
    pthread_t generator;
    pthread_t first_filter;

    pthread_create(&generator, NULL, generator_thread, q);
    pthread_create(&first_filter, NULL, filter_thread, q);

    pthread_join(generator, NULL);
    pthread_join(first_filter, NULL);

    queue_cleanup(q);
    return 0;
}
