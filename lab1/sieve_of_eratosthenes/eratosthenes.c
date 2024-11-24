/*
 * SSID: 15225054 - Boris Vukajlovic
 * SSID: 14675218 - Marouan Bellari
 *
 * Program Description:
 * This program generates a sequence of prime numbers using a concurrent pipeline of threads and queues.
 * The generator thread produces an ascending sequence of integers, starting from 2, and pushes them into a shared queue.
 * The first filter thread retrieves numbers from this queue and filters out any multiples of its prime number,
 * then creates a new queue and filter thread for each prime found. Each subsequent filter thread is responsible
 * for removing multiples of its specific prime from the sequence.
 * The program stops once a maximum number of primes (MAX_PRIMES) has been printed to the output.
 */

#include "queue/queue.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define QUEUE_SIZE 100
#define MAX_PRIMES 5000

extern int terminated_queues;

volatile int primes_printed = 0;
pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;

// Generator thread function: Continuously generates an increasing sequence of integers, starting from 2,
// and pushes each integer to the queue until the queue is full or an error occurs. This function serves as
// the initial input for the pipeline of filter threads that will follow. The generator thread terminates
// when it can no longer push numbers into the queue.
void* generator_thread(void* arg) {
    struct queue* q = (struct queue*)arg;
    int num = 2;
    while (1) {
        if (queue_push(q, num) == -1) {
            break;
        }
        num++;
    }
    return NULL;
}

// Filter thread function: This function represents a filtering stage in the pipeline. Each filter thread retrieves
// a prime number from the input queue and prints it if the maximum limit (MAX_PRIMES) has not been reached.
// It then creates a new filter thread with a queue to handle numbers not divisible by its prime. Each filter thread
// continuously checks numbers from its input queue and pushes those that are not divisible by its prime to the next
// filter's queue. The filter threads continue filtering until they can no longer retrieve numbers from the queue.
void *filter_thread(void *arg) {
  struct queue *input_q = (struct queue *)arg;
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
        pthread_mutex_unlock(&count_mutex);
        exit(0);
    }

    struct queue* output_q = queue_init(QUEUE_SIZE);
    pthread_t next_filter_thread;
    pthread_create(&next_filter_thread, NULL, filter_thread, output_q);

    int num;
    while (1) {
        if (queue_pop(input_q, &num) == -1) {
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

// Main function: Initializes the main queue and creates the generator and first filter threads.
// It starts by launching the generator thread, which produces numbers for the initial queue,
// and the first filter thread to handle the filtering process. The main function waits for both
// the generator and first filter threads to complete, then outputs the number of terminated queues
// and cleans up resources before exiting.
int main() {
    struct queue *q = queue_init(QUEUE_SIZE);
    pthread_t generator;
    pthread_t first_filter;

    pthread_create(&generator, NULL, generator_thread, q);
    pthread_create(&first_filter, NULL, filter_thread, q);

    pthread_join(generator, NULL);
    pthread_join(first_filter, NULL);

    printf("%d", terminated_queues);
    queue_cleanup(q);
    return 0;
}
