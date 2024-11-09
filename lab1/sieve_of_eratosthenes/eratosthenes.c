#include "queue/queue.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define QUEUE_SIZE 100
#define MAX_PRIMES 5000

extern int terminated_queues;

volatile int primes_printed = 0;
volatile int done = 0;
pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;

void *generator_thread(void *arg) {
  struct queue *q = (struct queue *)arg;
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

void *filter_thread(void *arg) {
  struct queue *input_q = (struct queue *)arg;
  int prime;

  if (queue_pop(input_q, &prime) == -1) {
    return NULL;
  }

  pthread_mutex_lock(&count_mutex);
  if (primes_printed < MAX_PRIMES) {
    primes_printed++;
    printf("%d, total primes printed %d\n", prime, primes_printed);
    pthread_mutex_unlock(&count_mutex);
  } else {
    printf("setting flag...\n");
    done = 1;
    printf("flag set, done!\n");
    pthread_mutex_unlock(&count_mutex);
    queue_terminate(input_q);
    return NULL;
  }

  struct queue *output_q = queue_init(QUEUE_SIZE);
  pthread_t next_filter_thread;
  pthread_create(&next_filter_thread, NULL, filter_thread, output_q);

  int num;

  while (1) {
    if (done) {
      queue_terminate(output_q);
      break;
    }
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
