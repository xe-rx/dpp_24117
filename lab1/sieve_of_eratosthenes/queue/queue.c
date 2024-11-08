#include "queue.h"
#include <pthread.h>
#include <stdlib.h>

struct queue {
    int* data;
    int size;
    int front;
    int rear;
    int count;
    int terminated;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
};

struct queue* queue_init(int size) {
    struct queue* q = malloc(sizeof(struct queue));
    q->data = malloc(sizeof(int) * size);
    q->size = size;
    q->front = 0;
    q->rear = 0;
    q->count = 0;
    q->terminated = 0;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
    return q;
}

int queue_push(struct queue* q, int value) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == q->size && !q->terminated) {
        pthread_cond_wait(&q->not_full, &q->mutex);
    }
    if (q->terminated) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }
    q->data[q->rear] = value;
    q->rear = (q->rear + 1) % q->size;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

int queue_pop(struct queue* q, int* value) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0 && !q->terminated) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }
    if (q->terminated && q->count == 0) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }
    *value = q->data[q->front];
    q->front = (q->front + 1) % q->size;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

void queue_terminate(struct queue* q) {
    pthread_mutex_lock(&q->mutex);
    q->terminated = 1;
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
}

void queue_cleanup(struct queue* q) {
    free(q->data);
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
    free(q);
}
