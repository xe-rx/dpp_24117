// queue.h

#ifndef QUEUE_H
#define QUEUE_H

#include <pthread.h>

struct queue;

// Initializes a new queue with the specified size
// Returns a pointer to the initialized queue
struct queue* queue_init(int size);

// Pushes a value into the queue
// Returns 0 on success, -1 if the queue is terminated
int queue_push(struct queue* q, int value);

// Pops a value from the queue
// Returns 0 on success, -1 if the queue is empty and terminated
int queue_pop(struct queue* q, int* value);

// Terminates the queue, waking up all threads waiting on the queue
void queue_terminate(struct queue* q);

// Cleans up and deallocates the queue
void queue_cleanup(struct queue* q);

#endif // QUEUE_H
