#include <stdio.h>
#include <stdlib.h>

#include "queue.h"

struct queue {
    int *collection;
    size_t capacity;
    size_t max_size;
    size_t total_enqueue;
    size_t total_dequeue;
};

/* Initializes the queue with a specified capacity.

   capacity: The initial size to allocate for the queue.

   Returns: A pointer to the newly created queue or NULL if an error occurs.
*/
struct queue *queue_init(size_t capacity) {
    if (capacity <= 0) return NULL;

    struct queue *q = malloc(sizeof(struct queue));
    if (q == NULL) return NULL;

    q->collection = malloc(sizeof(int) * capacity);
    if (q->collection == NULL) {
        free(q);
        return NULL;
    }

    q->capacity = capacity;
    q->max_size = 0;
    q->total_dequeue = 0;
    q->total_enqueue = 0;

    return q;
}

/* Cleans up the queue and releases allocated memory.

   q: A pointer to the queue that is being destroyed.
*/
void queue_cleanup(struct queue *q) {
    if (q == NULL) {
        return;
    }
    free(q->collection);
    free(q);
}

/* Displays current statistics of the queue, such as push and pop counts and
    maximum elements seen.

   q: A pointer to the queue whose stats are to be printed.
*/
void queue_stats(const struct queue *q) {
    if (q == NULL) {
        return;
    }
    fprintf(stderr, "stats %zu %zu %zu\n",
            q->total_enqueue, q->total_dequeue, q->max_size);
}

/* Adds an element to the back of the queue.

   q: A pointer to the queue.
   e: The element to be added to the queue.

   Returns: 0 if the operation is successful, or 1 if the queue is full.
*/
int queue_push(struct queue *q, int e) {
    if (q == NULL) {
        perror("Pointer points to nothing.");
        return 1;
    } else if (!(q->capacity == queue_size(q))) {
        q->collection[q->total_enqueue++ % q->capacity] = e;
        if (queue_size(q) > q->max_size) {
            q->max_size = queue_size(q);
        }
        return 0;
    } else {
        return 1;
    }
}

/* Removes and returns the element from the front of the queue.

   q: A pointer to the queue.

   Returns: The element at the front of the queue or -1 if the queue is empty.
*/
int queue_pop(struct queue *q) {
    if (q == NULL) {
        perror("Pointer points to nothing");
        return -1;
    } else if (queue_size(q) > 0) {
        int dequeued = q->collection[q->total_dequeue++ % q->capacity];
        return dequeued;
    } else {
        return -1;
    }
}

/* Retrieves, but does not remove, the head of this queue.

   q: A pointer to the queue.

   Returns: The head of the queue or -1 if the queue is empty.
*/
int queue_peek(const struct queue *q) {
    if (q == NULL) {
        perror("Pointer points to nothing");
        return -1;
    } else if (queue_size(q) > 0){
        return q->collection[q->total_dequeue % q->capacity];
    } else {
        return -1;
    }
}

/* Checks if the queue is empty.

   q: A pointer to the queue.

   Returns: 1 if the queue is empty, 0 if it is not, or -1 if
   the pointer is NULL.
*/
int queue_empty(const struct queue *q) {
    if (q == NULL) {
        perror("Pointer points to nothing");
        return -1;
    } else if (queue_size(q) > 0) {
        return 0;
    } else if (queue_size(q) <= 0) {
        return 1;
    } else {
        return -1;
    }
}

/* Returns the number of elements in the queue.

   q: A pointer to the queue.

   Returns: The current size of the queue.
*/
size_t queue_size(const struct queue *q) {
    if (q == NULL) {
        return 0;
    }
    return (q->total_enqueue - q->total_dequeue);
}
