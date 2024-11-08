#include <stdio.h>
#include "queue/queue.h"

struct gen

int generator() {


  return 0;
}

int main(void){
  struct queue *q = queue_init(100);


  queue_cleanup(q);

  return 1;
}
