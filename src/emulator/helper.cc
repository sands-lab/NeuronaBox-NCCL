#include "helper.h"

using namespace std;

void Info2Task(ncclInfo *info, modTaskInfo *task) {
  task->unique_id = info->unique_id;
  task->nchannels = info->nchannels;
  task->nrankpernode = info->nrankpernode;
  task->nranks = info->nranks;
  task->nnodes = info->nnodes;
  task->myrank = info->myrank;
}