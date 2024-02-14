#include "helper.h"

using namespace std;

void Info2Task(ncclInfo *info, modTaskInfo *task) {
  task->count = info->count;
  task->tsize = 0;
  if (info->datatype == ncclInt8)
    task->tsize = 1;
  else if (info->datatype == ncclUint8)
    task->tsize = 1;
  else if (info->datatype == ncclInt32)
    task->tsize = 4;
  else if (info->datatype == ncclUint32)
    task->tsize = 4;
  else if (info->datatype == ncclInt64)
    task->tsize = 8;
  else if (info->datatype == ncclUint64)
    task->tsize = 8;
  else if (info->datatype == ncclFloat16)
    task->tsize = 2;
  else if (info->datatype == ncclFloat32)
    task->tsize = 4;
  else if (info->datatype == ncclFloat64)
    task->tsize = 8;
  else
    task->tsize = 4;
  task->coll = info->coll;
  task->reduceOp = info->op;
  task->algo = info->algorithm;
  task->proto = info->protocol;
  task->nchannels = info->nChannels;
  task->nthreads = info->nThreads;
  task->unique_id = info->unique_id;
}