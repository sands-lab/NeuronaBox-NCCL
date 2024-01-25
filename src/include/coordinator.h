#ifndef MOD_COORDINATOR_H_
#define MOD_COORDINATOR_H_

#include "nccl.h"
#include "proxy.h"
#include <map>
#include <vector>
extern int KERNEL_BYPASS;

// channel represents a connection between two ranks
struct modChannelInfo {
  int bid;
  std::vector<int> sendSizes;
  std::vector<int> recvSizes;
  int sendTail;
  int recvTail;
};

// rank represents a gpu device
struct modRankInfo {
  int myrank;
  int send;
  int recv;
  int done;
  std::vector<modChannelInfo> channels;
};

// Comm info
struct modCommInfo {
  int nranks;
  int nnodes;
  int nrankpernode;
  int mynode;
};

// Task Info
struct modTaskInfo {
  int count;     // number of elements
  int tsize;     // size of each element
  int primitive; // placeholder, always allreduce
  int reduceOp;  // placeholder, always sum
  int algo;      // placeholder, always ring
  int proto;     // placeholder, always Simple
};

struct modCoordinator {
  int done;
  int init;
  int sendrank;
  int recvrank;
  modCommInfo comm;
  modTaskInfo task;
  std::map<int, modRankInfo> ranks;

  ncclProxyOp proxyOp;
  ncclInfo info;
};

extern modCoordinator global_coordinator;

ncclResult_t modCoordinatorInit(modCoordinator *coordinator, ncclProxyOp* proxyOp, ncclInfo* info);

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator);

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int cid,
                                       int &size);

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int cid, int size);

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int cid, int size);

#endif