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
  int send;
  int recv;
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
  int nchannels;
  int nthreads;
};

struct modCoordinator {
  int done;
  int init;
  int sendrank;
  int recvrank;
  modCommInfo comm;
  modTaskInfo task;
  std::map<int, modRankInfo> ranks;

  // <channelId, ringIndex>
  std::map<int, int> sendRingMap;
  std::map<int, int> recvRingMap;

  ncclProxyOp *proxyOp;
  ncclInfo *info;
};

typedef enum {
  UNINITED = 0,
  META_INITED = 1,
  PER_CALL_INITED = 2,
} topoInitState;

struct modTopology {
  topoInitState init;
  int nranks;
  int nnodes;
  int nrankpernode;
  int nchannels;

  // ranks in this node
  std::vector<int> myranks;
  std::map<int, int> prev;
  std::map<int, int> next;
  // <rank, channel> -> <ringIndex>
  std::map<std::pair<int, int>, int> ringmap;
};

extern modCoordinator global_coordinator;
extern modTopology global_topology;

ncclResult_t modCoordinatorInit(modCoordinator *coordinator, ncclProxyOp* proxyOp, ncclInfo* info);

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator);

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int cid,
                                       int &size);

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int cid, int size);

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int cid, int size);

ncclResult_t modTopologyInit(modTopology *topology, ncclProxyOp *proxyOp,
                             ncclInfo *info);

ncclResult_t modTopologyUpdateMap(modTopology *topology, int rank, int channel,
                                  ncclRing *ring, int *ringranks, int nranks);

ncclResult_t modTopologyDestroy(modTopology *topology);

#endif