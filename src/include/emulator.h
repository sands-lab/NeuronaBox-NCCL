#ifndef EMULATOR_H
#define EMULATOR_H

#include "driver_types.h"
#include "nccl.h"
#include "proxy.h"
#include <cstdint>
#include <map>
#include <set>
#include <sys/types.h>
#include <vector>

// forward declarations
struct modCoordinator;
struct modTopology;
struct modController;

// begin global
// env vars
extern int MOD_KERNEL_BYPASS;
extern int MOD_N_NODES;
extern int MOD_MY_NODE;
extern modCoordinator global_coordinator;
extern modTopology global_topology;
extern modController global_controller;

ncclResult_t modGetAllEnvVars();
// end global

// begin coordinator

// channel represents a connection between two ranks
struct modChannelInfo {
  int bid;
  int send;
  int recv;
  std::vector<int> sendsizes;
  std::vector<int> recvsizes;
  int sendtail;
  int recvtail;
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
  int coll;      // i.e. allreduce
  int reduceOp;  // i.e. sum
  int algo;      // i.e. ring
  int proto;     // i.e. Simple
  int nchannels;
  int nthreads;
  uint64_t unique_id;
};

struct modEmulatorTask {
  int done;
  int init;
  int sendrank;
  int recvrank;
  modTaskInfo info;
  std::map<int, modRankInfo> ranks;
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
  std::map<uint64_t, modEmulatorTask> id2task;

  ncclProxyOp *proxyOp;
  ncclInfo *info;
};

ncclResult_t modCoordinatorInit(modCoordinator *coordinator,
                                ncclProxyOp *proxyOp, ncclInfo *info);

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator);

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int cid,
                                       int &size);

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int cid, int size);

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int cid, int size);

// end coordinator

// begin topology

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
  std::set<int> myranks;
  std::map<int, int> prev;
  std::map<int, int> next;
  // <rank, channel> -> <ringIndex>
  std::map<std::pair<int, int>, int> ringmap;
};

ncclResult_t modTopologyInit(modTopology *topology, ncclProxyOp *proxyOp,
                             ncclInfo *info);

ncclResult_t modTopologyUpdateMap(modTopology *topology, int rank, int channel,
                                  ncclRing *ring, int *ringranks, int nranks);

ncclResult_t modTopologyDestroy(modTopology *topology);

// end topology

// begin controller

struct modController {
  std::map<uint64_t, modEmulatorTask> id2task;
  std::map<cudaStream_t, std::vector<uint64_t>> stream2ids;
  modCoordinator *coordinator;
  modTopology *topology;
  modCommInfo *comm;
};

ncclResult_t
modControllerAddTask(modController *controller, ncclInfo *info);

ncclResult_t modControllerQueryTask(modController *controller,
                                    uint64_t unique_id, modTaskInfo *task);

ncclResult_t modControllerRemoveTask(modController *controller,
                                     uint64_t unique_id);

ncclResult_t modControllerCheck(modController *controller, uint64_t unique_id,
                                int &bypass);

#endif // EMULATOR_H
