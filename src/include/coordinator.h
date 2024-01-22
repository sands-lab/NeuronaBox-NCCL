#ifndef MOD_COORDINATOR_H_
#define MOD_COORDINATOR_H_

#include "nccl.h"
#include "proxy.h"
#include <vector>

extern int KERNEL_BYPASS;

struct modChannelInfo {
  int bid;
  std::vector<int> sendSizes;
  std::vector<int> recvSizes;
  int sendTail;
  int recvTail;
};

struct modCoordinator {
  int done;
  ncclProxyOp proxyOp;
  ncclInfo info;
  std::vector<modChannelInfo> channels;
};

extern modCoordinator global_coordinator;

ncclResult_t modCoordinatorInit(modCoordinator *coordinator, ncclProxyOp* proxyOp, ncclInfo* info);

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator);

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int cid,
                                       int &size);

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int cid, int size);

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int cid, int size);

#endif