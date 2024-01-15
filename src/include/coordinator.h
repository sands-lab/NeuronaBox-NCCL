#ifndef MOD_COORDINATOR_H_
#define MOD_COORDINATOR_H_

#include "nccl.h"
#include "proxy.h"
#include <vector>
const int KERNEL_BYPASS = 1;

struct modCoordinator {
    ncclProxyOp proxyOp;
    ncclInfo info;
    std::vector<int> sendSizes;
    std::vector<int> recvSizes;
    int sendTail;
    int recvTail;
};

extern modCoordinator global_coordinator;

ncclResult_t modCoordinatorInit(modCoordinator *coordinator, ncclProxyOp* proxyOp, ncclInfo* info);

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator);

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int &size);

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int size);

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int size);

#endif