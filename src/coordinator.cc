#include "coordinator.h"
using namespace std;
// total = 4000
modCoordinator global_coordinator;

ncclResult_t modCoordinatorInit(modCoordinator *coordinator, ncclProxyOp *proxyOp, ncclInfo *info) {
    coordinator->proxyOp = *proxyOp;
    coordinator->info = *info;
    coordinator->sendSizes = vector<int>(1, 4000);
    coordinator->recvSizes = vector<int>(1, 4000);
    coordinator->sendTail = 0;
    coordinator->recvTail = 0;
    return ncclSuccess;
}

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator) {
    return ncclSuccess;
}

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int &size) {
    size = 4000;
    return ncclSuccess;
}

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int size) {
    return ncclSuccess;
}

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int size) {
    return ncclSuccess;
}