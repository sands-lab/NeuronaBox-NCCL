#include "coordinator.h"
#include <assert.h>
#include <math.h>
using namespace std;
// total = 4000
modCoordinator global_coordinator;


#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

/*

template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}

using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4) = 2 SlicePerChunk
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2) = 4 StepPerSlice

stepSize(stepSize_ == 0 ? ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T) : stepSize_) 
stepSize = 131072


    int sliceSize = stepSize*StepPerSlice; // 131072 * 4 = 524288
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32); 
              = max([(nelem+32 - 1) / 32] * 16)
    sliceSize = sliceSize < nelem - offset ? sliceSize : nelem - offset;
    waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(
        srcIx, dstIx, offset, sliceSize);
*/
static void calc_size(int nranks, int myrank, int nelem, int tsize, vector<int> &res) {
    assert(nranks == 2);
    int stepSize = 524288; // DEFAULT_BUFFSIZE(simple) / NCCL_STEP
    int SlicePerChunk = 2; // all reduce
    int StepPerSlice = 4; // all reduce
    int sliceSize = stepSize*StepPerSlice; 
    sliceSize = std::max(DIVUP(nelem, 16*SlicePerChunk)*16, sliceSize/32);

    int offset = 0;
    while (offset < nelem) {
        int size = sliceSize < nelem - offset ? sliceSize : nelem - offset;
        res.push_back(size);
        offset += sliceSize;
    }
    // for rank 1, we need to reverse the order
    if (myrank == 1) {
        reverse(res.begin(), res.end());
    }
    LOG_MOD(NCCL_MOD, "Calculated size for rank %d:", myrank);
    for (int i = 0; i < res.size(); i++) {
        LOG_MOD(NCCL_MOD, "size[%d]=%d", i, res[i]);
    }
}

ncclResult_t modCoordinatorInit(modCoordinator *coordinator, ncclProxyOp *proxyOp, ncclInfo *info) {
    int count = info->count;
    int nranks = info->comm->nRanks;
    int myrank = info->comm->rank;
    count = count * (nranks - 1);
    LOG_MOD(NCCL_MOD, "modCoordinatorInit: count=%d, nranks=%d, myrank=%d", count, nranks, myrank);
    coordinator->proxyOp = *proxyOp;
    coordinator->info = *info;
    coordinator->sendSizes = vector<int>();
    coordinator->recvSizes = vector<int>();
    calc_size(nranks, myrank, count, sizeof(float), coordinator->sendSizes);
    calc_size(nranks, nranks - myrank, count, sizeof(float), coordinator->recvSizes);

    coordinator->sendTail = 0;
    coordinator->recvTail = 0;
    LOG_MOD(NCCL_MOD, "modCoordinatorInit");
    return ncclSuccess;
}

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator) {
    LOG_MOD(NCCL_MOD, "modCoordinatorDestroy");
    return ncclSuccess;
}

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int &size) {
    size = 4000;
    LOG_MOD(NCCL_MOD, "modCoordinatorGetSendSize: size=%d", size);
    return ncclSuccess;
}

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int size) {
    LOG_MOD(NCCL_MOD, "modCoordinatorSend: size=%d", size);
    return ncclSuccess;
}

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int size) {
    LOG_MOD(NCCL_MOD, "modCoordinatorRecv: size=%d", size);
    return ncclSuccess;
}