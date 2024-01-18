#include "coordinator.h"
#include "align.h"
#include "comm.h"
#include <assert.h>
#include <cinttypes>
#include <math.h>
using namespace std;
// total = 4000
modCoordinator global_coordinator;

/*

template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}

__device__ static int Simple::calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
}

using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS,
ALLREDUCE_SLICESTEPS>; #define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4) = 2
SlicePerChunk #define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2) = 4 StepPerSlice

stepSize(stepSize_ == 0 ?
ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T) : stepSize_)
stepSize = 131072


    int sliceSize = stepSize*StepPerSlice; // 131072 * 4 = 524288
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
              = max([(nelem+32 - 1) / 32] * 16)
    sliceSize = sliceSize < nelem - offset ? sliceSize : nelem - offset;
    waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(
        srcIx, dstIx, offset, sliceSize);


const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id ==
NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1)); = 524288

const ssize_t loopSize = nChannels*nranks*chunkSize;

!for now gridOffset = 0, since one loop is enough for our size

if (Proto::Id == NCCL_PROTO_SIMPLE) {
    realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*nranks));
    realChunkSize = roundUp(realChunkSize,
(nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T)); !for simple, WRAP_SIZE = 32
}
else
    realChunkSize = min(chunkSize, divUp(size-gridOffset,
nChannels*nranks*minChunkSize)*minChunkSize); realChunkSize =
int(realChunkSize); nelem = min(realChunkSize, size-offset);


*/

static void calc_size_inkernel(int nelem, vector<int> &res) {
  LOG_MOD(NCCL_MOD, "calc_size_inkernel: nelem=%d", nelem);
  int stepSize = 524288; // DEFAULT_BUFFSIZE(simple) / NCCL_STEP
  int SlicePerChunk = 2; // all reduce
  int StepPerSlice = 2;  //! i don't know why
  int sliceSize = stepSize * StepPerSlice;
  sliceSize = std::max(DIVUP(nelem, 16 * SlicePerChunk) * 16, sliceSize / 32);
  int offset = 0, slice = 0;
  while (offset < nelem) {
    int size = sliceSize < nelem - offset ? sliceSize : nelem - offset;
    res.push_back(size);
    offset += sliceSize;
    slice += 1;
  }

  while (slice < SlicePerChunk) {
    sliceSize = sliceSize < nelem - offset ? sliceSize : nelem - offset;
    res.push_back(0);
    offset += sliceSize;
    slice += 1;
  }
}

static void calc_size(int nranks, int myrank, int count, int nchannels,
                      int nthreads, int tsize, vector<int> &res) {
  assert(nranks == 2);
  int bid = 0;
  const int chunkSize = 524288;
  int loopSize = nchannels * nranks * chunkSize;
  int size = count;
  int ringIx = 0;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    ssize_t realChunkSize;
    // if proto == Simple
    realChunkSize =
        min(chunkSize, (int)DIVUP(size - gridOffset, nchannels * nranks));
    realChunkSize =
        ROUNDUP(realChunkSize, (nthreads - 32) * sizeof(uint64_t) / tsize);
    realChunkSize = int(realChunkSize);

    LOG_MOD(NCCL_MOD, "realChunkSize=%lu, nthreads=%d", realChunkSize,
            nthreads);

    auto calcOffset = [&](int chunk) -> ssize_t {
      return gridOffset + bid * nranks * realChunkSize + chunk * realChunkSize;
    };
    auto modRanks = [&](int r) -> int {
      return r - (r >= nranks ? nranks : 0);
    };

    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = modRanks(ringIx + nranks - 1);
    offset = calcOffset(chunk);
    nelem = std::min(realChunkSize, size - offset);
    calc_size_inkernel(nelem, res);

    // k-2 steps: reduce and copy to next GPU
    for (int j = 2; j < nranks; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      offset = calcOffset(chunk);
      nelem = std::min(realChunkSize, size - offset);
      calc_size_inkernel(nelem, res);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ringIx + 0;
    offset = calcOffset(chunk);
    nelem = std::min(realChunkSize, size - offset);
    calc_size_inkernel(nelem, res);

    // k-2 steps: copy to next GPU
    for (int j = 1; j < nranks - 1; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      offset = calcOffset(chunk);
      nelem = std::min(realChunkSize, size - offset);
      calc_size_inkernel(nelem, res);
    }
  }

  LOG_MOD(NCCL_MOD, "Calculated size for rank %d:", myrank);
  for (int i = 0; i < res.size(); i++) {
    LOG_MOD(NCCL_MOD, "size[%d]=%d", i, res[i]);
  }
}

ncclResult_t modCoordinatorInit(modCoordinator *coordinator, ncclProxyOp *proxyOp, ncclInfo *info) {
    int count = info->count;
    ncclComm *comm = info->comm;
    int nranks = comm->nRanks;
    int myrank = comm->rank;
    int nchannels = info->nChannels;
    int nthreads = info->nThreads;
    LOG_MOD(NCCL_MOD,
            "modCoordinatorInit: count=%d, nranks=%d, myrank=%d, nchannels=%d, "
            "nthreads=%d",
            count, nranks, myrank, nchannels, nthreads);
    coordinator->proxyOp = *proxyOp;
    coordinator->info = *info;
    coordinator->sendSizes = vector<int>();
    calc_size(nranks, myrank, count, nchannels, nthreads, 4,
              coordinator->sendSizes);
    // copy sendSizes and reverse, then we can use it as recvSizes
    coordinator->recvSizes = vector<int>(coordinator->sendSizes);
    std::reverse(coordinator->recvSizes.begin(), coordinator->recvSizes.end());

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
  if (coordinator->sendTail <= coordinator->recvTail) {
    size = coordinator->sendSizes[coordinator->sendTail];
  } else {
    size = -1;
    LOG_MOD(NCCL_MOD, "sendTail exceeds recvTail");
  }
    LOG_MOD(NCCL_MOD, "modCoordinatorGetSendSize: size=%d", size);
    return ncclSuccess;
}

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int size) {
  if (coordinator->sendSizes[coordinator->sendTail] == size) {
    coordinator->sendTail++;
  } else {
    LOG_MOD(NCCL_MOD, "send size unmatch actual: %d != expected: %d", size,
            coordinator->sendSizes[coordinator->sendTail]);
  }
  LOG_MOD(NCCL_MOD, "modCoordinatorSend: size=%d", size);
  return ncclSuccess;
}

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int size) {
  if (coordinator->recvSizes[coordinator->recvTail] == size) {
    coordinator->recvTail++;
  } else {
    LOG_MOD(NCCL_MOD, "recv size unmatch actual: %d != expected: %d", size,
            coordinator->recvSizes[coordinator->recvTail]);
  }
  LOG_MOD(NCCL_MOD, "modCoordinatorRecv: size=%d", size);
  return ncclSuccess;
}