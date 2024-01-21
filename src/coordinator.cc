#include "coordinator.h"
#include "align.h"
#include "comm.h"
#include <assert.h>
#include <cinttypes>
#include <math.h>
#include <stdlib.h>
#include <string>
using namespace std;
// total = 4000
modCoordinator global_coordinator;
int KERNEL_BYPASS = 0;
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

static int getKernelBypass() {
  char *env = getenv("NCCL_KERNEL_BYPASS");
  if (env == NULL) {
    return 0;
  }
  KERNEL_BYPASS = atoi(env);
  return KERNEL_BYPASS;
}

static void calc_size_inkernel(int nelem, vector<int> &res) {
  LOG_MOD(NCCL_MOD, "calc_size_inkernel: nelem=%d", nelem);
  int stepSize = 131072; // DEFAULT_BUFFSIZE(simple) / NCCL_STEP / sizeof(float)
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
                      int mychannel, int nthreads, int tsize,
                      vector<int> &res) {
  assert(nranks == 2);
  const int chunkSize = 524288;
  int bid = mychannel;
  int loopSize = nchannels * nranks * chunkSize;
  int size = count;
  int ringIx = myrank;

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
  std::string szs;
  for (int i = 0; i < res.size(); i++) {
    szs = szs + " " + std::to_string(res[i]);
    res[i] *= tsize;
  }
  LOG_MOD(NCCL_MOD, "Calculated sizes for rank %d: %s", myrank, szs.c_str());
}

ncclResult_t modCoordinatorInit(modCoordinator *coordinator, ncclProxyOp *proxyOp, ncclInfo *info) {
    int count = info->count;
    ncclComm *comm = info->comm;
    int nranks = comm->nRanks;
    int myrank = comm->rank;
    int nchannels = info->nChannels;
    int nthreads = info->nThreads;
    int tsize = sizeof(float);
    getKernelBypass();
    LOG_MOD(NCCL_MOD,
            "modCoordinatorInit: K_BYPASS=%d, count=%d, nranks=%d, myrank=%d, "
            "nchannels=%d, "
            "nthreads=%d",
            KERNEL_BYPASS, count, nranks, myrank, nchannels, nthreads);
    coordinator->proxyOp = *proxyOp;
    coordinator->info = *info;
    for (int i = 0; i < nchannels; ++i) {
      modChannelInfo ch;
      ch.bid = i;
      calc_size(nranks, myrank, count, nchannels, i, nthreads, tsize,
                ch.sendSizes);
      calc_size(nranks, 1 - myrank, count, nchannels, i, nthreads, tsize,
                ch.recvSizes);
      ch.sendTail = 0;
      ch.recvTail = 0;
      coordinator->channels.push_back(ch);
    }
    LOG_MOD(NCCL_MOD, "modCoordinatorInit");
    return ncclSuccess;
}

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator) {
    LOG_MOD(NCCL_MOD, "modCoordinatorDestroy");
    return ncclSuccess;
}

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int cid,
                                       int &size) {
  auto &ch = coordinator->channels[cid];
  if (ch.sendTail <= ch.recvTail) {
    size = ch.sendSizes[ch.sendTail];
  } else {
    size = -1;
    LOG_MOD(NCCL_MOD, "sendTail exceeds recvTail");
  }
    LOG_MOD(NCCL_MOD, "modCoordinatorGetSendSize: size=%d", size);
    return ncclSuccess;
}

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int cid,
                                int size) {
  auto &ch = coordinator->channels[cid];
  if (ch.sendSizes[ch.sendTail] == size) {
    ch.sendTail++;
  } else {
    LOG_MOD(NCCL_MOD, "send size unmatch actual: %d != expected: %d", size,
            ch.sendSizes[ch.sendTail]);
  }
  LOG_MOD(NCCL_MOD, "modCoordinatorSend: size=%d", size);
  return ncclSuccess;
}

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int cid,
                                int size) {
  auto &ch = coordinator->channels[cid];
  if (ch.recvSizes[ch.recvTail] == size) {
    ch.recvTail++;
  } else {
    LOG_MOD(NCCL_MOD, "recv size unmatch actual: %d != expected: %d", size,
            ch.recvSizes[ch.recvTail]);
  }
  LOG_MOD(NCCL_MOD, "modCoordinatorRecv: size=%d", size);
  return ncclSuccess;
}