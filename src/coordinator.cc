#include "coordinator.h"
#include "align.h"
#include "comm.h"
#include <assert.h>
#include <cinttypes>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <string>
using namespace std;
// total = 4000
modCoordinator global_coordinator;
int KERNEL_BYPASS = -1;
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
  if (KERNEL_BYPASS != -1) {
    return KERNEL_BYPASS;
  }
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

static void calc_sendsize_channel(int nranks, int myrank, int count,
                                  int nchannels, int mychannel, int nthreads,
                                  int tsize, vector<int> &res) {
  // assert(nranks == 2);
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

static void calc_recvsize_channel(int nranks, int myrank, int count,
                                  int nchannels, int mychannel, int nthreads,
                                  int tsize, vector<int> &res) {}

static int check_done_ch(modChannelInfo &ch) {
  if (ch.sendTail == ch.sendSizes.size() &&
      ch.recvTail == ch.recvSizes.size()) {
    return 1;
  }
  return 0;
}

static int check_done_rank(modRankInfo &rank) {
  if (rank.done) {
    return 1;
  }
  int done = 1;
  for (int i = 0; i < rank.channels.size(); ++i) {
    done = done & check_done_ch(rank.channels[i]);
  }
  rank.done = 1;
  LOG_MOD(NCCL_MOD, "rank update check_done_rank: done=%d, rank=%d", done,
          rank.myrank);
  return done;
}

static void check_done(modCoordinator *coordinator) {
  if (coordinator->done) {
    return;
  }
  int done = 1;
  for (int i = 0; i < coordinator->ranks.size(); ++i) {
    done = done & check_done_rank(coordinator->ranks[i]);
  }
  if (done) {
    coordinator->done = 1;
    LOG_MOD(NCCL_MOD, "coordinator update check_done: done");
  }
}

static void rankInit(modCoordinator *coordinator, ncclProxyOp *proxyOp,
                     ncclInfo *info) {
  modRankInfo &rankinfo = coordinator->ranks[info->comm->rank];
  rankinfo.myrank = info->comm->rank;
  if (rankinfo.myrank % 2 == 0) {
    rankinfo.recv = 1;
    rankinfo.send = 0;
  } else {
    rankinfo.recv = 0;
    rankinfo.send = 1;
  }
  rankinfo.channels = vector<modChannelInfo>();
  for (int i = 0; i < info->nChannels; ++i) {
    modChannelInfo ch;
    ch.bid = i;
    ch.sendSizes = vector<int>();
    ch.recvSizes = vector<int>();
    if (rankinfo.send) {
      calc_sendsize_channel(info->comm->nRanks, rankinfo.myrank, info->count,
                            info->nChannels, i, info->nThreads, sizeof(float),
                            ch.sendSizes);
    }
    if (rankinfo.recv) {
      calc_recvsize_channel(info->comm->nRanks, 1 - rankinfo.myrank,
                            info->count, info->nChannels, i, info->nThreads,
                            sizeof(float), ch.recvSizes);
    }
    ch.sendTail = 0;
    ch.recvTail = 0;
    rankinfo.channels.push_back(ch);
  }
}

static void metaInit(modCoordinator *coordinator, ncclProxyOp *proxyOp,
                     ncclInfo *info) {
  if (!coordinator->init) {
    coordinator->init = 1;
    coordinator->done = 0;
    coordinator->proxyOp = *proxyOp;
    coordinator->info = *info;

    modTaskInfo task;
    task.count = info->count;
    task.tsize = sizeof(float);
    task.primitive = 0;
    task.reduceOp = 0;
    task.algo = 0;

    modCommInfo comm;
    comm.nranks = info->comm->nRanks;
    comm.nnodes = atoi(getenv("N_MPI_RANKS")); // should be set by application!
    comm.mynode = atoi(getenv("MY_MPI_RANK")); // should be set by application!
    comm.nrankpernode = comm.nranks / comm.nnodes;
    assert(comm.nranks % comm.nnodes == 0);

    coordinator->comm = comm;
    coordinator->task = task;
    coordinator->ranks = map<int, modRankInfo>();
    LOG_MOD(
        NCCL_MOD,
        "comm.nranks=%d, comm.nnodes=%d, comm.mynode=%d, comm.nrankpernode=%d",
        comm.nranks, comm.nnodes, comm.mynode, comm.nrankpernode);
  }
}

ncclResult_t modCoordinatorInit(modCoordinator *coordinator, ncclProxyOp *proxyOp, ncclInfo *info) {
  getKernelBypass();
  metaInit(coordinator, proxyOp, info);
  int count = coordinator->task.count;
  assert(count == info->count);
  ncclComm *comm = info->comm;
  int nranks = comm->nRanks;
  int myrank = comm->rank;
  int nchannels = info->nChannels;
  int nthreads = info->nThreads;
  LOG_MOD(NCCL_MOD,
          "modCoordinatorInit: kbypass=%d, count=%d, nranks=%d, myrank=%d, "
          "nchannels=%d, "
          "nthreads=%d",
          KERNEL_BYPASS, count, nranks, myrank, nchannels, nthreads);
   rankInit(coordinator, proxyOp, info);
   return ncclSuccess;
}

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator) {
    LOG_MOD(NCCL_MOD, "modCoordinatorDestroy");
    return ncclSuccess;
}

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int myrank,
                                       int cid, int &size) {
  auto &ch = coordinator->ranks[myrank].channels[cid];
  if (ch.sendTail <= ch.recvTail) {
    size = ch.sendSizes[ch.sendTail];
  } else {
    size = -1;
    LOG_MOD(NCCL_MOD, "sendTail exceeds recvTail");
  }
    LOG_MOD(NCCL_MOD, "modCoordinatorGetSendSize: size=%d", size);
    return ncclSuccess;
}

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int myrank,
                                int cid, int size) {
  auto &ch = coordinator->ranks[myrank].channels[cid];
  if (ch.sendSizes[ch.sendTail] == size) {
    ch.sendTail++;
  } else {
    LOG_MOD(NCCL_MOD, "send size unmatch actual: %d != expected: %d", size,
            ch.sendSizes[ch.sendTail]);
  }
  LOG_MOD(NCCL_MOD, "modCoordinatorSend: size=%d", size);
  return ncclSuccess;
}

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int myrank,
                                int cid, int size) {
  auto &ch = coordinator->ranks[myrank].channels[cid];
  if (ch.recvSizes[ch.recvTail] == size) {
    ch.recvTail++;
  } else {
    LOG_MOD(NCCL_MOD, "recv size unmatch actual: %d != expected: %d", size,
            ch.recvSizes[ch.recvTail]);
  }
  LOG_MOD(NCCL_MOD, "modCoordinatorRecv: size=%d", size);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclModSync, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclModSync(ncclComm_t comm, cudaStream_t stream) {
  if (KERNEL_BYPASS) {
    LOG_MOD(NCCL_MOD, "ncclModSync");
    while (global_coordinator.done == 0) {
      sched_yield();
      check_done(&global_coordinator);
    }
  }
  return ncclSuccess;
}