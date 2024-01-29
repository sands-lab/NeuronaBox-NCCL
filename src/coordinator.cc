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

int KERNEL_BYPASS = -1;

modCoordinator global_coordinator;

modTopology global_topology;

ncclResult_t modTopologyInit(modTopology *topology, ncclProxyOp *proxyOp,
                             ncclInfo *info) {
  ncclComm *comm = info->comm;
  int nranks = comm->nRanks;
  int myrank = comm->rank;
  int nchannels = info->nChannels;
  LOG_MOD(NCCL_MOD, "modTopologyInit %d, ringmapsize=%lu, inited:%d", myrank,
          topology->ringmap.size(), topology->init);
  if (!topology->init) {

    topology->nranks = nranks;
    topology->nnodes =
        atoi(getenv("N_MPI_RANKS")); // should be set by application!
    topology->nrankpernode = topology->nranks / topology->nnodes;
    topology->nchannels = nchannels;
    assert(topology->nranks % topology->nnodes == 0);
    topology->myranks = vector<int>();
    topology->init = 1;
  }

  topology->myranks.push_back(myrank);
  return ncclSuccess;
}

ncclResult_t modTopologyUpdateMap(modTopology *topology, int rank, int channel,
                                  ncclRing *ring, int *ringranks, int nranks) {
  topology->prev[rank] = ring->prev;
  topology->next[rank] = ring->next;
  topology->ringmap[make_pair(rank, channel)] = ring->index;

  if (rank == 0) {
    for (int i = 1; i < nranks; ++i) {
      topology->ringmap[make_pair(i, channel)] = ringranks[i];
      LOG_MOD(NCCL_MOD, "update ringmap rk%d ringidx%d ch%d", i,
              topology->ringmap[make_pair(i, channel)], channel);
    }
  }
  LOG_MOD(NCCL_MOD, "modTopologyUpdateMap [%d->%d->%d], mapsize=%lu",
          topology->prev[rank], rank, topology->next[rank],
          topology->ringmap.size());
  return ncclSuccess;
}

static int getKernelBypass() {
  LOG_MOD(NCCL_MOD, "getKernelBypass called");
  if (KERNEL_BYPASS != -1) {
    return KERNEL_BYPASS;
  }
  char *env = getenv("NCCL_KERNEL_BYPASS");
  if (env == NULL) {
    KERNEL_BYPASS = 0;
  } else {
    KERNEL_BYPASS = atoi(env);
  }
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

// calculate the expected send size for myrank
// this is also used to calculated the recv size for the rank that will
// receive from myrank
static void calc_size_channel(int nranks, int ringindex, int count,
                              int nchannels, int mychannel, int nthreads,
                              int tsize, vector<int> &res) {
  const int chunkSize = 524288;
  int bid = mychannel;
  int loopSize = nchannels * nranks * chunkSize;
  int size = count;
  int ringIx = ringindex;

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
  for (int i = 0; i < res.size(); i++) {
    res[i] *= tsize;
  }
}

static void calc_sendsize_channel(int nranks, int myrank, int count,
                                  int nchannels, int mychannel, int nthreads,
                                  int tsize, vector<int> &res) {
  auto &ringmap = global_topology.ringmap;
  assert(ringmap.count(make_pair(myrank, mychannel)) > 0);
  int myringix = ringmap[make_pair(myrank, mychannel)];
  calc_size_channel(nranks, myringix, count, nchannels, mychannel, nthreads,
                    tsize, res);
  std::string szs;
  for (int i = 0; i < res.size(); ++i) {
    szs = szs + " " + std::to_string(res[i]);
  }
  LOG_MOD(NCCL_MOD, "Calculated send sizes for ringix:%d rank %d: %s, ch=%d",
          myringix, myrank, szs.c_str(), mychannel);
}

static void calc_recvsize_channel(int nranks, int myrank, int count,
                                  int nchannels, int mychannel, int nthreads,
                                  int tsize, vector<int> &res) {
  auto &ringmap = global_topology.ringmap;
  assert(ringmap.count(make_pair(myrank, mychannel)) > 0);
  int target = global_topology.prev[myrank];
  int target_ringix = ringmap[make_pair(target, mychannel)];
  calc_size_channel(nranks, target_ringix, count, nchannels, mychannel,
                    nthreads, tsize, res);
  std::string szs;
  for (int i = 0; i < res.size(); ++i) {
    szs = szs + " " + std::to_string(res[i]);
  }
  LOG_MOD(NCCL_MOD,
          "Calculated recv sizes for ringix:%d targetrk:%d, rank %d: %s, ch=%d",
          target_ringix, target, myrank, szs.c_str(), mychannel);
}

static int check_done_ch(modChannelInfo &ch) {
  if (ch.sendTail == ch.sendSizes.size() &&
      ch.recvTail == ch.recvSizes.size()) {
    return 1;
  }
  return 0;
}

static int update_done_rank(modRankInfo &rank) {
  if (rank.done) {
    return 1;
  }
  int done = 1;
  for (int i = 0; i < rank.channels.size(); ++i) {
    done = done & check_done_ch(rank.channels[i]);
  }
  rank.done = done;
  LOG_MOD(NCCL_MOD, "rank update check_done_rank: done=%d, rank=%d", done,
          rank.myrank);
  return done;
}

static void update_done(modCoordinator *coordinator) {
  if (coordinator->done) {
    return;
  }
  int done = 1;
  for (int i = 0; i < coordinator->ranks.size(); ++i) {
    done = done & update_done_rank(coordinator->ranks[i]);
  }
  if (done) {
    coordinator->done = 1;
    LOG_MOD(NCCL_MOD, "coordinator update check_done: done");
  }
}

static void rankInit(modCoordinator *coordinator, int rank) {
  int nchannels = coordinator->task.nchannels;
  int nthreads = coordinator->task.nthreads;
  int nranks = coordinator->comm.nranks;
  int count = coordinator->task.count;

  modRankInfo &rankinfo = coordinator->ranks[rank];
  rankinfo.myrank = rank;
  rankinfo.send = 0;
  rankinfo.recv = 0;
  if (rankinfo.myrank == coordinator->sendrank) {
    rankinfo.send = 1;
  }
  if (rankinfo.myrank == coordinator->recvrank) {
    rankinfo.recv = 1;
  }
  LOG_MOD(NCCL_MOD, "rankInit: myrank=%d, send=%d, recv=%d", rankinfo.myrank,
          rankinfo.send, rankinfo.recv);
  rankinfo.channels = vector<modChannelInfo>();
  for (int i = 0; i < nchannels; ++i) {
    modChannelInfo ch;
    ch.bid = i;
    ch.sendSizes = vector<int>();
    ch.recvSizes = vector<int>();
    ch.send = rankinfo.send;
    ch.recv = rankinfo.recv;
    if (rankinfo.send) {
      calc_sendsize_channel(nranks, rankinfo.myrank, count, nchannels, i,
                            nthreads, sizeof(float), ch.sendSizes);
    }
    if (rankinfo.recv) {
      calc_recvsize_channel(nranks, rankinfo.myrank, count, nchannels, i,
                            nthreads, sizeof(float), ch.recvSizes);
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

    delete coordinator->proxyOp;
    coordinator->proxyOp = new ncclProxyOp;
    *coordinator->proxyOp = *proxyOp;
    delete coordinator->info;
    coordinator->info = new ncclInfo;
    *coordinator->info = *info;

    modTaskInfo task;
    task.count = info->count;
    task.tsize = sizeof(float);
    task.primitive = 0;
    task.reduceOp = 0;
    task.algo = 0;
    task.nchannels = info->nChannels;
    task.nthreads = info->nThreads;

    modCommInfo comm;
    comm.nranks = info->comm->nRanks;
    comm.nnodes = atoi(getenv("N_MPI_RANKS")); // should be set by application!
    comm.mynode = atoi(getenv("MY_MPI_RANK")); // should be set by application!
    comm.nrankpernode = comm.nranks / comm.nnodes;
    assert(comm.nranks % comm.nnodes == 0);

    coordinator->comm = comm;
    coordinator->task = task;
    coordinator->ranks = map<int, modRankInfo>();
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
  map<int, bool> ismynode;
  auto &g = global_topology;
  for (int i = 0; i < g.nranks; ++i) {
    ismynode[i] = false;
  }
  for (int i = 0; i < g.myranks.size(); ++i) {
    ismynode[g.myranks[i]] = true;
  }
  coordinator->sendrank = -1;
  coordinator->recvrank = -1;
  for (auto i : g.myranks) {
    auto prev = g.prev[i];
    auto next = g.next[i];
    LOG_MOD(NCCL_MOD, "rank=%d, prev=%d, next=%d, ismynode[rank]=%d", i, prev,
            next, (int)ismynode[i]);
    if (ismynode[prev] && !ismynode[next]) {
      assert(coordinator->sendrank == -1);
      coordinator->sendrank = i;
    }
    if (!ismynode[prev] && ismynode[next]) {
      assert(coordinator->recvrank == -1);
      coordinator->recvrank = i;
    }
  }
  if (coordinator->sendrank != -1 && coordinator->recvrank != -1) {
    LOG_MOD(NCCL_MOD,
            "sendrecv solved: sendrank=%d, recvrank=%d, ringmapsize=%lu",
            coordinator->sendrank, coordinator->recvrank, g.ringmap.size());
    for (auto i : g.myranks) {
      rankInit(coordinator, i);
    }
  }
  return ncclSuccess;
}

ncclResult_t modCoordinatorDestroy(modCoordinator *coordinator) {
  coordinator->init = 0;
  coordinator->done = 0;
  coordinator->sendrank = -1;
  coordinator->recvrank = -1;

  LOG_MOD(NCCL_MOD, "modCoordinatorDestroy");
  return ncclSuccess;
}

ncclResult_t modCoordinatorGetSendSize(modCoordinator *coordinator, int cid,
                                       int &size) {
  auto &ch = coordinator->ranks[coordinator->sendrank].channels[cid];
  auto &chrecv = coordinator->ranks[coordinator->recvrank].channels[cid];
  if (ch.sendTail <= chrecv.recvTail) {
    size = ch.sendSizes[ch.sendTail];
  } else {
    size = -1;
    LOG_MOD(NCCL_MOD, "sendTail=%d > recvTail=%d", ch.sendTail, ch.recvTail);
  }
    LOG_MOD(NCCL_MOD, "modCoordinatorGetSendSize: size=%d", size);
    return ncclSuccess;
}

ncclResult_t modCoordinatorSend(modCoordinator *coordinator, int cid,
                                int size) {
  auto &ch = coordinator->ranks[coordinator->sendrank].channels[cid];
  if (ch.sendSizes[ch.sendTail] == size) {
    ch.sendTail++;
    update_done(coordinator);
  } else {
    LOG_MOD(NCCL_MOD, "send size unmatch actual: %d != expected: %d", size,
            ch.sendSizes[ch.sendTail]);
  }
  LOG_MOD(NCCL_MOD, "modCoordinatorSend: size=%d, tail=%d", size, ch.sendTail);
  return ncclSuccess;
}

ncclResult_t modCoordinatorRecv(modCoordinator *coordinator, int cid,
                                int size) {
  auto &ch = coordinator->ranks[coordinator->recvrank].channels[cid];
  if (ch.recvSizes[ch.recvTail] == size) {
    ch.recvTail++;
    update_done(coordinator);
  } else {
    LOG_MOD(NCCL_MOD, "recv size unmatch actual: %d != expected: %d", size,
            ch.recvSizes[ch.recvTail]);
  }
  LOG_MOD(NCCL_MOD, "modCoordinatorRecv: size=%d, recvtail=%d", size,
          ch.recvTail);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclModSync, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclModSync(ncclComm_t comm, cudaStream_t stream) {
  if (KERNEL_BYPASS) {
    LOG_MOD(NCCL_MOD, "ncclModSync");
    while (global_coordinator.done == 0) {
      usleep(100);
    }
    modCoordinatorDestroy(&global_coordinator);
  }
  return ncclSuccess;
}