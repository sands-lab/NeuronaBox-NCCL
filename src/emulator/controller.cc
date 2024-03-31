#include "align.h"
#include "comm.h"
#include "emulator.h"
#include "helper.h"
#include "nccl.h"
#include <assert.h>
#include <cassert>
#include <cinttypes>
#include <map>
#include <math.h>
#include <sched.h>
#include <stdlib.h>
#include <string>

using namespace std;

modController global_controller;

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

static inline __attribute__((always_inline)) void
calc_sendsize_channel(int nranks, int myrank, int count, int nchannels,
                      int mychannel, int nthreads, int tsize,
                      vector<int> &res) {
  //   auto &ringmap = global_topology.ringmap;
  //   assert(ringmap.count(make_pair(myrank, mychannel)) > 0);
  // int myringix = ringmap[make_pair(myrank, mychannel)];
  //! todo multiple gpu per proc
  int myringix = myrank;
  calc_size_channel(nranks, myringix, count, nchannels, mychannel, nthreads,
                    tsize, res);
  std::string szs;
  for (int i = 0; i < res.size(); ++i) {
    szs = szs + " " + std::to_string(res[i]);
  }
  LOG_MOD(NCCL_MOD, "Calculated send sizes for ringix:%d rank %d: %s, ch=%d",
          myringix, myrank, szs.c_str(), mychannel);
}

static inline __attribute__((always_inline)) void
calc_recvsize_channel(int nranks, int myrank, int count, int nchannels,
                      int mychannel, int nthreads, int tsize,
                      vector<int> &res) {
  //   auto &ringmap = global_topology.ringmap;
  //   assert(ringmap.count(make_pair(myrank, mychannel)) > 0);
  //   int target = global_topology.prev[myrank];
  //   int target_ringix = ringmap[make_pair(target, mychannel)];
  //! todo
  int target = (myrank + 1) % nranks;
  int target_ringix = target;
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

static inline __attribute__((always_inline)) void
channelInit(modChannelInfo *ch, modRankInfo *rankinfo, int nranks, int myrank,
            int chid, int count, int nchannels, int nthreads, int tsize) {
  ch->bid = chid;
  ch->sendsizes = vector<int>();
  ch->recvsizes = vector<int>();
  ch->send = rankinfo->send;
  ch->recv = rankinfo->recv;
  if (rankinfo->send) {
    calc_sendsize_channel(nranks, myrank, count, nchannels, chid, nthreads,
                          sizeof(float), ch->sendsizes);
  }
  if (rankinfo->recv) {
    calc_recvsize_channel(nranks, myrank, count, nchannels, chid, nthreads,
                          sizeof(float), ch->recvsizes);
  }
  ch->sendtail = 0;
  ch->recvtail = 0;
  ch->senddone = 0;
  ch->recvdone = 0;
  LOG_MOD(NCCL_MOD, "channelInit: myrank=%d, chid=%d, send=%d, recv=%d",
          rankinfo->myrank, chid, ch->send, ch->recv);
}

static inline __attribute__((always_inline)) void
rankInit(modRankInfo *rankinfo, modEmulatorTask *task, modCommInfo *comm,
         int rank) {
  int nchannels = task->info.nchannels;
  int nthreads = task->info.nthreads;
  int nranks = comm->nranks;
  int count = task->info.count;
  rankinfo->done = 0;
  rankinfo->myrank = rank;
  rankinfo->send = 0;
  rankinfo->recv = 0;
  if (rankinfo->myrank == task->sendrank) {
    rankinfo->send = 1;
  }
  if (rankinfo->myrank == task->recvrank) {
    rankinfo->recv = 1;
  }
  //! todo consider multiple rank per proc
  assert(rankinfo->send == 1 && rankinfo->recv == 1);
  LOG_MOD(NCCL_MOD, "rankInit: myrank=%d, send=%d, recv=%d", rankinfo->myrank,
          rankinfo->send, rankinfo->recv);
  rankinfo->channels = vector<modChannelInfo>();
  for (int i = 0; i < nchannels; ++i) {
    modChannelInfo ch;
    channelInit(&ch, rankinfo, nranks, rank, i, count, nchannels, nthreads,
                sizeof(float));
    //! todo fix tsize
    rankinfo->channels.push_back(ch);
  }
}

static inline __attribute__((always_inline)) int
bypassCheckInternal(modTaskInfo info, uint64_t unique_id) {
  return MOD_KERNEL_BYPASS == 1 && info.coll == ncclFuncAllReduce &&
         unique_id >= 89; //(39 * 2) + 8;
}

static inline __attribute__((always_inline)) int
emulatorTaskInit(modEmulatorTask *task, modCommInfo *comm, ncclInfo *info) {
  Info2Task(info, &task->info);
  task->init = 1;
  task->done = 0;
  task->ranks = map<int, modRankInfo>();
  task->sendrank = comm->mynode;
  task->recvrank = comm->mynode;
  if (bypassCheckInternal(task->info, task->info.unique_id)) {
    //! fix me here we assume 2 node, 1 rank per node
    for (int i = 0; i < comm->nrankpernode; ++i) {
      int rank = comm->nrankpernode * comm->mynode + i;
      modRankInfo rankinfo;
      rankInit(&rankinfo, task, comm, rank);
      task->ranks[rank] = rankinfo;
      LOG_MOD(NCCL_MOD, "emulatorTaskInit: rank=%d", rank);
    }
    //! todo sendrecv init
  } else {
    LOG_MOD(NCCL_MOD, "emulatorTaskInit: bypassed task, unique_id=%lu",
            task->info.unique_id);
  }
  LOG_MOD(NCCL_MOD, "emulatorTaskInit: unique_id=%lu", task->info.unique_id);
  return 0;
}

static inline __attribute__((always_inline)) int
emulatorTaskDestroy(modEmulatorTask *task) {
  task->init = 0;
  task->done = 0;
  task->ranks.clear();
  task->sendrank = -1;
  task->recvrank = -1;
  return 0;
}

static inline __attribute__((always_inline)) int
check_done_ch(modChannelInfo *ch) {
  return ch->sendtail == ch->sendsizes.size() && ch->senddone;
  return ch->sendtail == ch->sendsizes.size() &&
         ch->recvtail == ch->recvsizes.size() && ch->senddone && ch->recvdone;
}

static inline __attribute__((always_inline)) int
check_done_rank(modRankInfo *rank) {
  if (rank->done) {
    return 1;
  }
  int done = 1;
  for (int i = 0; i < rank->channels.size(); ++i) {
    done = done & check_done_ch(&rank->channels[i]);
  }
  rank->done = done;
  LOG_MOD(NCCL_MOD, "check_done_rank: done=%d, rank=%d", done, rank->myrank);
  return done;
}

static inline __attribute__((always_inline)) int
check_done_task(modEmulatorTask *task) {
  if (task->done) {
    return 1;
  }
  int done = 1;
  for (int i = 0; i < task->ranks.size(); ++i) {
    done = done & check_done_rank(&task->ranks[i]);
  }
  if (done) {
    task->done = 1;
    // printf("[sync] task unique_id = %lu done\n", task->info.unique_id);
    LOG_MOD(NCCL_MOD, "check_done_task: done=%d, unique_id=%lu, rksize=%lu",
            done, task->info.unique_id, task->ranks.size());
  }
  return done;
}

static inline __attribute__((always_inline)) int
syncTask(modEmulatorTask *task) {
  if (task->done == 1) {
    return 1;
  } else {
    check_done_task(task);
    return task->done;
  }
}

ncclResult_t ncclModStreamSyncFunc(modController *controller, cudaStream_t s) {
  if (MOD_KERNEL_BYPASS != 1) {
    LOG_MOD(NCCL_MOD, "ncclModStreamSyncFunc: bypass is off, return");
    return ncclSuccess;
  }
  emulator_lock.lock();
  if (controller->stream2ids.count(s) == 0) {
    LOG_MOD(NCCL_LOG_WARN, "ncclModStreamSyncFunc: stream not found");
    return ncclSuccess;
  }
  static int last = 0;
  auto ids = controller->stream2ids[s];

  // int flag = 1;
  // while (1) {
  //   flag = 1;
  //   for (auto i : ids) {
  //     auto &task = controller->id2task[i];
  //     if (task.done || !task.info.bypass) {
  //       continue;
  //     }
  //     flag = flag & syncTask(&task);
  //   }
  //   if (flag) {
  //     // LOG_MOD(NCCL_MOD, "ncclModStreamSyncFunc: all bypassed tasks are
  //     // done");
  //     break;
  //   } else {
  //     sched_yield();
  //   }
  // }

  vector<uint64_t> rest_ids;
  for (int i = last; i < ids.size(); ++i) {
    rest_ids.push_back(ids[i]);
  }
  int flag = 1;
  while (1) {
    flag = 1;
    for (auto i : rest_ids) {
      assert(controller->id2task.count(i) > 0);
      auto &task = controller->id2task[i];
      if (task.done || task.info.bypass == 0) {
        continue;
      }
      flag = flag & syncTask(&task);
    }
    if (flag) {
      // LOG_MOD(NCCL_MOD, "ncclModStreamSyncFunc: all bypassed tasks are
      // done");
      break;
    } else {
      emulator_lock.unlock();
      sched_yield();
      emulator_lock.lock();
    }
  }
  last = ids.size();
  emulator_lock.unlock();
  // controller->id2task.clear();
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclModStreamSync, cudaStream_t s);
ncclResult_t ncclModStreamSync(cudaStream_t s) {
  return ncclModStreamSyncFunc(&global_controller, s);
}

static uint64_t gen_unique_id() {
  static uint64_t unique_id = 0;
  return ++unique_id;
}

static int gen_stream_id() {
  static int stream_id = 0;
  return ++stream_id;
}

ncclResult_t modAddTask(modController *controller, ncclInfo *info) {
  info->unique_id = gen_unique_id();
  controller->stream2ids[info->stream].push_back(info->unique_id);
  if (controller->stream2int.count(info->stream) == 0) {
    controller->stream2int[info->stream] = gen_stream_id();
  }
  LOG_MOD(NCCL_MOD, "modAddTask for unique_id: %lu in stream %lu",
          info->unique_id, (uint64_t)info->stream);
  return ncclSuccess;
}

ncclResult_t modInitTask(modController *controller, ncclInfo *info) {
  modEmulatorTask task;
  auto unique_id = info->unique_id;
  LOG_MOD(NCCL_MOD, "modInitTask for unique_id: %lu", unique_id);
  assert(controller->id2task.count(unique_id) == 0);
  emulatorTaskInit(&task, controller->comm, info);
  controller->id2task[task.info.unique_id] = task;
  // printf("[emulator] task unique_id = %lu inited\n", task.info.unique_id);
  int nchannels = task.info.nchannels;
  for (int i = 0; i < nchannels; ++i) {
    if (controller->cid2bypassed.count(i) == 0) {
      controller->cid2bypassed[i] = make_pair(0, 0);
    }
  }
  return ncclSuccess;
}

ncclResult_t modQueryTask(modController *controller, uint64_t unique_id,
                          modTaskInfo *task) {
  LOG_MOD(NCCL_MOD, "modQueryTask for unique_id: %lu", unique_id);
  auto it = controller->id2task.find(unique_id);
  if (it != controller->id2task.end()) {
    *task = it->second.info;
    return ncclSuccess;
  } else {
    LOG_MOD(NCCL_LOG_WARN, "modQueryTask: task not found");
    return ncclSuccess;
  }
}

ncclResult_t modRemoveTask(modController *controller, uint64_t unique_id) {
  LOG_MOD(NCCL_MOD, "modRemoveTask for unique_id: %lu", unique_id);
  if (controller->id2task.count(unique_id) > 0) {
    controller->id2task.erase(unique_id);
    return ncclSuccess;
  } else {
    LOG_MOD(NCCL_LOG_WARN, "modRemoveTask: task not found");
    abort();
    return ncclSuccess;
  }
}

ncclResult_t modBypassCheck(modController *controller, uint64_t unique_id,
                            int &bypass, std::string msg) {
  if (controller->id2task.count(unique_id) <= 0) {
    // LOG_MOD(NCCL_MOD, "modBypassCheck: task for unique_id: %lu not
    // recorded!",
    //         unique_id);
    fprintf(stderr,
            "ERRORtask for unique_id: %lu not found from %s, size = %u\n",
            unique_id, msg.c_str(), controller->id2task.size());
  }
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  bypass = bypassCheckInternal(task.info, unique_id);
  task.info.bypass = bypass;
  //! fix me
  LOG_MOD(NCCL_MOD, "modBypassCheck for unique_id: %lu, bypass = %d", unique_id,
          bypass);
  return ncclSuccess;
}

ncclResult_t modGlobalInit(modController *controller, ncclComm *comm) {

  controller->comm = new modCommInfo();
  controller->comm->nranks = comm->nRanks;
  controller->comm->mynode = MOD_MY_NODE;
  controller->comm->nnodes = MOD_N_NODES;
  controller->comm->nrankpernode = comm->nRanks / MOD_N_NODES;
  controller->id2task = map<uint64_t, modEmulatorTask>();
  printf("[emulator] global init done");

  controller->stream2ids = map<cudaStream_t, vector<uint64_t>>();

  controller->coordinator = &global_coordinator;

  controller->topology = &global_topology;
  //! todo init topology and coordinator here!

  return ncclSuccess;
}

__attribute__((always_inline)) int
modProxyGetSendSize(modController *controller, int unique_id, int cid,
                    int &size) {
  LOG_MOD(NCCL_MOD, "modProxyGetSendSize for unique_id: %d, cid: %d, size: %d",
          unique_id, cid, size);
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  auto &rank = task.ranks[task.sendrank];
  auto &recvch = task.ranks[task.recvrank].channels[cid];
  auto &ch = rank.channels[cid];
  if (ch.sendtail <= recvch.recvtail) {
    size = ch.sendsizes[ch.sendtail];
  } else {
    size = -1;
  }
  return 0;
 }

__attribute__((always_inline)) int
modProxySend(modController *controller, int unique_id, int cid, int size) {
   LOG_MOD(NCCL_MOD, "modProxySend for unique_id: %d, cid: %d, size: %d",
           unique_id, cid, size);
   assert(controller->id2task.count(unique_id) > 0);
   auto &task = controller->id2task[unique_id];
   auto &rank = task.ranks[task.sendrank];
   auto &ch = rank.channels[cid];
   assert(ch.sendsizes[ch.sendtail] == size);
   ch.sendtail++;

   return 0;
}

__attribute__((always_inline)) int
modProxyRecv(modController *controller, int unique_id, int cid, int size) {
  LOG_MOD(NCCL_MOD, "modProxyRecv for unique_id: %d, cid: %d, size: %d",
          unique_id, cid, size);
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  auto &rank = task.ranks[task.recvrank];
  auto &ch = rank.channels[cid];
  assert(ch.recvsizes[ch.recvtail] == size);
  ch.recvtail++;
  return 0;
}

__attribute__((always_inline)) int modProxySendDone(modController *controller,
                                                    int unique_id, int cid,
                                                    uint64_t bypassed) {
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  auto &rank = task.ranks[task.sendrank];
  auto &ch = rank.channels[cid];
  assert(ch.sendtail == ch.sendsizes.size() && ch.senddone == 0);
  ch.senddone = 1;
  auto &c = controller->cid2bypassed[cid];
  c.first += bypassed;
  LOG_MOD(NCCL_MOD, "modProxySendDone for unique_id: %d, cid: %d, inc = %lu",
          unique_id, cid, bypassed);
  return 0;
}

__attribute__((always_inline)) int modProxyRecvDone(modController *controller,
                                                    int unique_id, int cid,
                                                    uint64_t bypassed) {
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  auto &rank = task.ranks[task.recvrank];
  auto &ch = rank.channels[cid];
  assert(ch.recvtail == ch.recvsizes.size() && ch.recvdone == 0);
  ch.recvdone = 1;
  auto &c = controller->cid2bypassed[cid];
  c.second += bypassed;
  LOG_MOD(NCCL_MOD, "modProxyRecvDone for unique_id: %d, cid: %d, inc = %lu",
          unique_id, cid, bypassed);
  return 0;
}

__attribute__((always_inline)) int
modProxyBypassedSend(modController *controller, int unique_id, int cid,
                     uint64_t &bypassed) {
  auto &c = controller->cid2bypassed[cid];
  bypassed = c.first;
  return 0;
}

__attribute__((always_inline)) int
modProxyBypassedRecv(modController *controller, int unique_id, int cid,
                     uint64_t &bypassed) {
  auto &c = controller->cid2bypassed[cid];
  bypassed = c.second;
  return 0;
}