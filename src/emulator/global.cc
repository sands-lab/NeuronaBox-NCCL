#include "align.h"
#include "comm.h"
#include "emulator.h"
#include <assert.h>
#include <cinttypes>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <string>
using namespace std;

int MOD_KERNEL_BYPASS = -1;
int MOD_N_NODES = -1;
int MOD_MY_NODE = -1;
modCoordinator global_coordinator;
modTopology global_topology;

ncclResult_t modGetAllEnvVars() {
  LOG_MOD(NCCL_MOD, "modGetAllEnvVars");
  char *env = getenv("MOD_N_MPI_RANKS");
  if (env == NULL) {
    LOG_MOD(NCCL_LOG_ABORT, "Error: N_MPI_RANKS not set");
    return ncclModError;
  } else {
    MOD_N_NODES = atoi(env);
    LOG_MOD(NCCL_MOD, "MOD_N_MPI_RANKS=%d", MOD_N_NODES);
  }
  env = getenv("MOD_MY_MPI_RANK");
  if (env == NULL) {
    LOG_MOD(NCCL_LOG_ABORT, "Error: MY_MPI_RANK not set");
    return ncclModError;
  } else {
    MOD_MY_NODE = atoi(env);
    LOG_MOD(NCCL_MOD, "MY_MPI_RANK=%d", MOD_MY_NODE);
  }
  env = getenv("MOD_KERNEL_BYPASS");
  if (env == NULL) {
    LOG_MOD(NCCL_MOD, "MOD_KERNEL_BYPASS not set, default to 0");
    MOD_KERNEL_BYPASS = 0;
  } else {
    MOD_KERNEL_BYPASS = atoi(env);
    LOG_MOD(NCCL_MOD, "MOD_KERNEL_BYPASS=%d", MOD_KERNEL_BYPASS);
  }
  return ncclSuccess;
}
