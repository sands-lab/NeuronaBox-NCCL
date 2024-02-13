#include "align.h"
#include "comm.h"
#include "emulator.h"
#include "helper.h"
#include <assert.h>
#include <cinttypes>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <string>
using namespace std;

modController global_controller;

static uint64_t gen_unique_id() {
  static uint64_t unique_id = 0;
  return unique_id++;
}

ncclResult_t modControllerAddTask(modController *controller, ncclInfo *info) {
  modTaskInfo task;
  info->unique_id = gen_unique_id();
  Info2Task(info, &task);
  assert(controller->taskMap.count(task.unique_id) > 0);
  controller->taskMap[task.unique_id] = task;
  return ncclSuccess;
}

ncclResult_t modControllerQueryTask(modController *controller,
                                    uint64_t unique_id, modTaskInfo *task) {
  auto it = controller->taskMap.find(unique_id);
  if (it != controller->taskMap.end()) {
    *task = it->second;
    return ncclSuccess;
  } else {
    LOG_MOD(NCCL_LOG_WARN, "modControllerQueryTask: task not found");
    return ncclSuccess;
  }
}

ncclResult_t modControllerRemoveTask(modController *controller,
                                     uint64_t unique_id) {
  if (controller->taskMap.count(unique_id) > 0) {
    controller->taskMap.erase(unique_id);
    return ncclSuccess;
  } else {
    LOG_MOD(NCCL_LOG_WARN, "modControllerRemoveTask: task not found");
    abort();
    return ncclSuccess;
  }
}

ncclResult_t modControllerCheck(modController *controller, uint64_t unique_id,
                                int &admit) {
  assert(controller->taskMap[unique_id].count() > 0);
  auto task = controller->taskMap[unique_id];
  admit = task->primitive == ncclFuncAllReduce;
  return ncclSuccess;
}