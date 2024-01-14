#ifndef MOD_COORDINATOR_H_
#define MOD_COORDINATOR_H_

const int KERNEL_BYPASS = 0;


struct modCoordinator {
    int send;
    int recv;
};

extern modCoordinator coordinator;

#endif