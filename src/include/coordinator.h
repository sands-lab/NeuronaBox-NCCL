#ifndef MOD_COORDINATOR_H_
#define MOD_COORDINATOR_H_

const int KERNEL_BYPASS = 1;


struct modCoordinator {
    int send;
    int recv;
    int total;
};

extern modCoordinator coordinator;

#endif