#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <iostream>
#include <string.h>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <condition_variable>


class nn_progress_bar{

uint m_unCurrVal = 0;
uint m_unMaxVal = 0;
char* m_parrValName = NULL;
bool m_bStopped = false;
bool m_bUpdating = false;

std::mutex m_mtx;
std::condition_variable m_CondVar;

public:

    nn_progress_bar(const char* const_parrcValName, uint unVal);

    ~nn_progress_bar();

    void print_progress_bar(uint unVal);

    void print_progress_bar_periodic(uint unVal, uint unMs);

    void update_progress_bar(uint unVal);

    void reset();

    void stop();

    void setMax(uint mVal);

};
#endif