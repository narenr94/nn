#include "nn_logger.h"


nn_progress_bar::~nn_progress_bar()
{
    delete [] m_parrValName;
}

nn_progress_bar::nn_progress_bar(const char* vName, uint mVal)
{
    uint size = 0;
    size = (uint)strlen(vName);

    std::lock_guard<std::mutex> lock(m_mtx);

    if(m_parrValName)
    {
        free(m_parrValName);
        m_parrValName = nullptr;
    }
    // m_parrValName = (char *)malloc(size);
    m_parrValName = new char[size];

    strcpy(m_parrValName, vName);

    m_unCurrVal = 0;

    m_unMaxVal = mVal;

    m_bStopped = false;
}

void nn_progress_bar::print_progress_bar(uint cVal)
{
    uint percentComplete = 0;
    uint i = 0;
    std::unique_lock<std::mutex>lock(m_mtx);
    do
    {
        std::cout<<"\r["<<m_parrValName<<":"<<m_unCurrVal<<"/"<<m_unMaxVal<<"]";
        percentComplete = m_unCurrVal * 100;
        percentComplete /= m_unMaxVal;

        for(i = 0; i < percentComplete; i++)
        {
            std::cout<<"#";
        }

        for(i = 0; i < (100 - percentComplete); i++)
        {
            std::cout<<" ";
        }

        std::cout<<"["<<percentComplete<<"%]";

        std::cout<<std::flush;

        m_bUpdating = false;

        m_CondVar.wait(lock);

    } while (!m_bStopped);

    std::cout<<"\n";
    
    lock.unlock();

    return;
}

void nn_progress_bar::print_progress_bar_periodic(uint cVal, uint ms)
{
    uint percentComplete = 0;
    uint i = 0;
    std::unique_lock<std::mutex>lock(m_mtx);
    do
    {
        
        std::cout<<"\r["<<m_parrValName<<":"<<m_unCurrVal<<"/"<<m_unMaxVal<<"]";
        percentComplete = m_unCurrVal * 100;
        percentComplete /= m_unMaxVal;

        for(i = 0; i < percentComplete; i++)
        {
            std::cout<<"#";
        }

        for(i = 0; i < (100 - percentComplete); i++)
        {
            std::cout<<" ";
        }

        std::cout<<"["<<percentComplete<<"%]";

        std::cout<<std::flush;

        m_bUpdating = false;

        lock.unlock();
        usleep(ms * 1000);
        lock.lock();

    } while (!m_bStopped);

    std::cout<<"\n";
    
    lock.unlock();

    return;
}

void nn_progress_bar::update_progress_bar(uint cVal)
{
    std::unique_lock<std::mutex>lock(m_mtx);
    m_unCurrVal = cVal;
    m_bUpdating = true;
    m_CondVar.notify_one();
}

void nn_progress_bar::reset()
{
    while(m_bUpdating); //wait until current update of progress bar completes

    std::unique_lock<std::mutex>lock(m_mtx);
    m_unCurrVal = 0;
    m_bStopped = false;
}

void nn_progress_bar::stop()
{
    std::unique_lock<std::mutex>lock(m_mtx);
    m_bStopped = true;
    m_CondVar.notify_one();
}

void nn_progress_bar::setMax(uint mVal)
{
    std::unique_lock<std::mutex>lock(m_mtx);
    m_unMaxVal = mVal;
}
