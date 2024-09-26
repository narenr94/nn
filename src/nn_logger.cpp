#include "nn_logger.h"

static FILE* log_file; //fd for log file

static bool log_initialized = false; //is logs initialized?

static elog_level m_log_level = eLOGLEVEL_ERROR; //log level to dump, default error level

static bool m_print_console = false; //does log need to be printed in concole too?

static const char * nn_log_file = "./nn.log"; //log file location

static const char *mLogLevelStr[eLOGLEVEL_TRACE+1] =
{
	"ERROR", // eLOGLEVEL_ERROR
	"MIL", // eLOGLEVEL_MIL
	"WARN",  // eLOGLEVEL_WARN
	"INFO",  // eLOGLEVEL_INFO
	"DEBUG",   // eLOGLEVEL_DEBUG
	"TRACE", // eLOGLEVEL_TRACE
};

elog_level get_log_level()
{
    return m_log_level;
}


void init_nn_logger(uint ll, bool printConsole)
{

    m_log_level = (elog_level)ll;

    m_print_console = printConsole;

    log_initialized = true;

}



void logprintf(elog_level logLevelIndex, const char* func, int line, const char *format, ...)
{
    if(log_initialized)
    {
        va_list args;
        va_start(args, format);
        char gDebugPrintBuffer[MAX_DEBUG_LOG_BUFF_SIZE];
        int len_header = snprintf(gDebugPrintBuffer, sizeof(gDebugPrintBuffer),
                                "[%s][%s][%d]",
                                mLogLevelStr[logLevelIndex],
                                func,
                                line);
        int len_message = 0;
        if ((uint)len_header >= (uint)sizeof(gDebugPrintBuffer))
        { // Header is too long to print in one log line, no space left for the message
        }
        else
        {
            if (len_header < 0)
            { // Encoding error, let's print only the message
                len_header = 0;
            }

            len_message = vsnprintf(gDebugPrintBuffer+len_header, MAX_DEBUG_LOG_BUFF_SIZE-len_header, format, args);
            if (len_message < 0)
            { // Encoding error, let's print only the header
                len_message = 0;
            }
        }
        int len_total = len_header + len_message;
        if ((uint)len_total >= (uint)sizeof(gDebugPrintBuffer))
        {
            // If the log line is too long, truncate it and add the long line suffix at the end
            (void)snprintf(gDebugPrintBuffer + MAX_DEBUG_LOG_BUFF_SIZE - sizeof(LONG_LINE_SUFFIX), sizeof(LONG_LINE_SUFFIX), LONG_LINE_SUFFIX);
        }
        va_end(args);

        if(m_print_console)
        {
            std::cout << gDebugPrintBuffer << std::endl;
        }

        //open log file and write
        log_file = fopen(nn_log_file, (log_initialized ? "a" : "w"));

        if(log_file)
        {
            fprintf(log_file, "%s\n", gDebugPrintBuffer);
            fclose(log_file);
        }

        
        
    }


}

nn_progress_bar::nn_progress_bar(const char* vName, uint mVal)
{
    uint size = 0;
    size = (uint)strlen(vName);

    std::lock_guard<std::mutex> lock(m_mtx);

    if(m_parrValName)
    {
        free(m_parrValName);
    }
    m_parrValName = (char *)malloc(size);

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
