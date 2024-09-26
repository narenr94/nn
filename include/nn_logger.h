#ifndef NN_LOGGER
#define NN_LOGGER

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <iostream>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "nn_math.h"

#define MAX_DEBUG_LOG_BUFF_SIZE 512

#define MAX_DUMP_FILE_SIZE 10000000 //10MBytes

#define	LONG_LINE_SUFFIX "(...)"


/*
    list of log levels
*/
enum elog_level{
eLOGLEVEL_ERROR,
eLOGLEVEL_MIL,
eLOGLEVEL_WARN,
eLOGLEVEL_INFO,
eLOGLEVEL_DEBUG,
eLOGLEVEL_TRACE
};

/**
    NN logging defines
*/
#define NNLOG_TRACE(FORMAT, ...) NNLOG(eLOGLEVEL_TRACE, FORMAT, ##__VA_ARGS__)
#define NNLOG_DEBUG(FORMAT, ...) NNLOG(eLOGLEVEL_DEBUG, FORMAT, ##__VA_ARGS__)
#define NNLOG_INFO(FORMAT, ...)  NNLOG(eLOGLEVEL_INFO, FORMAT, ##__VA_ARGS__)
#define NNLOG_WARN(FORMAT, ...)  NNLOG(eLOGLEVEL_WARN, FORMAT, ##__VA_ARGS__)
#define NNLOG_MIL(FORMAT, ...)   NNLOG(eLOGLEVEL_MIL, FORMAT, ##__VA_ARGS__)
#define NNLOG_ERR(FORMAT, ...)   NNLOG(eLOGLEVEL_ERROR, FORMAT, ##__VA_ARGS__)


/*
    NNLOG : print nnlog of mentioned level
    @level : gives priority for the logging
    @FORMAT is standard printf style format string followed by arguments
*/
#define NNLOG(LEVEL, FORMAT, ... ) \
do { \
    if( LEVEL <= get_log_level() ) \
	{ \
		logprintf( LEVEL, __FUNCTION__, __LINE__, FORMAT, ##__VA_ARGS__); \
	} \
} while(0)




/*
    init_nn_logger : initialize logger
    @ll : log level to be set
*/
void init_nn_logger(uint unLogLevel, bool bPrintConsole);



/*
    logprintf : print to log file
    @logLevel : log level to be printed at
    @func : name of func from where log is printed
    @line : line number from where log is printed
    @format : standard printf style format string followed by arguments
*/
void logprintf(elog_level eLogLevel, const char* const_parrcFunc, int nLine, const char* const_parrFormat, ...);

/*
    get_log_level: returns log level
*/
elog_level get_log_level();


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

    void print_progress_bar(uint unVal);

    void print_progress_bar_periodic(uint unVal, uint unMs);

    void update_progress_bar(uint unVal);

    void reset();

    void stop();

    void setMax(uint mVal);

};
#endif