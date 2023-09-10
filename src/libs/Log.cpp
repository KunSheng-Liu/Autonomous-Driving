/**
 * \name    Log.cpp
 * 
 * \brief   Passing the log information by the logType
 * 
 * \date    Mar 6, 2023
 */

#include "../include/Log.hpp"

/* ************************************************************************************************
 * Global Resource
 * ************************************************************************************************
 */
pthread_mutex_t ioMutex;
std::ofstream logFile;

/* ************************************************************************************************
 * Global Resource
 * ************************************************************************************************
 */
void logInit (void)
{
    /* I/O protector */
    pthread_mutex_init(&ioMutex, NULL);

    /* Log file */
    logFile.open("log.txt");
}

void logDestory (void)
{
    pthread_mutex_destroy(&ioMutex);
    logFile.close();
}


/** ===============================================================================================
 * \name    log
 *
 * \brief   Print the predefined information by the corresponding condition
 * 
 * \param   tag the group of information
 * \param   logType the element of enum Log_t
 * ================================================================================================
 */
void log (std::string tag, Log_t logType)
{
    switch(logType)
    {
        case ONNX_SETUPMODEL_START:
            log_I(tag, "Start up model...");
            break;

        case ONNX_SETUPMODEL_WARMUP:
            log_I(tag, "Warm up model...");
            break;

        case ONNX_INFERENCE_INPUTSIZE_ZERO:
            log_D(tag, "Input size is zero, skip inference.");
            break;

        case ONNX_INFERENCE_INPUTSIZE_WRONG:
            log_E(tag, "Input size not match, skip inference.");
            break;
    }
}