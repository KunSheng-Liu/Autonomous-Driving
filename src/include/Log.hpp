/** 
 * \name    Log.hpp
 * 
 * \brief   Log the information of the application
 * 
 * \date    Mar 6, 2023
 */

#ifndef _LOG_HPP_
#define _LOG_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.hpp"

#include <iostream>
#include <fstream>

/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */
typedef enum {
    ONNX_SETUPMODEL_START               = 0x00,
    ONNX_SETUPMODEL_WARMUP              = 0x01,

    ONNX_INFERENCE_INPUTSIZE_ZERO       = 0x10,
    ONNX_INFERENCE_INPUTSIZE_WRONG      = 0x11
}Log_t;

/* ************************************************************************************************
 * Global Resource
 * ************************************************************************************************
 */
extern pthread_mutex_t ioMutex;
extern std::ofstream logFile;

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
void logInit (void);
void logDestory (void);

void log (std::string tag, Log_t logType);

/* ERROR */
inline void log_E (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= ERROR
    pthread_mutex_lock(&ioMutex);
    std::cout << "\033[1;31mLogE:\033[0m Tag: " << tag << ": " << logInfo << std::endl;
    pthread_mutex_unlock(&ioMutex);
#endif
}

/* WARNNING */
inline void log_W (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= WARNNING
    pthread_mutex_lock(&ioMutex);
    std::cout << "\033[1;34mLogW:\033[0m Tag: " << tag << ": " << logInfo << std::endl;
    pthread_mutex_unlock(&ioMutex);
#endif
}

/* INFO */
inline void log_I (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= INFO
    pthread_mutex_lock(&ioMutex);
    std::cout << "\033[1;32mLogI:\033[0m Tag: " << tag << ": " << logInfo << std::endl;
    logFile << "Tag: " << tag << ": " << logInfo << std::endl;
    pthread_mutex_unlock(&ioMutex);
#endif
}

/* DEBUG */
inline void log_D (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= DEBUG
    pthread_mutex_lock(&ioMutex);
    std::cout << "\033[1;36mLogD:\033[0m Tag: " << tag << ": " << logInfo << std::endl;
    pthread_mutex_unlock(&ioMutex);
#endif
}

/* VERBOSE */
inline void log_V (std::string tag, std::string logInfo)
{
#if LOG_LEVEL >= VERBOSE
    pthread_mutex_lock(&ioMutex);
    std::cout << "LogV: Tag: " << tag << ": " << logInfo << std::endl;
    pthread_mutex_unlock(&ioMutex);
#endif
}

#endif