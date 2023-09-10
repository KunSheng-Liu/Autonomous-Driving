/**
 * \name    SensingEngine.hpp
 * 
 * \brief   Declare the perception sensor behavier
 * 
 * \date    Mar 7, 2023
 */

#ifndef _SENSING_ENGINE_HPP_
#define _SENSING_ENGINE_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "Log.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <assert.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;


/** ===============================================================================================
 * \name    SensingEngine
 * 
 * \brief   The class for handling the peripheral sensor
 * ================================================================================================
 */
class SensingEngine
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    SensingEngine ();

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void run (void);
    void stop (void);
    void startNextSensing (void);

    bool readyToSync (void) {return dataReadyToSync;}
    cv::Mat getCameraData (void) {return cameraData;}
    vector<pair<pair<int, int>, float>> getLidarData (void) {return LidarData;}

private:
    static void* threadSensing (void* arg);
    void Sensing_Camera (string filePath);
    void Sensing_Lidar (string filePath);


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    bool dataReadyToSync;

    pthread_t mthread;

    cv::Mat cameraData;
    vector<pair<pair<int, int>, float>> LidarData;


};

#endif