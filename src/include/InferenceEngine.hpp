/**
 * \name    InferenceEngine.hpp
 * 
 * \brief   Declare the NN inference behavier
 * 
 * \date    Mar 8, 2023
 */

#ifndef _INFERENCE_ENGINE_HPP_
#define _INFERENCE_ENGINE_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */

#include "App_config.hpp"
#include "Log.hpp"
#include "OnnxModels.hpp"
#include "SensingEngine.hpp"

#include <algorithm>
// #include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <assert.h>
#include <sys/time.h>
#include <onnxruntime/session/onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;


/** ===============================================================================================
 * \name    InferenceEngine
 * 
 * \brief   The base class of inference engine. You can add new engine by inheritance this class and
 *          override the virtual function to implement the approach.
 * ================================================================================================
 */
class InferenceEngine
{
/** ***********************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    InferenceEngine(SensingEngine* SE);

/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
protected:
    typedef struct {
        void*           data;
        float           priority;
        OnnxModel*      model;
    }Inference_Task_t;

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void run (void);
    void stop (void);

protected:
    void onSyncData (void);
    static void* threadInference (void* arg);
    virtual void registerModels (void);
    virtual void dataPreprocessor(void);
    virtual void Inference_sched (void);
    virtual void onInference (timeval frameStart);

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:

protected:
    SensingEngine*                          mSE;
    cv::Mat                                 mImg;
    vector<pair<pair<int, int>, float>>     mLidarPoints;
    vector<OnnxModel*>                      models;
    vector<Inference_Task_t>                taskQueue;

};


/** ===============================================================================================
 * \name    CPS_Engine
 * 
 * \brief   The inference engine for implementing paper: Real-Time Task Scheduling for Machine Perception 
 * in Intelligent Cyber-Physical System.
 * ================================================================================================
 */
class CPS_Engine : public InferenceEngine
{
/* ************************************************************************************************
 * Local Configureation
 * ************************************************************************************************
 */
    #define LIDAR_GRADIENT_SENSITIVE        5
    #define LIDAR_MERGING_SENSITIVE         15

/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    CPS_Engine(SensingEngine* SE);


/* ************************************************************************************************
 * Type Define
 * ************************************************************************************************
 */
private:
    typedef struct {
        float left;
        float right;
        float top;
        float bottom;
    } boundingBox_t;


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
private:
    void registerModels (void) override;
    void dataPreprocessor(void) override;
    void Inference_sched (void) override;
    void onInference (timeval frameStart) override;


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    vector<pair<int, int>> imgShapes;
};


/** ===============================================================================================
 * \name    SGE_Engine
 * 
 * \brief   The inference engine for implementing paper: Real-Time Task Scheduling for Machine Perception 
 * in Intelligent Cyber-Physical System.
 * ================================================================================================
 */
class SGE_Engine : public InferenceEngine
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    SGE_Engine(SensingEngine* SE);


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
private:
    void registerModels (void) override;
    void dataPreprocessor(void) override;
    void Inference_sched (void) override;
    void onInference (timeval frameStart) override;

};


#endif