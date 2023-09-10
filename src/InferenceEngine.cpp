/**
 * \name    InferenceEngine.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 8, 2023
 */

#include "include/InferenceEngine.hpp"

/** ===============================================================================================
 * \name    InferenceEngine
 *
 * \brief   Construct an inference engine 
 * 
 * \param   SE a SensingEngine as the input source
 * ================================================================================================
 */
InferenceEngine::InferenceEngine(SensingEngine* SE) : mSE(SE)
{
    
}


/** ===============================================================================================
 * \name    run
 * 
 * \brief   start the inference engine
 * ================================================================================================
 */
void
InferenceEngine::run (void)
{
    struct timeval start, end;
    for (int frameId = 0; frameId < FRAME_NUM; frameId++)
    {
        log_I("main", "Start frame: " + to_string(frameId) + "-----------------");

        gettimeofday(&start, NULL);

            onSyncData();

            Inference_sched();

            onInference(start);

        gettimeofday(&end, NULL);

        float spendTime = (1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001;
        log_I("InferenceEngine", "Inference spend: " + to_string(spendTime) + " ms");
    }

    stop();
}


/** ===============================================================================================
 * \name    stop
 * 
 * \brief   stop the inference engine
 * ================================================================================================
 */
void
InferenceEngine::stop (void)
{
    for(auto model: models)
    {
        model->~OnnxModel();
    }

}


/** ===============================================================================================
 * \name    onSyncData
 * 
 * \brief   Load the data from the SensingEngine, and enable next sensing cycle.
 * ================================================================================================
 */
void 
InferenceEngine::onSyncData (void)
{    
    struct timeval start, end;
    gettimeofday(&start, NULL);
        while(!mSE->readyToSync());
        mImg = mSE->getCameraData();
        mLidarPoints.clear();
        mLidarPoints = mSE->getLidarData();
        log_D("onSyncData", "Image width: " + to_string(mImg.cols) + ", Image height: " + to_string(mImg.rows));
        log_D("onSyncData", "Lidar count: " + to_string(mLidarPoints.size()));
        mSE->startNextSensing();
    gettimeofday(&end, NULL);
    float spendTime = (1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001;
    log_I("InferenceEngine", "Data sync spend: " + to_string(spendTime) + " ms");

    dataPreprocessor();
}


/** ===============================================================================================
 * \name    registerModels
 * 
 * \brief   Register all used model
 * ================================================================================================
 */
void 
InferenceEngine::registerModels (void)
{
    log_V("InferenceEngine", "Base class function: registerModels");
}


/** ===============================================================================================
 * \name    dataPreprocessor
 * 
 * \brief   Perform data preprocessor if need
 * ================================================================================================
 */
void 
InferenceEngine::dataPreprocessor(void)
{
    log_W("InferenceEngine", "Not implement function: dataPreprocessor");
}


/** ===============================================================================================
 * \name    Inference_sched
 * 
 * \brief   Schedule the input datas into prioritied task
 * ================================================================================================
 */
void 
InferenceEngine::Inference_sched (void)
{

}


/** ===============================================================================================
 * \name    onInference
 * 
 * \brief  Keep inference util the meet deadline
 * ================================================================================================
 */
void 
InferenceEngine::onInference (timeval frameStart)
{

}


/** ===============================================================================================
 * \name    threadInference
 * 
 * \brief   Inference model in thread
 * 
 * \param   arg the pointer of the OnnxModel going to inference
 * ================================================================================================
 */
void* 
InferenceEngine::threadInference (void* arg)
{
    OnnxModel* model = (OnnxModel*) arg;

    log_I(model->modelName, "ThreadInference start");
    model->Onnx_inference();

    pthread_exit(nullptr);
}