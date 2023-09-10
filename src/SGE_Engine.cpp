/**
 * \name    SGE_Engine.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 8, 2023
 */

#include "include/InferenceEngine.hpp"

/** ===============================================================================================
 * \name    SGE_Engine
 *
 * \brief   Construct an inference engine 
 * 
 * \param   SE a SensingEngine as the input source
 * ================================================================================================
 */
SGE_Engine::SGE_Engine(SensingEngine* SE) : InferenceEngine(SE)
{
    registerModels();
}


/** ===============================================================================================
 * \name    registerModels
 * 
 * \brief   Register all used model
 * ================================================================================================
 */
void 
SGE_Engine::registerModels (void)
{
#if MODEL_MASK & YOLONET_256_256
    log_D("SGE_Engine", "Create model: yolov7-tiny_256_256");
    models.emplace_back(new OnnxYoloNet("yolov7-tiny_256_256", 4));
#endif
#if MODEL_MASK & YOLONET_384_384
    log_D("SGE_Engine", "Create model: yolov7-tiny_384_384");
    models.emplace_back(new OnnxYoloNet("yolov7-tiny_384_384", 4));
#endif
#if MODEL_MASK & YOLONET_512_512
    log_D("SGE_Engine", "Create model: yolov7-tiny_512_512");
    models.emplace_back(new OnnxYoloNet("yolov7-tiny_512_512", 4));
#endif
#if MODEL_MASK & YOLONET_640_640
    log_D("SGE_Engine", "Create model: yolov7-tiny_640_640");
    models.emplace_back(new OnnxYoloNet("yolov7-tiny_640_640", 4));
#endif
}


/** ===============================================================================================
 * \name    dataPreprocessor
 * 
 * \brief   Perform data preprocessor if need
 * ================================================================================================
 */
void 
SGE_Engine::dataPreprocessor(void)
{
    log_D("SGE_Engine", "dataPreprocessor");
    for(auto model : models)
    {
        Inference_Task_t task = {(void*)&mImg, -1, model};
        taskQueue.emplace_back(task);
    }
}


/** ===============================================================================================
 * \name    Inference_sched
 * 
 * \brief   Schedule the input datas into prioritied task
 * ================================================================================================
 */
void 
SGE_Engine::Inference_sched (void)
{

}


/** ===============================================================================================
 * \name    onInference
 * 
 * \brief   Keep inference util the meet deadline
 * ================================================================================================
 */
void 
SGE_Engine::onInference (timeval frameStart)
{
    struct timeval now;
    gettimeofday(&now, NULL);
    float spendTime = (1000000 * (now.tv_sec - frameStart.tv_sec) + (now.tv_usec - frameStart.tv_usec)) * 0.001;

    vector<pthread_t*> waitingThreads;
    while(SENSING_PERIOD - spendTime > 0 && taskQueue.size() > 0)
    {
        log_D("SGE_Engine", "Task queue size: " + to_string(taskQueue.size()));

        auto it = taskQueue.begin();
        Inference_Task_t task = *it;
        vector<float> dataStream(task.model->singleInputSize);
        task.model->dataPreprocess(task.data, &dataStream);
        task.model->Onnx_addInput(dataStream);
        // task.model->Onnx_inference();
        
        pthread_create(&task.model->mthread, NULL, threadInference, (void*)task.model);
        waitingThreads.push_back(&task.model->mthread);

        taskQueue.erase(it);

        gettimeofday(&now, NULL);
        spendTime = (1000000 * (now.tv_sec - frameStart.tv_sec) + (now.tv_usec - frameStart.tv_usec)) * 0.001;
    }

    for(auto thread: waitingThreads)
    {
        pthread_join((*thread), NULL);
    }
}