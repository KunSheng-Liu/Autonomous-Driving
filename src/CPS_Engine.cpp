/**
 * \name    CPS_Engine.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 8, 2023
 */

#include "include/InferenceEngine.hpp"

/** ===============================================================================================
 * \name    CPS_Engine
 *
 * \brief   Construct an inference engine 
 * 
 * \param   SE a SensingEngine as the input source
 * ================================================================================================
 */
CPS_Engine::CPS_Engine(SensingEngine* SE) : InferenceEngine(SE)
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
CPS_Engine::registerModels (void)
{
#if MODEL_MASK & RESNET_56_56
    imgShapes.emplace_back(make_pair(56, 56));
    models.emplace_back(new OnnxResNet("resnet50_56_56", 4));
#endif
#if MODEL_MASK & RESNET_112_112
    imgShapes.emplace_back(make_pair(112, 112));
    models.emplace_back(new OnnxResNet("resnet50_112_112", 4));
#endif
#if MODEL_MASK & RESNET_168_168
    imgShapes.emplace_back(make_pair(168, 168));
    models.emplace_back(new OnnxResNet("resnet50_168_168", 4));
#endif
#if MODEL_MASK & RESNET_224_224
    imgShapes.emplace_back(make_pair(224, 224));
    models.emplace_back(new OnnxResNet("resnet50_224_224", 2));
#endif
#if MODEL_MASK & RESNET_280_280
    imgShapes.emplace_back(make_pair(280, 280));
    models.emplace_back(new OnnxResNet("resnet50_280_280", 1));
#endif
#if MODEL_MASK & RESNET_336_336
    imgShapes.emplace_back(make_pair(336, 336));
    models.emplace_back(new OnnxResNet("resnet50_336_336", 1));
#endif
#if MODEL_MASK & RESNET_448_448
    imgShapes.emplace_back(make_pair(448, 448));
    models.emplace_back(new OnnxResNet("resnet50_448_448", 1));
#endif
#if MODEL_MASK & RESNET_1280_1920
    imgShapes.emplace_back(make_pair(1280, 1920));
    models.emplace_back(new OnnxResNet("resnet50_1280_1920", 1));
#endif
}


/** ===============================================================================================
 * \name    dataPreprocessor
 * 
 * \brief   Perform data preprocessor if need
 * ================================================================================================
 */
void 
CPS_Engine::dataPreprocessor(void)
{
    log_D("CPS_Engine", "dataPreprocessor");
    struct timeval start, end;
    gettimeofday(&start, NULL);
        /* ******************************************
         * Grouping the ranging points into box
         * ******************************************
         */
        vector<pair<float, boundingBox_t>> obstacles;
        for (auto& point : mLidarPoints) {
            float x = point.first.first;
            float y = point.first.second;
            float distant = point.second;

            bool new_obstacle = true;
            for (auto& obstacle : obstacles) {
                if (abs(obstacle.first - distant) < LIDAR_GRADIENT_SENSITIVE)
                {
                    if ((obstacle.second.top - LIDAR_MERGING_SENSITIVE) < y && y < (obstacle.second.bottom + LIDAR_MERGING_SENSITIVE) && 
                    (obstacle.second.left - LIDAR_MERGING_SENSITIVE) < x && x < (obstacle.second.right + LIDAR_MERGING_SENSITIVE))
                    {
                        obstacle.first = (obstacle.first + distant) / 2;
                        obstacle.second.left    = min(obstacle.second.left, x);
                        obstacle.second.right   = max(obstacle.second.right, x);
                        obstacle.second.top     = min(obstacle.second.top, y);
                        obstacle.second.bottom  = max(obstacle.second.bottom, y);
                        new_obstacle = false;
                        break;
                    }
                }
            }
            if (new_obstacle)
            {
                boundingBox_t box;
                box.left    = x;
                box.right   = x;
                box.top     = y;
                box.bottom  = y;

                obstacles.emplace_back(make_pair(distant, box));
            }
        }

        /* ******************************************
         * Merging the obstacles
         * ******************************************
         */
        for (int i = 0; i < obstacles.size(); ++i){
            auto& obstacle_i = obstacles[i];
            for (int j = 0; j < obstacles.size(); ++j){
                auto& obstacle_j = obstacles[j];
                if (i != j && obstacle_i.first != INFINITY && abs(obstacle_i.first - obstacle_j.first) < LIDAR_GRADIENT_SENSITIVE)
                {
                    bool hori_flag = max(obstacle_i.second.bottom - obstacle_j.second.top, obstacle_j.second.bottom - obstacle_i.second.top) < (obstacle_i.second.bottom - obstacle_i.second.top) + (obstacle_j.second.bottom - obstacle_j.second.top) + LIDAR_MERGING_SENSITIVE;
                    bool verti_flag = max(obstacle_i.second.right - obstacle_j.second.left, obstacle_j.second.right - obstacle_i.second.left) < (obstacle_i.second.right - obstacle_i.second.left) + (obstacle_j.second.right - obstacle_j.second.left) + LIDAR_MERGING_SENSITIVE;
                    if (hori_flag && verti_flag)
                    {
                        obstacle_i.first            = (obstacle_i.first + obstacle_j.first) / 2;
                        obstacle_i.second.top       = min(obstacle_i.second.top     , obstacle_j.second.top);
                        obstacle_i.second.bottom    = max(obstacle_i.second.bottom  , obstacle_j.second.bottom);
                        obstacle_i.second.left      = min(obstacle_i.second.left    , obstacle_j.second.left);
                        obstacle_i.second.right     = max(obstacle_i.second.right   , obstacle_j.second.right);
                        
                        obstacle_j.first = LIDAR_RANGING_MAX;
                    }
                }
            }
        }

        /* ******************************************
         * Removing too samll obstacle
         * ******************************************
         */
        for (auto obstacle: obstacles) {
            int area = (obstacle.second.right - obstacle.second.left) * (obstacle.second.bottom - obstacle.second.top);
            
            if (area > pow(56, 2) && LIDAR_RANGING_MAX > obstacle.first)
            {
                log_V("CPS_Engine", "Slincing obstacle: [" + to_string(obstacle.second.top) + ", " + to_string(obstacle.second.bottom) + ", " + to_string(obstacle.second.left) + ", " + to_string(obstacle.second.right) + "]");
                int diff_area = INT32_MAX;
                int shapeId = 0;
                for (int i = 0; i < imgShapes.size(); i++)
                {
                    int new_diff = abs(area - (imgShapes[i].first * imgShapes[i].second));
                    if (new_diff < diff_area)
                    {
                        shapeId = i;
                        diff_area = new_diff;
                    }
                }

                string logInfo = "assign [" + to_string(obstacle.second.bottom - obstacle.second.top) + ", " + to_string(obstacle.second.right - obstacle.second.left) + "] to shape: [" + to_string(imgShapes[shapeId].first) + ", " + to_string(imgShapes[shapeId].second) + "]";
                log_V("CPS_Engine::dataPreprocessor", logInfo);

                /* Create task by the object from the raw image */
                cv::Mat* croppedImage = new cv::Mat(mImg(
                    cv::Range(obstacle.second.top   , obstacle.second.bottom), 
                    cv::Range(obstacle.second.left  , obstacle.second.right)
                ));

                /* ************************************************************
                 * Set the priority as the fraction of the normalized distant
                 * ************************************************************
                 */
                Inference_Task_t task = {(void*)croppedImage, (LIDAR_RANGING_MAX - obstacle.first) / LIDAR_RANGING_MAX, models[shapeId]};
                taskQueue.emplace_back(task);

            }
        }
        mLidarPoints.clear();
        obstacles.clear();
    gettimeofday(&end, NULL);
    
    float spendTime = (1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001;   
    log_I("CPS_Engine", "Slicing spend: " + to_string(spendTime) + " ms");

}


/** ===============================================================================================
 * \name    Inference_sched
 * 
 * \brief   Schedule the input datas into prioritied task
 * ================================================================================================
 */
void 
CPS_Engine::Inference_sched (void)
{
    /* sort the taskQueue by non-ascending order of priority */
    sort(taskQueue.begin(), taskQueue.end(), [](Inference_Task_t x, Inference_Task_t y) {return x.priority > y.priority;});
    for(auto task : taskQueue)
    {
        log_V("CPS_Engine", "Sort tasks: " + to_string(task.priority));
    }
}


/** ===============================================================================================
 * \name    onInference
 * 
 * \brief   Keep inference util the meet deadline
 * ================================================================================================
 */
void 
CPS_Engine::onInference (timeval frameStart)
{
    
    log_D("CPS_Engine", "Task queue size: " + to_string(taskQueue.size()));
    if(taskQueue.size() == 0)
    {
        log_D("CPS_Engine", "taskQueue.size() == 0");
        return;
    }

    vector<Inference_Task_t*> taskSetPriorities;
    for(auto model : models)
    {
        taskSetPriorities.emplace_back(new Inference_Task_t({nullptr, 0, model}));
    }

    /* ******************************************
     * Calculate the total priority of model
     * ******************************************
     */
    for(auto task : taskQueue)
    {
        for (int i = 0; i < taskSetPriorities.size(); i++)
        {
            if(taskSetPriorities[i]->model == task.model)
            {
                taskSetPriorities[i]->priority += task.priority;
            }
        }
    }

    for(auto taskSetPriority : taskSetPriorities)
    {
        log_V("CPS_Engine", taskSetPriority->model->modelName + " priority: " + to_string(taskSetPriority->priority));
    }

    float spendTime;
    struct timeval now;

    do 
    {
        /* Update remaining time */
        gettimeofday(&now, NULL);
        spendTime = (1000000 * (now.tv_sec - frameStart.tv_sec) + (now.tv_usec - frameStart.tv_usec)) * 0.001;
        float remaingTime = SENSING_PERIOD - spendTime;
        log_D("CPS_Engine", "Remaining time: " + to_string(remaingTime));
        
        /* Choose the most priority model with enough time */
        auto max_task = max_element(taskSetPriorities.begin(), taskSetPriorities.end(), [](Inference_Task_t* x, Inference_Task_t* y) {return x->priority < y->priority;});

        float tolerance = 1e-6;  // The precision tolerance
        if(abs((*max_task)->priority) < tolerance)
        {
            break;

        } else if((*max_task)->model->spendTime > remaingTime) {
            (*max_task)->priority = 0;
            continue;
        }

        /* Pick task form the queue */
        for (auto it = taskQueue.begin(); it < taskQueue.end() && !(*max_task)->model->fullyBatch;)
        {
            Inference_Task_t task = (*it);

            if((*max_task)->model == task.model)
            {
                (*max_task)->priority -= task.priority;

                cv::Mat* img = (cv::Mat*) task.data;
                vector<float> dataStream(task.model->singleInputSize);
                task.model->dataPreprocess(task.data, &dataStream);
                task.model->Onnx_addInput(dataStream);
                taskQueue.erase(it);
                continue;
            }
            it++;
        }

        /* Start Inference */
        (*max_task)->model->Onnx_inference();

    }while(SENSING_PERIOD > spendTime);
    
    log_I("CPS_Engine", "Remaining tasks: " + to_string(taskQueue.size()));
    for(auto task : taskQueue)
    {
        log_D("CPS_Engine", task.model->modelName);
    }

    taskQueue.clear();
    taskSetPriorities.clear();

}