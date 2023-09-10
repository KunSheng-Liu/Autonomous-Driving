/**
 * \name    OnnxModels.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 7, 2023
 */

#include "include/SensingEngine.hpp"

/** ===============================================================================================
 * \name    SensingEngine
 * 
 * \brief   The class for handling the peripheral sensor
 * ================================================================================================
 */
SensingEngine::SensingEngine ()
{
    dataReadyToSync = false;
}


/** ===============================================================================================
 * \name    run
 * 
 * \brief   start the sensing thread
 * ================================================================================================
 */
void
SensingEngine::run (void)
{
    pthread_create(&mthread, 
                   NULL, 
                   SensingEngine::threadSensing, 
                   this
                );
}


/** ===============================================================================================
 * \name    stop
 * 
 * \brief   Stop the sensing thread
 * ================================================================================================
 */
void
SensingEngine::stop (void)
{
    pthread_cancel(mthread);
    dataReadyToSync = false;
    log_D("SensingEngine", "Stop the sensing thread");
}


/** ===============================================================================================
 * \name    threadSensing
 * 
 * \brief   Pooling all sensor nodes in periods
 * ================================================================================================
 */
void*
SensingEngine::threadSensing (void* arg)
{
    SensingEngine* param = (SensingEngine*) arg;
    for(int frameID = 0; frameID < FRAME_NUM; frameID++)
    {
        if (!param->dataReadyToSync)
        {
            log_D("SensingEngine", "Start sensing");

#if PERIPHERAL_MASK & SENSOR_CAMERA
            param->Sensing_Camera(DATASET_PATH + to_string(frameID) + "/FRONT.jpeg");
#endif

#if PERIPHERAL_MASK & SENSOR_LIDAR
            param->Sensing_Lidar(DATASET_PATH + to_string(frameID) + "/FRONT.txt");
#endif

#if PERIPHERAL_MASK & SENSOR_AUDIO
            std::cout << "Sensing_Audio haven't implement" << std::endl;
#endif
            param->dataReadyToSync = true;
            log_D("SensingEngine", "Done sensing");
        }

        do{
            usleep(1000);
        }while(param->dataReadyToSync);
    }
    param->dataReadyToSync = false;
    pthread_exit(nullptr);
}


/** ===============================================================================================
 * \name    startNextSensing
 * 
 * \brief   reset the data ready flag \b dataReadyToSync
 * ================================================================================================
 */
void
SensingEngine::startNextSensing (void)
{
    log_D("SensingEngine", "Reset dataReadyToSync");
    dataReadyToSync = false;
}


/** ===============================================================================================
 * \name    Sensing_Camera
 * 
 * \brief   load the image form the dataset
 * 
 * \param   filePath image file path for loading
 * ================================================================================================
 */
void
SensingEngine::Sensing_Camera (string filePath)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);

        cameraData = cv::imread(filePath, cv::ImreadModes::IMREAD_COLOR);

    gettimeofday(&end, NULL);
    float spendTime = (1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001;
    log_D("SensingEngine", "Sensing_Camera spend: " + to_string(spendTime) + " ms");
}


/** ===============================================================================================
 * \name    Sensing_Lidar
 * 
 * \brief   load the lidar points form the dataset
 * 
 * \param   filePath lidar file path for loading
 * ================================================================================================
 */
void
SensingEngine::Sensing_Lidar (string filePath)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
        fstream file; 
        file.open(filePath, ios::in);
        assert(file.is_open() && "dataset file is not exist");
    
        LidarData.clear();

        // parser lines into ranging points
        string readLine;
        getline(file, readLine); // skip title line
        while(getline(file, readLine, '\t')){
            int x = stoi(readLine);
            getline(file, readLine, '\t');
            int y = stoi(readLine);
            getline(file, readLine, '\n');
            float distant = stof(readLine);
            LidarData.emplace_back(make_pair(make_pair(x, y), distant));
        }
        file.close();
    gettimeofday(&end, NULL);

    float spendTime = (1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)) * 0.001;
    log_D("SensingEngine", "Sensing_Lidar spend: " + to_string(spendTime) + " ms");

}