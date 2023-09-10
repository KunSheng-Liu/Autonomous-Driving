/**
 * \name    OnnxModels.hpp
 * 
 * \brief   Declare the model API 
 *          
 * \note    Available model type:
 *          - \b OnnxResNet
 *          - \b OnnxYoloNet 
 * 
 * \date    Mar 6, 2023
 */

#ifndef _ONNX_MODEL_HPP_
#define _ONNX_MODEL_HPP_

/* ************************************************************************************************
 * Include Library
 * ************************************************************************************************
 */
#include "App_config.hpp"
#include "Log.hpp"

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
 * \name    OnnxModel
 * 
 * \brief   The base class of onnx model. You can add new model by inheritance this class and
 *          override the virtual function to fit the model desire.
 * ================================================================================================
 */
class OnnxModel
{
/** ***********************************************************************************************
 * Class Constructor
 *
 * \param   model_name the model you want to load
 * \param   batch_limit the constraint of batch inference
 * ************************************************************************************************
 */ 
public:
    OnnxModel (string model_name, int batch_limit);
    ~OnnxModel (void);

/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    void Onnx_addInput (vector<float> dataStream);
    void Onnx_inference (void);
    virtual void dataPreprocess (void* data, vector<float>* preprocessData);

private:
    void Onnx_modelSetup (void);
    virtual void decodeResult (vector<Ort::Value> results);


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
public:
    /* The default input size of the model under 1 batch size */
    int singleInputSize;

    /* The maxinum batch size for the model, could be remove in future */
    int batchLimit;
    
    /* The inputStreams already filly the batchLimit? */
    bool fullyBatch;

    /* Model is inferencing? */
    bool busyFlag;

    /* Last inference spend time */
    float spendTime;

    /* The thread instance of this model, use for inference in thread */
    pthread_t mthread;

    /* The model name */
    string modelName;

protected:
    /* Stash the input data before inference */
    vector<float> inputStreams;

    vector<int64_t> inputNodeDims;
    vector<int64_t> outputNodeDims;
    vector<const char*> inputNodeNames;
    vector<const char*> outputNodeNames;
    
private:
    /* Used for optimizing the model in setup phase,  */
    Ort::Env *env;
    Ort::Session *session;
    vector<Ort::AllocatedStringPtr> namesPtr;
};


/** ===============================================================================================
 * \name    OnnxResNet
 * 
 * \brief   A OnnxModel extension class for ResNet. 
 * 
 * \param   model_name the model you want to load
 * \param   batch_limit the constraint of batch inference
 * ================================================================================================
 */
class OnnxResNet: public OnnxModel
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    OnnxResNet (string model_name, int batch_limit);


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    /* Implement virtual functions */
    void dataPreprocess (void* data, vector<float>* preprocessData) override;

private:
    /* Implement virtual functions */
    void decodeResult (vector<Ort::Value> results) override;

    /* local functions */
    void loadLabels (void);
    

/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    vector<string> labels;

};


/** ===============================================================================================
 * \name    OnnxYoloNet
 * 
 * \brief   A OnnxModel extension class for YoloNet. 
 * 
 * \param   model_name the model you want to load
 * \param   batch_limit the constraint of batch inference
 * ================================================================================================
 */
class OnnxYoloNet: public OnnxModel
{
/* ************************************************************************************************
 * Class Constructor
 * ************************************************************************************************
 */ 
public:
    OnnxYoloNet (string model_name, int batch_limit);


/* ************************************************************************************************
 * Functions
 * ************************************************************************************************
 */
public:
    /* Implement virtual functions */
    void dataPreprocess (void* data, vector<float>* preprocessData) override;

private:
    /* Implement virtual functions */
    void decodeResult (vector<Ort::Value> results) override;

    /* local functions */
    void loadLabels (void);


/* ************************************************************************************************
 * Parameter
 * ************************************************************************************************
 */
private:
    vector<string> labels;

};

#endif