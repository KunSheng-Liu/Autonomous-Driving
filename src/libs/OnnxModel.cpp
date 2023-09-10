/**
 * \name    OnnxModels.cpp
 * 
 * \brief   Implement the API
 * 
 * \date    Mar 6, 2023
 */

#include "../include/OnnxModels.hpp"

/** ===============================================================================================
 * \name    OnnxModel
 *
 * \brief   Construct an onnx model 
 * 
 * \param   model_name the model you want to load
 * \param   batch_limit the constraint of batch inference
 * ================================================================================================
 */
OnnxModel::OnnxModel (string model_name, int batch_limit) : modelName(model_name), batchLimit(batch_limit), fullyBatch(false), busyFlag(false)
{
    Onnx_modelSetup();
}


/** ===============================================================================================
 * \name    ~OnnxModel
 *
 * \brief   Destruct an onnx model 
 * ================================================================================================
 */
OnnxModel::~OnnxModel (void)
{
    Ort::AllocatorWithDefaultOptions allocator;
    session->EndProfilingAllocated(allocator);
}


/** ===============================================================================================
 * \name    Onnx_inference
 *
 * \brief   Inference the data in the input streaming
 * ================================================================================================
 */
void 
OnnxModel::Onnx_inference (void) {

    if (inputStreams.size() == 0)
    {
        log(modelName, ONNX_INFERENCE_INPUTSIZE_ZERO);
        return;
    }
    
    if (inputStreams.size() % singleInputSize != 0)
    {
        log(modelName, ONNX_INFERENCE_INPUTSIZE_WRONG);
        return;
    }

    /* ******************************************
     * Using fix batch size, if the input size is
     * not enough, fill up by 0 data.
     * ******************************************
     */
    inputNodeDims[0] = batchLimit;
    vector<float> inputTensorValues(batchLimit * singleInputSize, 0);
    copy(inputStreams.begin(), inputStreams.begin() + inputTensorValues.size(), inputTensorValues.begin());

    log_D(modelName, "Input Tensor size: " + to_string(inputTensorValues.size()));

    vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>( memory_info, 
                                                            inputTensorValues.data(), 
                                                            inputTensorValues.size(), 
                                                            inputNodeDims.data(), 
                                                            inputNodeDims.size()
                                                          ));
    
    struct timeval inference_start, inference_end;
    gettimeofday(&inference_start, NULL);
        busyFlag = true;
        vector<Ort::Value> outputTensors = session->Run( Ort::RunOptions{nullptr}, 
                                                         inputNodeNames.data(), 
                                                         inputTensors.data(), 
                                                         inputTensors.size(), 
                                                         outputNodeNames.data(), 
                                                         outputNodeNames.size()
                                                       );
        busyFlag = false;
    gettimeofday(&inference_end, NULL);

    spendTime = (1000000 * (inference_end.tv_sec - inference_start.tv_sec) + (inference_end.tv_usec - inference_start.tv_usec)) * 0.001;
    log_I(modelName, "Inference " + to_string(inputNodeDims[0]) +  " batch spend: " + to_string(spendTime) + " ms");

    decodeResult(move(outputTensors));

    /* clear the resource */
    inputStreams.clear();
    fullyBatch = false;
    
    log_V(modelName, "Clear inputStreams, size: " + to_string(inputStreams.size()));

}


/** ===============================================================================================
 * \name    Onnx_addInput
 *
 * \brief   Stash the data into the inference data queue
 * 
 * \param   dataStream the prepocessed input data
 * ================================================================================================
 */
void 
OnnxModel::Onnx_addInput (vector<float> dataStream) 
{
    inputStreams.insert(
        inputStreams.end(), 
        make_move_iterator(dataStream.begin()), 
        make_move_iterator(dataStream.end())
    );
    if (inputStreams.size() == singleInputSize * batchLimit)
    {
        fullyBatch = true;
    }

    log_V(modelName, "Add input size: " + to_string(dataStream.size()));
    log_V(modelName, "Stashed data size: " + to_string(inputStreams.size()));
}


/** ===============================================================================================
 * \name    Onnx_modelSetup
 *
 * \brief   Load and warm-up the model through standard onnx API
 * ================================================================================================
 */
void
OnnxModel::Onnx_modelSetup () 
{
    struct timeval setup_start, setup_end;
    gettimeofday(&setup_start, NULL);
        log(modelName, ONNX_SETUPMODEL_START);

        env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
        Ort::SessionOptions session_options;
        OrtCUDAProviderOptions cuda_options;

        int cpu_threads = 8;
        cuda_options.device_id = 0;
#if PROFILE_MODEL
        session_options.EnableProfiling(("../profile/" + modelName).c_str());
#endif
        cuda_options.gpu_mem_limit = 1 << 30;
        session_options.SetIntraOpNumThreads(cpu_threads);
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        string model_path = MODEL_PATH + modelName + ".onnx";
        session = new Ort::Session(*env, model_path.c_str(), session_options);

        /* get the number of model input/output nodes */
        const size_t num_input_nodes = session->GetInputCount();
        const size_t num_output_nodes = session->GetOutputCount();
        inputNodeDims.reserve(num_input_nodes);
        outputNodeDims.reserve(num_output_nodes);

        /* ******************************************
         * Iterate all nodes for record the input 
         * dimension
         * ******************************************
         */
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            auto type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            /* get the name of input nodes */
            inputNodeNames.push_back(input_name.get());

            /* save the local pointer */
            namesPtr.push_back(move(input_name));

            /* get the dimension of input nodes */
            inputNodeDims = tensor_info.GetShape();
        }

        /* ******************************************
         * Iterate all nodes for record the output 
         * dimension
         * ******************************************
         */
        for (size_t i = 0; i < num_output_nodes; i++) {
            // print output node names
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            auto type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            /* get the name of output nodes */
            outputNodeNames.push_back(output_name.get());

            /* save the local pointer */
            namesPtr.push_back(move(output_name));

            /* get the dimension of output nodes */
            outputNodeDims = tensor_info.GetShape();
        }

        /* ******************************************
         * Print out model detial
         * ******************************************
         */
        log_V(modelName, "Number of inputs = " + to_string(num_input_nodes));
        log_V(modelName, "Number of outputs = " + to_string(num_output_nodes));

        for (size_t i = 0; i < num_input_nodes; i++) {
            string logInfo = "Input dims = [";
            for (size_t j = 0; j < inputNodeDims.size(); j++) {
                logInfo += to_string(inputNodeDims[j]) + ", ";
            }
            logInfo += "]";
            
            log_V(modelName, logInfo);
        }

        for (size_t i = 0; i < num_output_nodes; i++) {
            string logInfo = "Output dims = [";
            for (size_t j = 0; j < outputNodeDims.size(); j++) {
                logInfo += to_string(outputNodeDims[j]) + ", ";
            }
            logInfo += "]";
            
            log_V(modelName, logInfo);
        }

        /* ******************************************
         * Record the single data size
         * 
         * Note: index 0 represent batch size
         * ******************************************
         */
        singleInputSize = 1;
        for (int i = 1; i < inputNodeDims.size(); i++)
        {
            singleInputSize *= inputNodeDims[i];
        }

        /* ******************************************
         * Warm up model for optimized model for GPU
         * ******************************************
         */
        log(modelName, ONNX_SETUPMODEL_WARMUP);
        inputStreams = vector<float>(batchLimit * singleInputSize, 0);
        Onnx_inference();   // take longer time for optimize the model
        inputStreams = vector<float>(batchLimit * singleInputSize, 0);
        Onnx_inference();   // for record the runtime inference time

    gettimeofday(&setup_end, NULL);

    float spendTime = (1000000 * (setup_end.tv_sec - setup_start.tv_sec) + (setup_end.tv_usec - setup_start.tv_usec)) * 0.001;
    log_I(modelName, "Model setup with " + to_string(inputNodeDims[0]) +  " batch spend: " + to_string(spendTime) + " ms");

}



/** ===============================================================================================
 * \name    dataPreprocess
 * 
 * \param   data the raw input data
 * \param   precessedStream the preprocessed datastream, could be push into inference queue
 * ================================================================================================
 */
void
OnnxModel::dataPreprocess (void* data, vector<float> *precessedStream) 
{
    log_D(modelName, "Base case not implement function: dataPreprocess");
}


/** ===============================================================================================
 * \name    decodeResult
 * 
 * \param   results the model result
 * ================================================================================================
 */
void 
OnnxModel::decodeResult (vector<Ort::Value> results) 
{
    log_D(modelName, "Base case not implement function: decodeResult");
}


/** ***********************************************************************************************
 * \name OnnxResNet
 * 
 * \brief   Override the virtual function to fix YoloNet requirement
 * ************************************************************************************************
 */
OnnxResNet::OnnxResNet (string model_name, int batch_limit) : OnnxModel(model_name, batch_limit)
{
    loadLabels();
}


/** ===============================================================================================
 * \name    dataPreprocess
 * 
 * \param   data the raw input data, cv::Mat
 * \param   precessedStream the preprocessed datastream, could be push into inference queue
 * ================================================================================================
 */
void
OnnxResNet::dataPreprocess (void* data, vector<float> *precessedStream)
{
    log_V("OnnxResNet", "dataPreprocess");
    cv::Mat* img = (cv::Mat*) data;
    float multiplier = 1.0 / 255;
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};

    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    
    cv::resize(*img, resizedImageBGR,
                cv::Size(inputNodeDims[2], inputNodeDims[3]),
                cv::InterpolationFlags::INTER_CUBIC);

    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
    
    resizedImageRGB.convertTo(resizedImage, CV_32F, multiplier);
    
    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    channels[0] = (channels[0] - mean[0]) / std[0];
    channels[1] = (channels[1] - mean[1]) / std[1];
    channels[2] = (channels[2] - mean[2]) / std[2];
    cv::merge(channels, 3, resizedImage);

    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), precessedStream->begin());

}


/** ===============================================================================================
 * \name    decodeResult
 * 
 * \param   results the model result
 * ================================================================================================
 */
void 
OnnxResNet::decodeResult (vector<Ort::Value> results) 
{
    log_V("OnnxResNet", "decodeResult");
    /* log the results informations */
    vector<int64_t> outputNodeDims = results[0].GetTensorTypeAndShapeInfo().GetShape();
    string logInfo = "Result dims = [";
    for (int dim: outputNodeDims)
    {
        logInfo += to_string(dim) + ", ";
    }
    logInfo += "]";

    log_D(modelName, logInfo);

    /* ******************************************
    * Decode the results
    * ******************************************
    */
    const float* outStream = results[0].GetTensorMutableData<float>();
        
    log_I(modelName, "Num, id, label, confidence");
    for (int i = 0; i < outputNodeDims[0]; i++) {
        int id = 0;
        float max_confidence = 0;

        /* find the entry with max confidence */
        for (int j = 0; j < outputNodeDims[1]; j++) {
            float confidence = outStream[i * outputNodeDims[1] + j];
            if (confidence > max_confidence)
            {
                max_confidence = confidence;
                id = j;
            }
        }
        
        /* log out result */
        logInfo  = to_string(i) + ", ";
        logInfo += to_string(id) + ", ";
        logInfo += labels[id] + ", ";
        logInfo += to_string(max_confidence);

        log_I(modelName, logInfo);
    }

    results.clear();
}


/** ===============================================================================================
 * \name    loadLabels
 * 
 * \brief   Load the corresponding dataset
 * ================================================================================================
 */
void
OnnxResNet::loadLabels (void)
{
    labels.clear();
    string line;
    ifstream filePtr(IMAGENET_DATASET_LABEL);
    
    while (getline(filePtr, line))
    {
        labels.push_back(line);
    }
}



/** ***********************************************************************************************
 * \name OnnxYoloNet
 * 
 * \brief   Override the virtual function to fix YoloNet requirement
 * ************************************************************************************************
 */
OnnxYoloNet::OnnxYoloNet (string model_name, int batch_limit) : OnnxModel(model_name, batch_limit)
{
    loadLabels();
}


/** ===============================================================================================
 * \name    dataPreprocess
 * 
 * \param   data the raw input data, cv::Mat
 * \param   precessedStream the preprocessed datastream, could be push into inference queue
 * ================================================================================================
 */
void
OnnxYoloNet::dataPreprocess (void* data, vector<float> *precessedStream)
{
    log_V("OnnxYoloNet", "dataPreprocess");
    cv::Mat* img = (cv::Mat*) data;
    log_D("dataPreprocess", "Image width: " + to_string(img->cols) + ", Image height: " + to_string(img->rows));

    float multiplier = 1.0 / 255;

    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;

    cv::resize(*img, resizedImageBGR,
            cv::Size(inputNodeDims[2], inputNodeDims[3]),
            cv::InterpolationFlags::INTER_CUBIC);

    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);

    resizedImageRGB.convertTo(resizedImage, CV_32F, multiplier);

    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    
    copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), precessedStream->begin());

}


/** ===============================================================================================
 * \name    decodeResult
 * 
 * \param   results the model result
 * ================================================================================================
 */
void 
OnnxYoloNet::decodeResult (vector<Ort::Value> results) 
{
    log_V("OnnxYoloNet", "decodeResult");
    /* log the results informations */
    vector<int64_t> output_node_dims = results[0].GetTensorTypeAndShapeInfo().GetShape();
    string logInfo = "Result dims = [";
    for (int dim: output_node_dims)
    {
        logInfo += to_string(dim) + ", ";
    }
    logInfo += "]";

    log_D(modelName, logInfo);

    /* ******************************************
    * Decode the results
    * ******************************************
    */
    const float* outStream = results[0].GetTensorMutableData<float>();

    log_I(modelName, "Batch, x, y, width, height, label, confidence");
    for (int i = 0; i < output_node_dims[0]; i++)
    {
        logInfo  = to_string((int)outStream[i * output_node_dims[1]]) + ", ";
        logInfo += to_string(outStream[i * output_node_dims[1] + 1]) + ", ";
        logInfo += to_string(outStream[i * output_node_dims[1] + 2]) + ", ";
        logInfo += to_string(outStream[i * output_node_dims[1] + 3]) + ", ";
        logInfo += to_string(outStream[i * output_node_dims[1] + 4]) + ", ";
        logInfo += labels[outStream[i * output_node_dims[1] + 5]]    + ", ";
        logInfo += to_string(outStream[i * output_node_dims[1] + 6]);

        log_I(modelName, logInfo);
    }
}


/** ===============================================================================================
 * \name    loadLabels
 * 
 * \brief   Load the corresponding dataset
 * ================================================================================================
 */
void
OnnxYoloNet::loadLabels (void)
{
    labels.clear();
    string line;
    ifstream filePtr(COCO_DATASET_LABEL);
    
    while (getline(filePtr, line))
    {
        labels.push_back(line);
    }
}

