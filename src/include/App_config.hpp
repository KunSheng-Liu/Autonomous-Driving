/** 
 * \name    APP_config.hpp
 * 
 * \brief   Declare all configure
 * 
 * \date    Mar 6, 2023
 */

#ifndef _APP_CONFIG_HPP_
#define _APP_CONFIG_HPP_

/* ************************************************************************************************
 * Enumeration
 * ************************************************************************************************
 */
/* Logger level */
#define ERROR                   0
#define WARNNING                1
#define INFO                    2
#define DEBUG                   3
#define VERBOSE                 4

/* Peripheral */
#define SENSOR_CAMERA           0x01
#define SENSOR_LIDAR            0x02
#define SENSOR_AUDIO            0x04

/* Models */
#define RESNET_56_56            0x01
#define RESNET_112_112          0x02
#define RESNET_168_168          0x04
#define RESNET_224_224          0x08
#define RESNET_280_280          0x10
#define RESNET_336_336          0x20
#define RESNET_448_448          0x40
#define RESNET_1280_1920        0x80

#define YOLONET_256_256         0x01
#define YOLONET_384_384         0x02
#define YOLONET_512_512         0x04
#define YOLONET_640_640         0x08

/* Approach  */
#define RT_CPS                  0       // the related work: "Real-Time Task Scheduling for Machine Perception in Intelligent Cyber-Physical System."
#define RT_SGE                  1       // my approach


/* ************************************************************************************************
 * Application Configuration
 * ************************************************************************************************
 */
#define LOG_LEVEL               DEBUG
#define PROFILE_MODEL           false   
#define THREAD_INFERENCE        true
#define FRAME_NUM               10
#define LIDAR_RANGING_MAX       75
#define COCO_DATASET_LABEL      "../models/coco_labels.txt"
#define IMAGENET_DATASET_LABEL  "../models/imagenet_labels.txt"

/* Benchmark config */
#define INFERENCE_ENGINE        RT_SGE
#define SENSING_PERIOD          100     // ms
#define PERIPHERAL_MASK         (SENSOR_CAMERA | SENSOR_LIDAR)
#define DATASET_PATH            "../dataset/segment-10243642118467607790_880_000_900_000/"
#define MODEL_PATH              "../models/"

/* ************************************************************************************************
 * Declaration for each approach
 * ************************************************************************************************
 */
#if (INFERENCE_ENGINE == RT_CPS)
#define MODEL_MASK              0x0f

#elif (INFERENCE_ENGINE == RT_SGE)
#define MODEL_MASK              0x0f

#endif

#endif