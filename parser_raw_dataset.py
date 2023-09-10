# data process packages
import os
import time
import threading
import numpy as np
import tensorflow as tf

# image packages
import cv2
import matplotlib.pyplot as plt

# waymo open dataset package
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# -----------------------------------------------------------------------
# file config
SRCDIR = '/media/kunsheng/AOSP/dataset/testing/'
DESDIR = '/home/kunsheng/Desktop/KunSheng/Autonomous Driving/dataset/'

# FILENAME = 'segment-2002071659309679036_720_000_740_000'
# FILENAME = 'segment-2368807205588573344_3616_000_3636_000'
FILENAME = 'segment-17292852148707454629_140_000_160_000'

# -----------------------------------------------------------------------
# load dataset
for file in os.listdir(SRCDIR):
    FILENAME = file[:-9]
    if not os.path.exists(DESDIR + FILENAME):
        os.mkdir(DESDIR + FILENAME)

    dataset = tf.data.TFRecordDataset(SRCDIR + FILENAME + '.tfrecord', compression_type='')

    for frame_id, data in enumerate(dataset):
        frame_folder = DESDIR + FILENAME + '/' + str(frame_id)
        if not os.path.exists(frame_folder):
            os.mkdir(frame_folder)

        # -----------------------------------------------------------------------
        # start parser this frame
        start_time = time.time()

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # ------------------------------------------------------------------------
        # handling images data
        for image in frame.images:
            img = tf.image.decode_jpeg(image.image).numpy()
            # img = cv2.resize(img, (960, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow (open_dataset.CameraName.Name.Name(image.name), img)
            cv2.imwrite(frame_folder + '/' + str(open_dataset.CameraName.Name.Name(image.name)) + '.jpeg', img)

        # -------------------------------------------------------------------------
        # handle lasers data

        (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1)

        points_all = np.concatenate(points, axis=0)
        points_all_ri2 = np.concatenate(points_ri2, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)
        cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

        cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
        cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

        points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        # get the laser points and distance of each image
        for image in frame.images:
            mask = tf.equal(cp_points_all_tensor[..., 0], image.name)
            frame_cp_points_all_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
            frame_points_all_tensor_ = tf.gather_nd(points_all_tensor, tf.where(mask))
            
            # get the laser distance of all points
            projected_points_all_from_raw_data = tf.concat([frame_cp_points_all_tensor[..., 1:3], frame_points_all_tensor_], axis=-1).numpy()

            file_laser_point = open(frame_folder + '/' + str(open_dataset.CameraName.Name.Name(image.name)) + '.txt', "w")
            for point in projected_points_all_from_raw_data:
                # print(point)
                file_laser_point.write(str(point) + '\n')
            file_laser_point.close()

        print("frame " + str(frame_id) + ": " + str(time.time() - start_time))