import os
import sys
import argparse
import numpy as np
import pyrealsense2 as rs
import time
import glob
import cv2

from typing import Dict

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_2d_grasps, show_image


def get_session() -> tf.Session:
    '''
    Create a session
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return tf.Session(config=config)


def get_grasp_estimator(global_config: Dict,
                        sess: tf.Session,
                        checkpoint_dir: str) -> GraspEstimator:
    '''
    Create a working (pretrained) Grasp Estimator

    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param sess: a tensor flow session
    '''
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    return grasp_estimator


def start_camera():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    pipeline.start(config)
    return pipeline

def get_camera_params(pipeline):
    # Get the intrinsics of the depth stream
    profiles = pipeline.get_active_profile()
    depth_profile = profiles.get_stream(rs.stream.depth)
    intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    # The intrinsic matrix is composed of fx, fy (focal lengths), and cx,
    # cy (optical centers)
    cam_k = np.array([[intrinsics.fx, 0, intrinsics.ppx ],
                      [0, intrinsics.fy, intrinsics.ppy],
                      [0, 0, 1]])

    # Distortion coefficients
    distortion = np.array(intrinsics.coeffs)
    return cam_k, distortion



def preprocess_frames(depth_frame,
                      color_frame,
                      cam_k,
                      skip_border_objects=False,
                      z_range=[0.2,1.8]):
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                       cv2.COLORMAP_JET)

    # Convert depth to meters by dividing by 1000.0
    depth_image = depth_image / 1000.0

    print('Converting depth to point cloud(s)...')
    pc_full, _, _ = grasp_estimator.extract_point_clouds(depth_image,
                                                         cam_k,
                                                         rgb=color_image,
                                                         skip_border_objects=skip_border_objects,
                                                         z_range=z_range)

    return pc_full, color_image, depth_colormap

 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    # print(str(global_config))
    # print('pid: %s'%(str(os.getpid())))

    sess = get_session()
    grasp_estimator = get_grasp_estimator(global_config, sess, FLAGS.ckpt_dir)
    pipeline = start_camera()
    cam_k, distortion = get_camera_params(pipeline)

    # Create alignment primitive with color as its target stream
    align = rs.align(rs.stream.color)

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # Align the depth frames to color frame

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        pc_full, color_image, depth_colormap = preprocess_frames(depth_frame,
                                                                 color_frame,
                                                                 cam_k,
                                                                 skip_border_objects=FLAGS.skip_border_objects,
                                                                 z_range=eval(str(FLAGS.z_range)))
        
        print('Generating Grasps...')
        (pred_grasps_cam,
         scores,
         contact_pts,
         grasp_opennings) = grasp_estimator.predict_scene_grasps(sess, 
                                                                 pc_full, 
                                                                 pc_segments={}, 
                                                                 local_regions=False, 
                                                                 filter_grasps=False,
                                                                 forward_passes=FLAGS.forward_passes)
        
        rgb = np.array(cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        visualize_2d_grasps(rgb, cam_k, distortion, pred_grasps_cam, scores, gripper_openings= grasp_opennings)
        cv2.imshow('Projected Shape', rgb)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):  # Exit on 'q' key press
            break
        

