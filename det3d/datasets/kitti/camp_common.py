import pathlib, json
import pickle
import re
import numpy as np
import os.path as osp
from pypcd import pypcd
from IPython import embed

from collections import OrderedDict
from pathlib import Path
from skimage import io
from tqdm import tqdm

from det3d.core import box_np_ops

def create_kitti_info_file(data_path, save_path=None, relative_path=True):
    print("Generate info. this may take several minutes.")
    
    """ train """
    kitti_infos_train = get_kitti_image_info(
        data_path,
        txt_name='train_filter.txt'
    )
    if not save_path:
        save_path = data_path
    filename = osp.join(save_path, "kitti_infos_train.pkl")
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(kitti_infos_train, f)


    """ val """
    kitti_infos_val = get_kitti_image_info(
        data_path,
        txt_name='val_filter.txt')
    filename = osp.join(save_path, 'kitti_infos_val.pkl')
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)

    """ test """
#     kitti_infos_test = get_kitti_image_info(
#         data_path,
#         txt_name='test.txt')
#     filename = osp.join(save_path, 'kitti_infos_test.pkl')
#     print(f"Kitti info test file is saved to {filename}")
#     with open(filename, 'wb') as f:
#         pickle.dump(kitti_infos_test, f)

def get_kitti_image_info(
    path,
    label_info=True,
    velodyne=True,
    calib=True,
    txt_name='train_filter.txt'
):
    """
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    config_txt = osp.join(path, 'labels_filer', txt_name)

    image_infos = []
    with open(config_txt, 'r') as wf:
        lines = wf.readlines()
        for line in lines:
            # parse line
            record = json.loads(line.strip())
            idx = record['id']

            info = {}
            pc_info = {"num_features": 4}
            calib_info = {}

            image_info = {"image_idx": idx}
            annotations = None
            if velodyne:
                pc_info["velodyne_path"] = osp.join(path, record['path'])
            if label_info:
                annotations = get_label_anno(record['gts'])
            info["image"] = image_info
            info["point_cloud"] = pc_info
            if calib:
                P0 = np.eye(4)
                P1 = np.eye(4)
                P2 = np.eye(4)
                P3 = np.eye(4)
                rect_4x4 = np.eye(4)
                Tr_velo_to_cam = np.eye(4)
                Tr_imu_to_velo = np.eye(4)

                calib_info["P0"] = P0
                calib_info["P1"] = P1
                calib_info["P2"] = P2
                calib_info["P3"] = P3
                calib_info["R0_rect"] = rect_4x4
                calib_info["Tr_velo_to_cam"] = Tr_velo_to_cam
                calib_info["Tr_imu_to_velo"] = Tr_imu_to_velo
                info["calib"] = calib_info

            if annotations is not None:
                info["annos"] = annotations
            image_infos.append(info)

    return image_infos

def get_label_anno(gts):
    annotations = {}
    annotations.update(
        {
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
            "index": [],
            "difficulty": [],
            "num_points_in_gt": [],
        }
    )
    
    ind = 0
    for gt in gts:
        name = gt['class_name']
        truncated = -1
        occluded = -1
        alpha = -1
        bbox = [-1, -1, -1, -1]
        dimensions = np.array([float(dim) for dim in gt['dimension']])
        location = np.array([float(loc) for loc in gt['location']])
        rotation_y = float(gt['rotation'][-1])
        difficulty = 0
        num_points_in_gt = int(gt['num_points'])
        if name == 'DontCare':
            index = -1
        else:
            index = ind
            ind += 1
        
        annotations['name'].append(name)
        annotations['truncated'].append(truncated)
        annotations['occluded'].append(occluded)
        annotations['alpha'].append(alpha)
        annotations['bbox'].append(bbox)
        annotations['dimensions'].append(dimensions)
        annotations['location'].append(location)
        annotations['rotation_y'].append(rotation_y)
        annotations['index'].append(index)
        annotations['difficulty'].append(difficulty)
        annotations['num_points_in_gt'].append(num_points_in_gt)

    annotations['name'] = np.array(annotations['name'])
    annotations['truncated'] = np.array(annotations['truncated'])
    annotations['occluded'] = np.array(annotations['occluded'])
    annotations['alpha'] = np.array(annotations['alpha'])
    annotations['bbox'] = np.array(annotations['bbox'])
    annotations['dimensions'] = np.array(annotations['dimensions'])
    annotations['location'] = np.array(annotations['location'])
    annotations['rotation_y'] = np.array(annotations['rotation_y'])
    annotations["index"] = np.array(annotations['index'], dtype=np.int32)
    annotations["difficulty"] = np.array(annotations['difficulty'], np.int32)
    annotations['num_points_in_gt'] = np.array(annotations['num_points_in_gt'], np.int32)

    num_gt = len(gts)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations








