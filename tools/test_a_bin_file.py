import sys
import time

import numpy as np
import open3d as o3d
import torch
from alfred.dl.torch.common import device
from alfred.fusion.common import compute_3d_box_lidar_coords
from alfred.utils.log import init_logger
from alfred.vis.pointcloud.pointcloud_vis import draw_pcs_open3d
from apex import amp
from loguru import logger as logging

from det3d.builder import build_voxel_generator, build_target_assigners
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint

init_logger()

# set box colors
box_colors = [
    [1.0, 0.5, 0.0],  # orange, Car
    [1.0, 0.0, 1.0],  # magenta, Truck
    [1.0, 1.0, 1.0],  # white, Tricar
    [0.0, 1.0, 0.0],  # green, Cyclist
    [1.0, 0.0, 0.0]  # red, Pedestrian
]


def box_to_geometries(box3d, labels, color_fade_factor=1.0):
    geometries = list()

    # get 3d boxes coordinates
    for ind in range(box3d.shape[0]):
        p = box3d[ind, :]
        xyz = np.array([p[: 3]])
        hwl = np.array([p[3: 6]])
        r_y = [-p[6]]  # it seems the angle needs to be inverted in the visualization
        pts3d = compute_3d_box_lidar_coords(xyz, hwl, angles=r_y, origin=(0.5, 0.5, 0.5), axis=2)[0]
        lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        color = np.asarray(box_colors[labels[ind]])
        color *= color_fade_factor
        colors = [color for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts3d)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    return geometries


def convert_detection_to_geometries(points, box3d, labels):
    geometries = list()

    # add points first
    pcs = np.array(points[:, :3])
    pcobj = o3d.geometry.PointCloud()
    pcobj.points = o3d.utility.Vector3dVector(pcs)
    geometries.append(pcobj)

    # add boxes
    geometries += box_to_geometries(box3d, labels)

    return geometries


def visualize_deecamp(points, box3d, labels):
    geometries = convert_detection_to_geometries(points, box3d, labels)
    draw_pcs_open3d(geometries)


class Deecamp3DDector(object):

    def __init__(self, config, model_p, calib_data=None):
        self.config = config
        self.model_p = model_p
        self.calib_data = calib_data
        self._init_model()

    def _init_model(self):
        self.net = build_detector(self.config.model, train_cfg=None, test_cfg=self.config.test_cfg).to(device).eval()

        # use mixed precision
        opt_level = 'O1'
        self.net = amp.initialize(self.net, opt_level=opt_level)

        checkpoint = load_checkpoint(self.net, self.model_p, map_location="cpu")

        # create voxel_generator
        self.voxel_generator = build_voxel_generator(self.config.voxel_generator)
        logging.info('network done, voxel done.')

        # create target_assigners
        target_assigners = build_target_assigners(self.config.assigner)

        # generate anchors
        # -- calculate output featuremap size
        self.grid_size = self.voxel_generator.grid_size
        feature_map_size = self.grid_size[:2] // self.config.assigner.out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]

        anchors_by_task = [
            t.generate_anchors(feature_map_size) for t in target_assigners
        ]
        anchors = [
            t["anchors"].reshape([-1, t["anchors"].shape[-1]]) for t in anchors_by_task
        ]

        self.anchors = list()
        for anchor in anchors:
            self.anchors.append(torch.tensor(anchor, dtype=torch.float32, device=device).view(1, -1, 7))

        logging.info('anchors generated.')

    @staticmethod
    def load_pc_from_deecamp_file(pc_f):
        logging.info('loading pc from: {}'.format(pc_f))
        return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 4])

    def load_an_in_example_from_points(self, points):
        voxels, coords, num_points = self.voxel_generator.generate(points)
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)  # add batch idx to coords
        num_voxels = np.array([voxels.shape[0]], dtype=np.int32)

        # convert to tensor
        voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
        coords = torch.tensor(coords, dtype=torch.int32, device=device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=device)

        return {
            'anchors': self.anchors,
            'voxels': voxels,
            'num_points': num_points,
            'num_voxels': num_voxels,
            'coordinates': coords,
            'shape': [self.grid_size]
        }

    def predict_on_deecamp_local_file(self, v_p):
        points = self.load_pc_from_deecamp_file(v_p)[:, :4]
        logging.info('points shape: {}'.format(points.shape))

        # remove points far under the ground
        z_filt = points[:, 2] > -10.0
        points = points[z_filt, :]

        # make the example
        example = self.load_an_in_example_from_points(points)

        # infer
        tic = time.time()
        with torch.no_grad():
            pred = self.net(example, return_loss=False)[0]

        # measure elapsed time
        logging.info("Predict time: {:.3f}".format(time.time() - tic))

        box3d = pred['box3d_lidar'].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["label_preds"].detach().cpu().numpy()

        # filter results by the confidence score
        idx = np.where(scores > 0.11)[0]
        box3d = box3d[idx, :]
        labels = np.take(labels, idx)
        scores = np.take(scores, idx)

        return points, box3d, labels


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('Please give a cloud .bin file as the argument.')
    else:
        config_file = 'examples/second/configs/deepcamp_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py'
        config = Config.fromfile(config_file)
        model_file = 'res/latest.pth'
        detector = Deecamp3DDector(config, model_file)
        points, box3d, labels = detector.predict_on_deecamp_local_file(sys.argv[1])
        visualize_deecamp(points, box3d, labels)
