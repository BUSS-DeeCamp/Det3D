import sys
import time

import numpy as np
import torch
from apex import amp

from det3d.builder import build_voxel_generator, build_target_assigners
from det3d.models import build_detector
from det3d.torchie import Config


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Deecamp3DDector(object):

    def __init__(self, config, calib_data=None):
        self.config = config
        self.calib_data = calib_data
        self._init_model()

    def _init_model(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.net = build_detector(self.config.model, train_cfg=None, test_cfg=self.config.test_cfg).to(
            self.device).eval()

        # use mixed precision
        opt_level = 'O1'
        self.net = amp.initialize(self.net, opt_level=opt_level)

        # create voxel_generator
        self.voxel_generator = build_voxel_generator(self.config.voxel_generator)
        print('network done, voxel done.')

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
            self.anchors.append(torch.tensor(anchor, dtype=torch.float32, device=self.device).view(1, -1, 7))

        print('anchors generated.')

    def load_an_in_example_from_points(self, points):
        voxels, coords, num_points = self.voxel_generator.generate(points)
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)  # add batch idx to coords
        num_voxels = np.array([voxels.shape[0]], dtype=np.int32)

        # convert to tensor
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)

        return {
            'anchors': self.anchors,
            'voxels': voxels,
            'num_points': num_points,
            'num_voxels': num_voxels,
            'coordinates': coords,
            'shape': [self.grid_size]
        }

    def eval_fps(self, cloud, num_of_iteration=100):

        print('points shape: {}'.format(cloud.shape))

        # make the example
        example = self.load_an_in_example_from_points(cloud)

        # infer
        infer_time = AverageMeter()

        for i in range(num_of_iteration):
            tic = time.time()
            with torch.no_grad():
                pred = self.net(example, return_loss=False)[0]

            toc = time.time()
            infer_time.update(toc - tic)
            print('#{}: {:.3f} s'.format(i, toc - tic))

        print("Average inference time: {:.3f} s".format(infer_time.avg))


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Please give a cloud .bin file as the argument.')
    else:
        config_file = 'examples/second/configs/deepcamp_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py'
        config = Config.fromfile(config_file)
        cloud = np.fromfile(sys.argv[1], dtype=np.float32, count=-1).reshape(
            [-1, 4])
        detector = Deecamp3DDector(config)
        detector.eval_fps(cloud, 200)
