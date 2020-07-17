import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from loguru import logger as logging

from tools.tests.ray_ground_filter import RayGroundFilter


class Splitter(object):

    def __init__(self, working_dir):
        self.working_dir = working_dir

    def split_cloud(self, cloud_file):

        # load cloud data
        cloud = np.fromfile(cloud_file, dtype=np.float32, count=-1).reshape([-1, 4])

        # remove points far under or over ground
        z_filt = np.logical_and(cloud[:, 2] > -10.0, cloud[:, 2] < 30.0)
        cloud = cloud[z_filt, :]

        # create ground filter
        # rgf = RayGroundFilter()
        # rgf = RayGroundFilter(refinement_mode='sliding_window')
        rgf = RayGroundFilter(refinement_mode='nearest_neighbor')

        ground_points, non_ground_points = rgf.filter(cloud)

        # save output cloud files
        working_dir = Path(self.working_dir)
        filename = Path(cloud_file).name
        ground_cloud_file = working_dir.joinpath('ground_part', filename)
        non_ground_cloud_file = working_dir.joinpath('non_ground_part', filename)

        ground_points.tofile(ground_cloud_file)
        non_ground_points.tofile(non_ground_cloud_file)

        logging.info('Ground cloud is saved to: {}'.format(ground_cloud_file))
        logging.info('Non-ground cloud is saved to: {}'.format(non_ground_cloud_file))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logging.error('Please give a path to a folder including cloud .bin files and the output folder as the argument.')
    else:
        data_folder = sys.argv[1]
        working_dir = sys.argv[2]

        # collect cloud files
        cloud_files = list()
        for filename in Path(data_folder).rglob('*.bin'):
            cloud_files.append(filename)
        cloud_files.sort()

        total_num = len(cloud_files)
        logging.info('Total cloud file num: {}'.format(total_num))

        # create output folder
        output = Path(working_dir)
        output.joinpath('ground_part').mkdir(parents=True, exist_ok=True)
        output.joinpath('non_ground_part').mkdir(parents=True, exist_ok=True)

        # process and save results
        sp = Splitter(working_dir)

        p = Pool()
        p.map(sp.split_cloud, cloud_files)

        logging.info('Finished!')
