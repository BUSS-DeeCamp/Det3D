import sys
import json
from pathlib import Path

import open3d as o3d
import torch
from loguru import logger as logging

from det3d.torchie import Config

from tools.tests.test_a_bin_file import Deecamp3DDector
from tools.tests.object_utils import VisualizerSequence, generate_ego_geometries, convert_label_to_geometries

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('Please give a folder containing cloud .bin file as the argument. A label file is optional.')
    else:
        config_file = 'examples/second/configs/deepcamp_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py'
        model_file = 'res/latest.pth'

        config = Config.fromfile(config_file)

        # set label file if exists
        label_file = None
        labels = list()

        if len(sys.argv) == 3:
            label_file = sys.argv[2]
            print('Using label file: {}'.format(label_file))
            # load labels
            with open(label_file) as rf:
                for line in rf:
                    data_dict = json.loads(line.strip())
                    labels.append(data_dict)

        # collect cloud files
        cloud_folder = sys.argv[1]
        cloud_files = list()
        for filename in Path(cloud_folder).rglob('*.bin'):
            cloud_files.append(filename)
        cloud_files.sort()

        total_num = len(cloud_files)
        logging.info('Total cloud file num: {}'.format(total_num))

        # create detector
        detector = Deecamp3DDector(config, model_file)

        # visualize
        vis = VisualizerSequence()

        # evaluate model on each cloud file
        for file in cloud_files:
            # inference
            detector.predict_on_deecamp_local_file(file)

            geometries = list()

            # -- get detection geometries
            detection_geometries = detector.convert_detection_to_geometries()

            # -- find corresponding label if exists
            label_geometries = None
            label_data_dict = None

            if len(labels) > 0:
                label_data_dict = next((la for la in labels if Path(la['path']).name == file.name), None)
                if label_data_dict is None:
                    print('No label found for cloud: {}'.format(file))
                else:
                    # -- generate label geometries
                    label_geometries = convert_label_to_geometries(label_data_dict)

            # -- generate ego geometries
            ego_geometries = generate_ego_geometries()

            geometries.extend(detection_geometries)
            if label_data_dict is not None:
                geometries.extend(label_geometries)
            geometries.extend(ego_geometries)

            # show
            vis.show(geometries)
