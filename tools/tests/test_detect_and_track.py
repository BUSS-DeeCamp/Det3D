import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from alfred.utils.log import init_logger
from det3d.torchie import Config
from loguru import logger as logging

init_logger()

from tools.tests.test_a_bin_file import Deecamp3DDector, box_colors, box_to_geometries
from tools.tests.object_utils import VisualizerContinuous, generate_ego_geometries


class SimpleTrackObject(object):

    def __init__(self, timestamp, box, label, distance_thresh=5.0, window_size=15, life_period=5):
        self.label = label
        self.timestamp = timestamp
        self.last_measurement_timestamp = timestamp

        self.distance_thresh = distance_thresh
        self.window_size = window_size
        self.life_period = life_period

        self.num_of_predictions = 0
        self.history = list()
        self.history.append(box)

    def get_xyz(self):
        return self.history[-1][: 3]

    def check_alive(self):
        return (self.timestamp - self.last_measurement_timestamp) <= self.life_period

    def check_need_of_prediction(self, timestamp):
        return timestamp != self.last_measurement_timestamp

    def update_motion(self, timestamp, life_length_thresh=3):
        self.timestamp = timestamp

        # using constant velocity model if being alive long enough
        if len(self.history) > life_length_thresh:
            new_box = self.history[-1].copy()

            # calculate velocity
            cur_velocity = self.history[-1][: 3] - self.history[-2][: 3]
            pre_velocity = self.history[-2][: 3] - self.history[-3][: 3]

            # weighted smoother
            velocity = 0.8 * cur_velocity + 0.2 * pre_velocity

            # create prediction
            new_box[: 3] += velocity

            # add to history
            self.update_history(new_box)

            # update num of predictions
            self.num_of_predictions += 1

    def update_measurement(self, timestamp, new_box):
        self.timestamp = timestamp
        self.last_measurement_timestamp = timestamp
        self.update_history(new_box)

        # clear num of predictions
        self.num_of_predictions = 0

    def update_history(self, new_box):
        self.history.append(new_box)
        if len(self.history) > self.window_size:
            del self.history[0]


class SimpleTracker(object):

    def __init__(self, distance_thresh=3.0, tracking_range=100.0):
        self.distance_thresh = distance_thresh
        self.tracking_range = tracking_range

        self.objects = list()
        self.initialized = False
        self.current_timestamp = 0

    def add_object(self, obj):
        self.objects.append(obj)

    def update_object(self, timestamp, new_box, label):

        closest_obj_ind = -1
        closest_obj_dist = 1e6

        for i in range(len(self.objects)):

            cur_obj = self.objects[i]
            # check label
            if label != cur_obj.label:
                continue

            # check if in tracking range
            r = np.linalg.norm(new_box[: 3])
            if r > self.tracking_range:
                continue

            # check association by distance
            dist = np.linalg.norm(cur_obj.get_xyz() - new_box[: 3])
            if dist < self.distance_thresh and dist < closest_obj_dist:
                closest_obj_ind = i
                closest_obj_dist = dist

        if closest_obj_ind > 0:
            self.objects[closest_obj_ind].update_measurement(timestamp, new_box)
        else:
            new_obj = SimpleTrackObject(timestamp, new_box, label)
            self.add_object(new_obj)

    def get_geometries_for_visualization(self):

        geometries = list()
        prediction_geometry = None

        for obj in self.objects:

            # ignore those objects with too short life
            if len(obj.history) < 3:
                continue

            # get number of predictions
            num_of_predictions = obj.num_of_predictions
            final_color_fade_factor = 1 - (num_of_predictions / (obj.life_period + 1))

            # add histories
            for i in range(len(obj.history)):
                box = obj.history[i]
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)

                # modify the color of prediction histories
                if i < (len(obj.history) - num_of_predictions):
                    mesh_sphere.paint_uniform_color(box_colors[obj.label])
                else:
                    color_fade_factor = (len(obj.history) - i) / (obj.life_period + 1)
                    color = np.asarray(box_colors[obj.label])
                    mesh_sphere.paint_uniform_color(color * color_fade_factor)

                mesh_sphere.compute_vertex_normals()
                transformation = np.identity(4)
                transformation[:3, 3] = box[: 3]
                mesh_sphere.transform(transformation)
                geometries.append(mesh_sphere)

            # add virtual box if predicting
            if num_of_predictions > 0:
                boxes = np.expand_dims(obj.history[-1], axis=0)
                labels = np.expand_dims(np.array(obj.label), axis=0)
                prediction_geometry = box_to_geometries(boxes, labels, final_color_fade_factor)
                geometries.append(prediction_geometry[0])

        return geometries

    # refresh by prediction and cleaning, after finishing the updating
    def refresh(self, timestamp):
        self.predict_motion(timestamp)
        self.clean()

    # manually update those objects without measurement by motion model
    def predict_motion(self, timestamp):
        for obj in self.objects:
            # predict those objects without new measurements
            if obj.check_need_of_prediction(timestamp):
                obj.update_motion(timestamp)

    # manually clean the dead objects
    def clean(self):
        alive_list = list()
        for ind in range(len(self.objects)):
            if self.objects[ind].check_alive():
                alive_list.append(ind)

        self.objects = [self.objects[i] for i in alive_list]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('Please give a path to a folder including cloud .bin files as the argument.')
    else:
        # config_file = 'examples/second/configs/deepcamp_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py'
        config_file = 'res/20200717-voxel-0.12-0.12-0.1/deepcamp_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py'
        config = Config.fromfile(config_file)
        # model_file = 'res/latest.pth'
        model_file = 'res/20200717-voxel-0.12-0.12-0.1/latest.pth'
        data_folder = sys.argv[1]

        # collect cloud files
        cloud_files = list()
        for filename in Path(data_folder).rglob('*.bin'):
            cloud_files.append(filename)
        cloud_files.sort()

        total_num = len(cloud_files)
        logging.info('Total cloud file num: {}'.format(total_num))

        # create detector
        detector = Deecamp3DDector(config, model_file)

        # create tracking objects and tracker
        tracker = SimpleTracker()
        tracker_initialized = False

        # create visualizer
        vis = VisualizerContinuous()

        # start to detect & track
        timestamp = 0
        for cloud_file in cloud_files:

            # detect
            detector.predict_on_deecamp_local_file(cloud_file)

            # track
            if not tracker.initialized:
                for box, label in zip(detector.box3d, detector.labels):
                    obj = SimpleTrackObject(timestamp, box, label)
                    tracker.add_object(obj)

                tracker.initialized = True

            else:
                for box, label in zip(detector.box3d, detector.labels):
                    tracker.update_object(timestamp, box, label)

            tracker.refresh(timestamp)  # make motion prediction and remove those objects which are dead

            # visualization
            # -- get geometries
            detection_geometries = detector.convert_detection_to_geometries()
            tracking_geometries = tracker.get_geometries_for_visualization()
            ego_geometries = generate_ego_geometries()

            geometries = list()
            geometries.extend(detection_geometries)
            geometries.extend(tracking_geometries)
            geometries.extend(ego_geometries)

            # -- show
            vis.show(geometries)

            timestamp += 1

        vis.destroy_window()
