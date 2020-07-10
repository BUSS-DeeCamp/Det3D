import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from alfred.utils.log import init_logger
from det3d.torchie import Config
from loguru import logger as logging

init_logger()

from tools.test_a_bin_file import Deecamp3DDector, convert_detection_to_geometries, box_colors


class SimpleTrackObject(object):

    def __init__(self, box, label, distance_thresh=0.5, window_size=15):
        self.box = box
        self.label = label
        self.distance_thresh = distance_thresh
        self.window_size = window_size

        self.history = list()
        self.history.append(box)

    def get_xyz(self):
        return self.box[: 3]

    def get_label(self):
        return self.label

    def get_history(self):
        return self.history

    def update_measurement(self, new_box):
        self.box = new_box
        self.history.append(new_box)
        if len(self.history) > self.window_size:
            del self.history[0]
        return


class SimpleTracker(object):

    def __init__(self, distance_thresh=3.0, tracking_range=100.0):
        self.distance_thresh = distance_thresh
        self.tracking_range = tracking_range

        self.objects = list()
        self.initialized = False

    def set_initialized(self):
        self.initialized = True

    def get_initialized_status(self):
        return self.initialized

    def add_object(self, obj):
        self.objects.append(obj)

    def get_objects(self):
        return self.objects

    def update_object(self, new_box, label):

        for cur_obj in self.objects:
            # check label
            if label != cur_obj.get_label():
                continue

            # check if in tracking range
            range = np.linalg.norm(new_box[: 3])
            if range > self.tracking_range:
                continue

            # check association by distance
            dist = np.linalg.norm(cur_obj.get_xyz() - new_box[: 3])
            if dist < self.distance_thresh:
                cur_obj.update_measurement(new_box)
                return

    def get_geometries_for_visualization(self):

        geometries = list()

        for obj in self.objects:
            label = obj.get_label()

            for box in obj.get_history():
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
                mesh_sphere.compute_vertex_normals()
                mesh_sphere.paint_uniform_color(box_colors[label])
                transformation = np.identity(4)
                transformation[:3, 3] = box[: 3]
                mesh_sphere.transform(transformation)
                geometries.append(mesh_sphere)

        return geometries


def key_callback_to_quit(vis):
    quit()


def visualize_open3d(geometries):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(key=ord("Q"), callback_func=key_callback_to_quit)
    logging.info('Press Q to exit.')

    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    for g in geometries:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()


def config_visualizer(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1

    vc = vis.get_view_control()
    camera_parameters = vc.convert_to_pinhole_camera_parameters()
    camera_parameters.extrinsic = np.array(
        [[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 100.],
         [0., 0., 0., 1.]])
    vc.convert_from_pinhole_camera_parameters(camera_parameters)
    vc.set_constant_z_far(10000.0)
    vc.set_constant_z_near(0.1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('Please give a path to a folder including cloud .bin files as the argument.')
    else:
        config_file = 'examples/second/configs/deepcamp_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py'
        config = Config.fromfile(config_file)
        model_file = 'res/latest.pth'
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
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(key=ord("Q"), callback_func=key_callback_to_quit)
        logging.info('Press Q to exit.')
        vis.create_window()

        # start to detect & track
        for cloud_file in cloud_files:
            points, boxes, labels = detector.predict_on_deecamp_local_file(cloud_file)

            if not tracker.get_initialized_status():
                for box, label in zip(boxes, labels):
                    obj = SimpleTrackObject(box, label)
                    tracker.add_object(obj)

                tracker.set_initialized()

            else:
                for box, label in zip(boxes, labels):
                    tracker.update_object(box, label)

            # visualization
            vis.clear_geometries()

            # -- get geometries
            detection_geometries = convert_detection_to_geometries(points, boxes, labels)
            tracking_geometries = tracker.get_geometries_for_visualization()

            # -- add geometries
            for g in detection_geometries:
                vis.add_geometry(g)

            for g in tracking_geometries:
                vis.add_geometry(g)

            # -- set config of visualizer
            config_visualizer(vis)

            vis.poll_events()
            vis.update_renderer()

        vis.destroy_window()
