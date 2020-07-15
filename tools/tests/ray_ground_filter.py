import math
import sys

import numpy as np
import open3d as o3d
from loguru import logger as logging


class PointXYZIRA(object):
    def __init__(self, original_index, radius, angle, height, intensity, angle_bin_index):
        self.original_index = original_index
        self.radius = radius
        self.angle = angle
        self.height = height
        self.intensity = intensity
        self.angle_bin_index = angle_bin_index


class RayGroundFilter(object):

    def __init__(self):
        self.radial_bin_angle = 0.4  # deg
        self.sensor_height = 0.0  # m
        self.min_local_height_threshold = 0.1  # m
        self.local_max_slope_angle = 10.0  # deg
        self.global_max_slope_angle = 5.0  # deg
        self.reclassify_distance_threshold = 0.2  # m

        self.cloud = None
        self.cloud_formatted = None

    def reformat(self):

        # compute params for bin formatting
        bin_num = math.ceil(360.0 / self.radial_bin_angle)
        radius = np.linalg.norm(self.cloud[:, :2], axis=1)
        angles = np.arctan2(self.cloud[:, 1], self.cloud[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)  # convert to range: [0,  2*pi)
        angle_bin_indices = np.floor(np.degrees(angles) / self.radial_bin_angle).astype(int)

        # add points
        self.cloud_formatted = [list() for j in range(bin_num)]

        for i in range(self.cloud.shape[0]):
            point = PointXYZIRA(i, radius[i], angles[i], self.cloud[i, 2], self.cloud[i, 3], angle_bin_indices[i])
            self.cloud_formatted[angle_bin_indices[i]].append(point)

        # sort each bin with radius in ascending order
        for i in range(bin_num):
            self.cloud_formatted[i].sort(key=lambda pt: pt.radius)

    def classify(self):

        ground_indices = list()
        non_ground_indices = list()

        # iter for each bin
        for bin in self.cloud_formatted:

            # [for debug use]
            # bin_point_radius = list()
            # bin_point_heights = list()
            # bin_point_color = list()

            # init buffer
            prev_radius = 0.0
            prev_height = 0.0
            prev_ground = False
            current_ground = False

            # iter for each point
            for pt in bin:
                distance_to_prev = pt.radius - prev_radius
                local_height_threshold = math.tan(math.radians(self.local_max_slope_angle)) * distance_to_prev
                global_height_threshold = math.tan(math.radians(self.global_max_slope_angle)) * pt.radius
                current_height = pt.height

                # constrain the local height threshold
                if local_height_threshold < self.min_local_height_threshold:
                    local_height_threshold = self.min_local_height_threshold

                # check local height
                if math.fabs(current_height - prev_height) <= local_height_threshold:
                    # check global height if the previous point is not ground point
                    if not prev_ground:
                        # check if current point satisfy the global requirement
                        if math.fabs(current_height - self.sensor_height) < global_height_threshold:
                            current_ground = True
                        else:
                            current_ground = False
                    else:
                        current_ground = True
                else:
                    # check the point is far away enough to previous point and satisfy the global requirement
                    if distance_to_prev > self.reclassify_distance_threshold and math.fabs(
                            current_height - self.sensor_height) < global_height_threshold:
                        current_ground = True
                    else:
                        current_ground = False

                # add to buffer
                if current_ground:
                    ground_indices.append(pt.original_index)
                    prev_ground = True
                else:
                    non_ground_indices.append(pt.original_index)
                    prev_ground = False

                # update other states
                prev_radius = pt.radius
                prev_height = pt.height

                # [for debug use]
                # bin_point_radius.append(pt.radius)
                # bin_point_heights.append(pt.height)
                # color = [0.0, 1.0, 0.0] if current_ground else [1.0, 0.0, 0.0]
                # bin_point_color.append(color)
                #
                # plt.scatter(bin_point_radius, bin_point_heights, c=bin_point_color)
                # plt.show()

        return ground_indices, non_ground_indices

    def filter(self, cloud):

        self.cloud = cloud
        ground_points = None
        non_ground_points = None

        # reformat cloud data to (radius, angle)
        self.reformat()

        # classify
        ground_indices, non_ground_indices = self.classify()
        ground_points = self.cloud[ground_indices, :]
        non_ground_points = self.cloud[non_ground_indices, :]

        return ground_points, non_ground_points


class Visualizer(object):

    def __init__(self):
        self.view_text = ['raw cloud', 'ground cloud', 'non-ground cloud']
        self.view_flag = [False, False, True]
        self.view_index = 2

        self.geometries = None

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(key=ord("S"), callback_func=self.key_callback_to_switch_view)
        logging.info('Press S to switch view from raw, ground and non-ground points.')

    def key_callback_to_switch_view(self, vis):

        # switch view flag
        self.view_flag[self.view_index] = False
        self.view_index += 1
        if self.view_index > 2:
            self.view_index = 0
        self.view_flag[self.view_index] = True
        logging.info('current view: {}'.format(self.view_text[self.view_index]))

        # backup camera settings
        vc = self.vis.get_view_control()
        camera_parameters = vc.convert_to_pinhole_camera_parameters()

        # add geometries
        self.vis.clear_geometries()
        for i in range(len(self.view_flag)):
            if self.view_flag[i]:
                self.vis.add_geometry(self.geometries[i])

        # restore camera settings
        vc.convert_from_pinhole_camera_parameters(camera_parameters)

    def show(self, raw_cloud, ground_cloud, non_ground_cloud):
        raw_pcd = o3d.geometry.PointCloud()
        raw_pcd.points = o3d.utility.Vector3dVector(raw_cloud[:, :3])

        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_cloud[:, :3])

        non_ground_pcd = o3d.geometry.PointCloud()
        non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_cloud[:, :3])

        self.vis.create_window()
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1

        # add geometries
        self.geometries = [raw_pcd, ground_pcd, non_ground_pcd]
        for i in range(len(self.view_flag)):
            if self.view_flag[i]:
                self.vis.add_geometry(self.geometries[i])

        self.vis.run()
        self.vis.destroy_window()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('Please give a cloud .bin file as the argument.')
    else:
        # load cloud data
        cloud = np.fromfile(sys.argv[1], dtype=np.float32, count=-1).reshape([-1, 4])

        # remove points far under or over ground
        z_filt = np.logical_and(cloud[:, 2] > -10.0, cloud[:, 2] < 30.0)
        cloud = cloud[z_filt, :]

        # create detector
        rgf = RayGroundFilter()

        # get output ground plane and points
        ground_points, non_ground_points = rgf.filter(cloud)

        # visualization
        vis = Visualizer()
        vis.show(cloud, ground_points, non_ground_points)
