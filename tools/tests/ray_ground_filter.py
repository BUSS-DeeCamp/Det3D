import math
import sys
import time

import numpy as np
import open3d as o3d
from loguru import logger as logging


class PointXYZRA(object):
    def __init__(self, original_index, radius, angle, height, angle_bin_index):
        self.original_index = original_index
        self.radius = radius
        self.angle = angle
        self.height = height
        self.angle_bin_index = angle_bin_index


class RayGroundFilter(object):

    def __init__(self, refinement_mode=None):
        self.radial_bin_angle = 0.4  # deg
        self.sensor_height = 0.0  # m
        self.min_local_height_threshold = 0.1  # m
        self.local_max_slope_angle = 10.0  # deg
        self.global_max_slope_angle = 5.0  # deg
        self.reclassify_distance_threshold = 0.2  # m

        self.refinement_mode = refinement_mode
        self.refinement_window_size = 0.5  # m
        self.refinement_non_ground_ratio_threshold = 0.1
        self.refinement_nearest_distance_threshold = 0.3  # m

        self.cloud = None
        self.cloud_formatted = None

    def reformat(self):

        # compute params for bin formatting
        bin_num = math.ceil(360.0 / self.radial_bin_angle)
        radius = np.linalg.norm(self.cloud[:, :2], axis=1)
        angles = np.arctan2(self.cloud[:, 1], self.cloud[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)  # convert to range: [0,  2*pi)
        angle_bin_indices = np.floor(np.degrees(angles) / self.radial_bin_angle).astype(int)

        logging.info('Total points: {}'.format(self.cloud.shape[0]))

        # add points
        self.cloud_formatted = [list() for j in range(bin_num)]

        for i in range(self.cloud.shape[0]):
            point = PointXYZRA(i, radius[i], angles[i], self.cloud[i, 2], angle_bin_indices[i])
            self.cloud_formatted[angle_bin_indices[i]].append(point)

        # sort each bin with radius in ascending order
        for i in range(bin_num):
            self.cloud_formatted[i].sort(key=lambda pt: pt.radius)

    def filter(self, cloud):

        self.cloud = cloud

        # reformat cloud data to (radius, angle)
        self.reformat()

        # classify
        ground_points, non_ground_points = self.classify()

        return ground_points, non_ground_points

    def classify(self):

        self.ground_indices = list()
        self.non_ground_indices = list()

        logging.info(
            'Using refinement mode: {}'.format(self.refinement_mode if self.refinement_mode is not None else 'None'))

        # iter for each bin
        for bin in self.cloud_formatted:
            self.process_bin(bin)

        ground_points = self.cloud[self.ground_indices, :]
        non_ground_points = self.cloud[self.non_ground_indices, :]

        if self.refinement_mode is 'nearest_neighbor':
            # compute distance from ground points to non-ground points
            ground_pcd = o3d.geometry.PointCloud()
            ground_pcd.points = o3d.utility.Vector3dVector(self.cloud[self.ground_indices, :3])

            non_ground_pcd = o3d.geometry.PointCloud()
            non_ground_pcd.points = o3d.utility.Vector3dVector(self.cloud[self.non_ground_indices, :3])

            # get those ground points needed to be changed to non-ground points
            ground_to_non_ground_distance = np.asarray(ground_pcd.compute_point_cloud_distance(non_ground_pcd))
            change_ground_flag_mask = ground_to_non_ground_distance < self.refinement_nearest_distance_threshold
            non_change_ground_flag_mask = np.logical_not(change_ground_flag_mask)

            # update ground and non-ground points
            non_ground_points = np.append(non_ground_points, ground_points[change_ground_flag_mask], axis=0)
            ground_points = ground_points[non_change_ground_flag_mask]

        return ground_points, non_ground_points

    def process_bin(self, bin):

            # [for debug use]
            bin_point_radius = list()
            bin_point_heights = list()
            bin_ground_flags = list()
            bin_point_color = list()

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
                bin_ground_flags.append(current_ground)

                # update other states
                prev_ground = current_ground
                prev_radius = pt.radius
                prev_height = pt.height

            # [for debug use]
            # for i in range(len(bin)):
            #     bin_point_radius.append(bin[i].radius)
            #     bin_point_heights.append(bin[i].height)
            #     color = [0.0, 1.0, 0.0] if bin_ground_flags[i] else [1.0, 0.0, 0.0]
            #     bin_point_color.append(color)
            # plt.scatter(bin_point_radius, bin_point_heights, c=bin_point_color)
            # plt.show()

            if self.refinement_mode is 'sliding_window':

                # revisit the points for further refinement with sliding window voting
                window_start_index = 0
                window_start_radius = 0.0
                ground_points_in_window = 0
                non_ground_points_in_window = 0
                refine_init = False

                for i in range(len(bin)):

                    pt = bin[i]
                    ground_flag = bin_ground_flags[i]

                    # use the first point to initialize the window
                    if not refine_init:
                        window_start_index = i
                        window_start_radius = pt.radius

                        if ground_flag:
                            ground_points_in_window = 1
                        else:
                            non_ground_points_in_window = 1

                        refine_init = True
                        continue

                    # check if current point is in the window
                    if pt.radius < window_start_radius + self.refinement_window_size:
                        if ground_flag:
                            ground_points_in_window += 1
                        else:
                            non_ground_points_in_window += 1
                    else:
                        # vote for classification
                        total_points_in_window = ground_points_in_window + non_ground_points_in_window
                        vote_for_non_ground = non_ground_points_in_window / total_points_in_window

                        if vote_for_non_ground > self.refinement_non_ground_ratio_threshold:
                            for j in range(window_start_index, i):
                                bin_ground_flags[j] = False
                        else:
                            for j in range(window_start_index, i):
                                bin_ground_flags[j] = True

                        # logging.info('pt #{}, window: {}-{}, ratio: {}'.format(
                        #     i, window_start_index, i - 1, vote_for_non_ground))

                        # reset window status
                        window_start_index = i
                        window_start_radius = pt.radius
                        if ground_flag:
                            ground_points_in_window = 1
                            non_ground_points_in_window = 0
                        else:
                            ground_points_in_window = 0
                            non_ground_points_in_window = 1

            # [for debug use]
            # bin_point_radius.clear()
            # bin_point_heights.clear()
            # bin_point_color.clear()
            # for i in range(len(bin)):
            #     bin_point_radius.append(bin[i].radius)
            #     bin_point_heights.append(bin[i].height)
            #     for ground_flag in bin_ground_flags:
            #         color = [0.0, 1.0, 0.0] if ground_flag else [1.0, 0.0, 0.0]
            #         bin_point_color.append(color)
            # plt.scatter(bin_point_radius, bin_point_heights, c=bin_point_color[:len(bin_point_radius)])
            # plt.show()

            # add bin results
            for pt, ground_flag in zip(bin, bin_ground_flags):
                if ground_flag:
                    self.ground_indices.append(pt.original_index)
                else:
                    self.non_ground_indices.append(pt.original_index)


class Visualizer(object):

    def __init__(self):
        self.view_text = ['raw cloud', 'ground cloud', 'non-ground cloud']
        self.view_flag = [False, False, True]
        self.view_index = 2

        self.geometries = None

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(key=ord("S"), callback_func=self.key_callback_to_switch_view)
        logging.info('Press \'s\' to switch view from raw, ground and non-ground points.')

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
        # rgf = RayGroundFilter()
        # rgf = RayGroundFilter(refinement_mode='sliding_window')
        rgf = RayGroundFilter(refinement_mode='nearest_neighbor')

        # get output ground plane and points
        tic = time.time()

        ground_points, non_ground_points = rgf.filter(cloud)

        toc = time.time()
        logging.info("Filtering time: {} s".format(toc - tic))

        # visualization
        vis = Visualizer()
        vis.show(cloud, ground_points, non_ground_points)
