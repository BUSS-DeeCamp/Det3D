import sys
import numpy as np
import open3d as o3d
from loguru import logger as logging


class GroundDetector(object):

    def __init__(self, plane_range=100.0, threshold=0.1, prior_z_min=-0.2, prior_z_max=0.2):
        self.plane_range = plane_range  # the range of points on the extracted ground plane
        self.threshold = threshold  # the threshold of ground points to the ground plane
        self.prior_z_min = prior_z_min  # the initial estimate of minimum z-component of ground points
        self.prior_z_max = prior_z_max  # the initial estimate of maximum z-component of ground points

    def compute(self, cloud):

        raw_cloud = cloud.copy()
        logging.info('Input cloud points: {}'.format(cloud.shape[0]))

        # threshold the cloud along z-axis with prior
        z_min_filter = np.abs(cloud[:, 2]) > self.prior_z_min
        z_max_filter = np.abs(cloud[:, 2]) < self.prior_z_max
        z_inliers = np.logical_and(z_min_filter, z_max_filter)
        cloud = cloud[z_inliers, :]

        logging.info('Points after z-thresholding: {}'.format(cloud.shape[0]))

        # threshold with range
        points_range = np.linalg.norm(cloud[:, :2], axis=1)
        range_inliers = points_range < self.plane_range
        cloud = cloud[range_inliers, :]

        logging.info('Points after range-thresholding: {}'.format(cloud.shape[0]))

        # estimate initial plane to prior
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])

        plane_model, inliers = pcd.segment_plane(distance_threshold=self.threshold, ransac_n=4, num_iterations=50)
        ground_inliers = pcd.select_down_sample(inliers)
        ground_outliers = pcd.select_down_sample(inliers, invert=True)

        logging.info('Estimated plane: {}, inliers: {}'.format(plane_model, len(inliers)))

        return plane_model, ground_inliers, ground_outliers


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

        self.vis.create_window()
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1

        # add geometries
        self.geometries = [raw_pcd, ground_cloud, non_ground_cloud]
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
        detector = GroundDetector()

        # get output ground plane and points
        plane, ground_points, non_ground_points = detector.compute(cloud)

        # visualization
        vis = Visualizer()
        vis.show(cloud, ground_points, non_ground_points)
