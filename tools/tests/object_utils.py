import time

import numpy as np
import open3d as o3d


class Box(object):
    def __init__(self, location, rotation, dimension):
        self.location = location
        self.rotation = rotation
        self.dimension = dimension


class ObjectWithBox(object):
    def __init__(self, cloud_points, box3d):
        # the origin of points is [0, 0, 0], while the origin of the 3D box records its original position
        self.cloud_points = cloud_points  # numpy.ndarray(N, 3)
        self.box3d = box3d


class ObjectManipulator(object):
    box_colors = {
        'Car': [1, 0.5, 0],  # orange
        'Truck': [1, 0, 1],  # magenta
        'Tricar': [1, 1, 1],  # white
        'Cyclist': [0, 1, 0],  # green
        'Pedestrian': [1, 0, 0],  # red
        'DontCare': [0.3, 0.3, 0.3]  # gray
    }

    class_ids = {
        'Car': 0,
        'Truck': 1,
        'Tricar': 2,
        'Cyclist': 3,
        'Pedestrian': 4,
        'DontCare': 5
    }

    def __init__(self):
        self.object = None
        self.class_name = None

        # open3d cloud
        self.object_cloud = None

        # transformation between original lidar frame and current frame
        self.transform_origin_lidar_to_current = np.eye(4)
        self.transform_current_to_origin_lidar = np.eye(4)

        # lidar ring index and elevation angle list
        self.lidar_ring_index = None
        self.lidar_elevation_angle = None
        self.lidar_azimuth_angle_start = -np.pi
        self.lidar_azimuth_angle_increment = None
        self.lidar_azimuth_angle_num = None
        self.lidar_elevation_angle_num = None

    def init_object(self, object, class_name):
        self.object = object
        self.class_name = class_name

        # construct open3d cloud
        self.object_cloud = o3d.geometry.PointCloud()
        self.object_cloud.points = o3d.utility.Vector3dVector(object.cloud_points[:, :3])

    def init_lidar_transform(self, lidar_rotation=None, lidar_location=None):

        if lidar_location is None:
            lidar_location = [0, 0, 0]
        if lidar_rotation is None:
            lidar_rotation = [0, 0, 0]

        self.transform_origin_lidar_to_current = np.eye(4)
        self.transform_origin_lidar_to_current[:3, :3] = \
            o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz(lidar_rotation)
        self.transform_origin_lidar_to_current[:3, 3] = lidar_location

        self.transform_current_to_origin_lidar = np.linalg.inv(self.transform_origin_lidar_to_current)
        # print('Current frame to original lidar frame: \n {}'.format(self.transform_current_to_origin_lidar))

    def init_lidar_param(self, ring_index, elevation_angle, azimuth_angle_increment):
        self.lidar_ring_index = ring_index
        self.lidar_elevation_angle = elevation_angle
        self.lidar_azimuth_angle_increment = azimuth_angle_increment
        self.lidar_azimuth_angle_num = int(360.0 / self.lidar_azimuth_angle_increment) + 1
        self.lidar_elevation_angle_num = len(self.lidar_elevation_angle)

    # rotate and elevate the object in the frame of itself
    def self_rotate_and_elevate_object(self, rotation_z_angle=0.0, elevation_angle=0.0):
        # rotate points
        # -- apply rotation along z-axis
        self.object_cloud.rotate(
            o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz([0, 0, np.radians(rotation_z_angle)]), center=False)

        # elevate points
        # Note: no need to elevate points, since the origin of points are locate at the origin of 3D box

        # update object points in numpy ndarray
        self.object.cloud_points = np.asarray(self.object_cloud.points)

        # rotate box
        self.object.box3d.rotation[2] += np.radians(rotation_z_angle)

        # elevate box
        radial_distance = np.linalg.norm(self.object.box3d.location[:2])
        elevation_z = radial_distance * np.tan(np.radians(elevation_angle))
        self.object.box3d.location[2] = elevation_z + self.object.box3d.dimension[2] / 2

        return object

    # rotate and move the object in the frame of lidar sensor
    def lidar_rotate_and_move_object(self, rotation_z_angle=0.0, radial_distance=0.0):
        # rotate points
        self.object_cloud.rotate(
            o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz([0, 0, np.radians(rotation_z_angle)]), center=False)

        # move points
        # Note: no need to move points, since the origin of points are locate at the origin of 3D box

        # update object points in numpy ndarray
        self.object.cloud_points = np.asarray(self.object_cloud.points)

        # rotate box
        self.object.box3d.rotation[2] += np.radians(rotation_z_angle)

        # move box
        # -- first transform to original lidar frame
        location_homogeneous = np.append(self.object.box3d.location, 1)
        self.object.box3d.location = np.matmul(self.transform_current_to_origin_lidar, location_homogeneous)[:3]

        # -- then rotate to desired angle
        pre_xy = self.object.box3d.location[:2]
        rotated_xy = [
            pre_xy[0] * np.cos(np.radians(rotation_z_angle)) - pre_xy[1] * np.sin(np.radians(rotation_z_angle)),
            pre_xy[0] * np.sin(np.radians(rotation_z_angle)) + pre_xy[1] * np.cos(np.radians(rotation_z_angle))]

        # -- then move to desired radius
        rotated_xy_normalized = rotated_xy / np.linalg.norm(pre_xy)
        new_xy = radial_distance * rotated_xy_normalized

        self.object.box3d.location[:2] = new_xy

        # -- finally transform back to current frame
        location_homogeneous = np.append(self.object.box3d.location, 1)
        self.object.box3d.location = np.matmul(self.transform_origin_lidar_to_current, location_homogeneous)[:3]

    # add the X-Z plane mirrored points of the object to itself
    def mirror_object_points(self):
        # Here the problem can be simplified as finding the image of a point to a line, as z is not changed
        # Line: Ax + By + C = 0 (C = 0, since the origin of points is zero, the line should also pass the
        # origin) Let B = 1, then A = -y/x = -tan(theta)
        tan_theta = np.tan(self.object.box3d.rotation[2])
        line_direction = np.array([1, tan_theta])
        line_norm = np.array([-tan_theta, 1])

        num_of_points = self.object.cloud_points.shape[0]
        mirrored_points = self.object.cloud_points.copy()
        for i in range(num_of_points):
            p = self.object.cloud_points[i, :3]
            p_x = p[0]
            p_y = p[1]

            # compute foot of the perpendicular from current point
            # p_foot follows the direction of line_norm: (p_foot_y - p_y) / (p_foot_x - p_x) = line_norm_y / line_norm_x
            # p_foot is on the mirror line: p_foot_y = p_foot_x * tan(theta)
            p_foot_x = p_foot_y = 0.0
            if np.fabs(line_norm[0]) < 1e-9:  # the mirror line is the X-axis
                p_foot_x = p_x
                p_foot_y = 0.0
            else:
                p_foot_x = (p_y - line_norm[1] / line_norm[0] * p_x) / (tan_theta - line_norm[1] / line_norm[0])
                p_foot_y = p_foot_x * tan_theta

            # get the mirrored point
            p_xy_image = np.asarray([2 * p_foot_x - p_x, 2 * p_foot_y - p_y])
            mirrored_points[i, :2] = p_xy_image

        # add to origin cloud points
        self.object.cloud_points = np.append(self.object.cloud_points, mirrored_points, axis=0)
        self.object_cloud.points = o3d.utility.Vector3dVector(self.object.cloud_points[:, :3])

    # resample the object points with lidar sensor
    def resample_by_lidar(self):
        # transform the object points to lidar frame
        # -- first recover translation from the location of 3D box
        self.object_cloud.translate(self.object.box3d.location)
        # -- then apply the transformation to lidar frame
        self.object_cloud.transform(self.transform_current_to_origin_lidar)
        # -- finally update points in numpy ndarray
        self.object.cloud_points = np.asarray(self.object_cloud.points)

        # print("point num before resample: {}".format(np.asarray(self.object_cloud.points).shape[0]))

        # construct a 2D polar buffer for resampling
        # azimuth angle range: -pi ~ pi
        azimuth_angle_start = -np.pi
        azimuth_angle_num = int(360.0 / self.lidar_azimuth_angle_increment) + 1
        elevation_angle_num = len(self.lidar_elevation_angle)
        distance_buffer = np.full((azimuth_angle_num, elevation_angle_num), -1.0)

        # resample the points by taking the closest point
        XYZ_range_distances = np.linalg.norm(self.object.cloud_points[:, :3], axis=1)
        XY_range_distances = np.linalg.norm(self.object.cloud_points[:, :2], axis=1)
        azimuth_angles = np.arctan2(self.object.cloud_points[:, 1], self.object.cloud_points[:, 0])
        elevation_angles = np.arctan2(self.object.cloud_points[:, 2], XY_range_distances)

        for i in range(self.object.cloud_points.shape[0]):

            # compute azimuth index
            azimuth_index = \
                np.floor((azimuth_angles[i] - azimuth_angle_start) /
                         np.radians(self.lidar_azimuth_angle_increment)).astype('int')

            # find elevation index
            elevation_index = min(range(elevation_angle_num),
                                  key=lambda j: abs(self.lidar_elevation_angle[j] - np.degrees(elevation_angles[i])))

            # ignore points with large elevation angle difference
            elevation_angle_diff_threshold = 0.5  # degree
            if abs(self.lidar_elevation_angle[elevation_index] - np.degrees(elevation_angles[i])) > \
                    elevation_angle_diff_threshold:
                continue

            # update the distance if closer
            if distance_buffer[azimuth_index, elevation_index] < 0 or \
                    XYZ_range_distances[i] < distance_buffer[azimuth_index, elevation_index]:
                distance_buffer[azimuth_index, elevation_index] = XYZ_range_distances[i]

        # update object points with resampled one
        updated_indices = np.nonzero(distance_buffer > 0)

        # -- check if no point is left after resampling
        if len(updated_indices[0]) == 0:
            return False

        resample_points = None
        for azimuth_index, elevation_index in zip(updated_indices[0], updated_indices[1]):
            # compute point coordinates
            azimuth_angle = azimuth_angle_start + azimuth_index * np.radians(self.lidar_azimuth_angle_increment)
            elevation_angle = np.radians(self.lidar_elevation_angle[elevation_index])
            xyz_range_distance = distance_buffer[azimuth_index, elevation_index]

            x = xyz_range_distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
            y = xyz_range_distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
            z = xyz_range_distance * np.sin(elevation_angle)

            # add to buffer
            if resample_points is None:
                resample_points = [[x, y, z]]
            else:
                resample_points.append([x, y, z])

        resample_points = np.array(resample_points)

        # print("point num after resample: {}".format(resample_points.shape[0]))

        # transform back
        self.object_cloud.clear()
        self.object_cloud.points = o3d.utility.Vector3dVector(resample_points[:, :3])
        self.object_cloud.transform(self.transform_origin_lidar_to_current)
        self.object_cloud.translate(-1 * self.object.box3d.location)

        # update object points in numpy ndarray
        self.object.cloud_points = np.asarray(self.object_cloud.points)

        return True

    # get object points as numpy array
    def get_object_points_numpy(self):
        # transform by the box location
        transformation = np.eye(4)
        transformation[:3, 3] = self.object.box3d.location

        transformed_cloud = o3d.geometry.PointCloud(self.object_cloud)
        transformed_cloud.transform(transformation)

        return np.asarray(transformed_cloud.points)

    # get object label info as a dictionary
    def get_object_label(self):
        label = {'location': self.object.box3d.location.tolist(),
                 'rotation': self.object.box3d.rotation.tolist(),
                 'dimension': self.object.box3d.dimension.tolist(),
                 'class_name': self.class_name,
                 'class_id': self.class_ids[self.class_name],
                 'num_points': np.asarray(self.object_cloud.points).shape[0]}

        return label

    def convert_object_to_geometries(self):
        geometries = list()

        color = self.box_colors[self.class_name]

        # add points
        # -- transform the points with the origin of 3D box
        transformation = np.eye(4)
        transformation[:3, 3] = self.object.box3d.location

        self.object_cloud.transform(transformation)
        self.object_cloud.paint_uniform_color(color)
        geometries.append(self.object_cloud)

        # add box
        box = o3d.geometry.OrientedBoundingBox(
            center=self.object.box3d.location,
            R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(self.object.box3d.rotation),
            extent=self.object.box3d.dimension,
        )
        box.color = color
        geometries.append(box)

        # add orientation
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.1, cone_radius=0.2, cylinder_height=self.object.box3d.dimension[0] * 0.6,
            cone_height=0.5)
        arrow.paint_uniform_color(color)
        transformation[:3, :3] = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(
            [np.pi / 2, self.object.box3d.rotation[2] + np.pi / 2, 0])
        arrow.transform(transformation)
        geometries.append(arrow)

        # add ego box
        ego_box = o3d.geometry.TriangleMesh.create_box(width=4.5, height=1.8, depth=1.6)
        ego_box.compute_vertex_normals()
        ego_box.paint_uniform_color([0.3, 0.8, 0.0])
        transformation = np.eye(4)
        transformation[:3, 3] = [-4.5, -0.9, 0.0]
        ego_box.transform(transformation)
        geometries.append(ego_box)

        # add lidar sensor
        lidar_sensor = o3d.geometry.TriangleMesh.create_cylinder(radius=0.15, height=0.2)
        lidar_sensor.compute_vertex_normals()
        lidar_sensor.paint_uniform_color([0.8, 0.0, 0.0])
        lidar_origin = self.transform_origin_lidar_to_current[:3, 3]
        transformation = np.eye(4)
        transformation[:3, 3] = lidar_origin
        lidar_sensor.transform(transformation)
        geometries.append(lidar_sensor)

        # add lidar sensor frame
        lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        lidar_frame.transform(self.transform_origin_lidar_to_current)
        geometries.append(lidar_frame)

        return geometries


class Visualizer(object):

    def __init__(self):

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.register_key_callback(key=ord("Q"), callback_func=self.quit)
        print('Press Q to exit.')

    def show(self, geometries):

        self.vis.clear_geometries()

        # add frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        geometries.append(frame)

        # add geometries
        for g in geometries:
            self.vis.add_geometry(g)

        # configure view
        self.config_visualizer()

        # wait for the next
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)

    def config_visualizer(self):
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1

        vc = self.vis.get_view_control()
        vc.set_constant_z_far(10000.0)
        vc.set_constant_z_near(0.1)

    def quit(self, vis):
        self.vis.destroy_window()
        quit()


class VisualizerSequence(Visualizer):

    def __init__(self):

        Visualizer.__init__(self)
        self.vis.register_key_callback(key=ord("N"), callback_func=self.switch_to_next)
        print('Press N to next.')

        self.next = False
        self.camera_parameters = None

    def show(self, geometries):

        self.vis.clear_geometries()

        # add frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        geometries.append(frame)

        # add geometries
        for g in geometries:
            self.vis.add_geometry(g)

        # configure view
        self.config_visualizer()
        if self.camera_parameters is not None:
            self.update_camera_param()

        # wait for the next
        self.next = False
        while not self.next:
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)

    def switch_to_next(self, vis):
        # backup camera settings
        vc = self.vis.get_view_control()
        self.camera_parameters = vc.convert_to_pinhole_camera_parameters()

        # set flag
        self.next = True

    def update_camera_param(self):
        vc = self.vis.get_view_control()
        vc.convert_from_pinhole_camera_parameters(self.camera_parameters)
