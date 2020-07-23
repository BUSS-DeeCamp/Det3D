import sys
import json
import pickle
import random
from pathlib import Path
import csv
import hashlib

import numpy as np
import open3d as o3d
from loguru import logger as logging
from tools.tests.ray_ground_filter import RayGroundFilter
from tools.tests.object_utils import Box, ObjectWithBox, ObjectManipulator, VisualizerSequence

# set seed for debug
seed = random.randrange(sys.maxsize)
# seed = 1000
random.seed(seed)
logging.info('Random seed: {}'.format(seed))


class SceneGenerator(object):

    def __init__(self, cloud_data_folder, output_folder):
        self.cloud_data_folder = cloud_data_folder
        self.output_folder = output_folder

        self.output_cloud_file = None  # path to save output cloud .bin file
        self.output_label_file = None  # path to save output label .txt file
        self.label_data_dict = None  # label data dict of the original scene
        self.scene_labels = None  # label data dict of the generated scene

        self.output_file_name = None

        self.cloud = None  # cloud as numpy ndarray type
        self.pcd = None  # cloud as Open3d type
        self.scene_points = None  # generated scene cloud as numpy ndarray type

        self.point_distance_buffer = None
        self.lidar_mask_buffer = None

        self.selected_objects = list()
        self.labels_of_objects = list()
        self.labels_of_valid_objects = list()

        self.object_manipulator = None
        self.create_object_manipulator()

        # num of each classes in a scene
        self.num_of_objects = {'Car': 15, 'Truck': 5, 'Tricar': 5, 'Cyclist': 10, 'Pedestrian': 10}

        # radial distance range of each classes in a scene, can be set as absolute or relative
        # -- absolute
        # self.range_of_distances = {'Car': [5.0, 100.0],
        #                            'Truck': [8.0, 120.0],
        #                            'Tricar': [5.0, 80.0],
        #                            'Cyclist': [5.0, 80.0],
        #                            'Pedestrian': [5.0, 60.0]}
        # -- relative
        self.range_of_distances = {'Car': [-10.0, 10.0],
                                   'Truck': [-10.0, 10.0],
                                   'Tricar': [-10.0, 10.0],
                                   'Cyclist': [-10.0, 10.0],
                                   'Pedestrian': [-10.0, 10.0]}

        # additional random rotation angle range applied to each object
        self.additional_rotation_range = 30.0  # deg

        # elevation angle range set to each object to control its height
        self.elevation_angle_range = 2.0  # deg

    def create_object_manipulator(self):
        # configure the object manipulator and the transform between the original lidar frame and current frame
        origin_lidar_rotation = [3.13742, -3.1309, 3.14101]
        origin_lidar_location = [-2.87509, -0.00462392, 1.83632]
        self.object_manipulator = ObjectManipulator()
        self.object_manipulator.init_lidar_transform(origin_lidar_rotation, origin_lidar_location)

        # configure lidar elevation angle distribution
        lidar_elevation_file = 'test_data/VLS-128-Figure9-8-Azimuth Offsets by Elevation.csv'
        azimuth_angle_increment = 0.2  # deg

        ring_index = list()
        elevation_angle = list()
        with open(lidar_elevation_file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            line_num = 0
            for row in csvreader:
                if line_num > 0:
                    ring_index.append(int(row[0]))
                    elevation_angle.append(float(row[1]))
                line_num += 1

        self.object_manipulator.init_lidar_param(ring_index, elevation_angle, azimuth_angle_increment)

    def remove_original_objects(self):
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.cloud[:, :3])

        # -- iterate for each object
        objs = self.label_data_dict['gts']
        for p in objs:
            # ignore DontCare objects
            if p['class_name'] == 'DontCare':
                continue

            # construct 3d box
            bbox = o3d.geometry.OrientedBoundingBox(
                center=p['location'],
                R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(p['rotation']),
                extent=p['dimension'],
            )

            # crop the object points
            object_points = self.pcd.crop(bbox)

            # check if not empty
            if np.asarray(object_points.points).shape[0] == 0:
                continue

            # remove the object with distance threshold
            dist_threshold = 1e-3  # m, it should be very small, since ideally the min distance is 0
            non_ground_to_object_distance = np.asarray(self.pcd.compute_point_cloud_distance(object_points))
            remain_points_mask = non_ground_to_object_distance > dist_threshold
            remain_points_indices = np.nonzero(remain_points_mask)
            self.pcd = self.pcd.select_down_sample(remain_points_indices[0])

    def random_select_candidate_objects(self, objects_data):
        class_num = len(objects_data)
        candidate_object_num = 0

        added_object_points_for_collision_test = o3d.geometry.PointCloud()
        logging.info('Random selecting candidate objects...')
        for i in range(class_num):
            class_name = objects_data[i]['class_name']
            samples = random.sample(objects_data[i]['objects'], self.num_of_objects[class_name])

            # randomly place the object with polar coordinates
            for sample in samples:
                # manipulate the object
                self.object_manipulator.init_object(sample, class_name)

                # -- first mirror the object points
                self.object_manipulator.mirror_object_points()

                # -- then rotate and move in lidar frame
                self.object_manipulator.lidar_rotate_and_move_object(
                    rotation_z_angle=random.uniform(0.0, 360.0),
                    radial_distance=random.uniform(self.range_of_distances[class_name][0],
                                                   self.range_of_distances[class_name][1]),
                    absolute_distance=False)

                # -- then rotate and elevate itself
                self.object_manipulator.self_rotate_and_elevate_object(
                    rotation_z_angle=random.uniform(-self.additional_rotation_range, self.additional_rotation_range),
                    elevation_angle=random.uniform(-self.elevation_angle_range, self.elevation_angle_range))

                # -- then check if collision happens with previous boxes
                if self.check_box_collision(self.object_manipulator.object.box3d,
                                            added_object_points_for_collision_test):
                    continue

                # -- finally resample with the lidar
                if not self.object_manipulator.resample_by_lidar():
                    # if failed to get resampled object, skip
                    continue

                # add object to list
                self.selected_objects.append({'class_name': class_name, 'object_data': self.object_manipulator.object})

                # debug
                # object_pcd = o3d.geometry.PointCloud()
                # object_pcd.points = o3d.utility.Vector3dVector(self.object_manipulator.object.cloud_points)
                # o3d.visualization.draw_geometries([object_pcd])
                
                # add label to the object
                self.labels_of_objects.append(self.object_manipulator.get_object_label())
                
                # update object id
                candidate_object_num += 1

        logging.info('Candidate object num: {}'.format(candidate_object_num))

    def sort_by_distance(self):

        object_list_for_sort = list()

        for sample, label in zip(self.selected_objects, self.labels_of_objects):
            # compute object's distance to lidar
            location_homogeneous = np.append(sample['object_data'].box3d.location, 1)
            location_in_lidar_frame = np.matmul(
                self.object_manipulator.transform_current_to_origin_lidar, location_homogeneous)
            distance_to_lidar = np.linalg.norm(location_in_lidar_frame[:2])

            object_list_for_sort.append([sample, label, distance_to_lidar])

        object_list_for_sort.sort(key=lambda object: object[2])

        for i in range(len(object_list_for_sort)):
            self.selected_objects[i] = object_list_for_sort[i][0]
            self.labels_of_objects[i] = object_list_for_sort[i][1]

    def add_object_to_scene(self, object, points_num_threshold=50):

        object_points = object['object_data'].cloud_points.copy()
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_points[:, :3])

        # transform the object points to lidar frame
        # -- first recover translation from the location of 3D box
        object_pcd.translate(object['object_data'].box3d.location)
        # -- then apply the transformation to lidar frame
        object_pcd.transform(self.object_manipulator.transform_current_to_origin_lidar)
        # -- finally update points in numpy ndarray
        object_points = np.asarray(object_pcd.points)

        # add to all cloud with occlusion handling
        # -- backup buffer in case not enough valid points
        point_distance_buffer_backup = self.point_distance_buffer.copy()

        valid_points_num = \
            self.handle_occlusion(object_points, use_lidar_mask=False, update_lidar_mask=True)

        if valid_points_num <= points_num_threshold:
            # restore previous point_distance_buffer
            self.point_distance_buffer = point_distance_buffer_backup

        return valid_points_num

    def check_box_collision(self, box3d, added_objects_points):
        # construct open3d box
        current_box = o3d.geometry.OrientedBoundingBox(
            center=box3d.location,
            R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(box3d.rotation),
            extent=box3d.dimension,
        )

        # check with previously added box points
        if np.asarray(added_objects_points.points).shape[0] > 0:
            intersection_points = added_objects_points.crop(current_box)
            if np.asarray(intersection_points.points).shape[0] > 0:
                return True

        # randomly generate points of current box
        current_box_pcd = o3d.geometry.PointCloud()
        step = 0.2
        box_points = np.mgrid[-0.5 * box3d.dimension[0]:0.5 * box3d.dimension[0]:step,
                              -0.5 * box3d.dimension[1]:0.5 * box3d.dimension[1]:step,
                              -0.5 * box3d.dimension[2]:0.5 * box3d.dimension[2]:step].T.reshape(-1, 3)
        current_box_pcd.points = o3d.utility.Vector3dVector(box_points)
        # -- transform with box's location and rotation
        transformation = np.eye(4)
        transformation[:3, :3] = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(box3d.rotation)
        transformation[:3, 3] = box3d.location
        current_box_pcd.transform(transformation)

        # check with ego box
        ego_box = o3d.geometry.OrientedBoundingBox(
            center=self.object_manipulator.transform_origin_lidar_to_current[:3, 3],
            R=self.object_manipulator.transform_origin_lidar_to_current[:3, :3],
            extent=np.array([4.5, 1.8, 1.6]) * 1.2,
        )
        # -- crop and check if intersecting
        intersection_points = current_box_pcd.crop(ego_box)
        if np.asarray(intersection_points.points).shape[0] > 0:
            return True

        # debug
        # current_box_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        # added_objects_points.paint_uniform_color([0.0, 1.0, 0.0])
        # o3d.visualization.draw_geometries([current_box_pcd, added_objects_points])

        # add points of current box to previous cloud
        added_objects_points += current_box_pcd

        return False

    def handle_occlusion(self, points, use_lidar_mask=False, update_lidar_mask=False):
        azimuth_angle_start = -np.pi
        XYZ_range_distances = np.linalg.norm(points[:, :3], axis=1)
        XY_range_distances = np.linalg.norm(points[:, :2], axis=1)
        azimuth_angles = np.arctan2(points[:, 1], points[:, 0])
        elevation_angles = np.arctan2(points[:, 2], XY_range_distances)

        valid_points_num = 0
        for i in range(points.shape[0]):

            # compute azimuth index
            azimuth_index = \
                np.floor((azimuth_angles[i] - azimuth_angle_start) /
                         np.radians(self.object_manipulator.lidar_azimuth_angle_increment)).astype('int')

            # find elevation index
            elevation_index = min(range(len(self.object_manipulator.lidar_elevation_angle)),
                                  key=lambda j:
                                  abs(self.object_manipulator.lidar_elevation_angle[j] -
                                      np.degrees(elevation_angles[i])))

            # update the distance if not masked and assigned yet
            lidar_mask = False if not use_lidar_mask else self.lidar_mask_buffer[azimuth_index, elevation_index]
            point_distance = self.point_distance_buffer[azimuth_index, elevation_index]

            if not lidar_mask and point_distance < 0:

                # update distance
                self.point_distance_buffer[azimuth_index, elevation_index] = XYZ_range_distances[i]

                if update_lidar_mask:
                    # update mask
                    self.update_lidar_mask(self.lidar_mask_buffer, azimuth_index, elevation_index, mask_window_size=5)

                # increase valid num
                valid_points_num += 1

        return valid_points_num

    def update_lidar_mask(self, lidar_mask_buffer, azimuth_index, elevation_index, mask_window_size=3):

        azimuth_angle_num = self.object_manipulator.lidar_azimuth_angle_num
        min_azimuth_index = azimuth_index - mask_window_size
        max_azimuth_index = azimuth_index + mask_window_size
        azimuth_index_range = np.array(range(min_azimuth_index, max_azimuth_index)) % azimuth_angle_num

        elevation_angle_num = self.object_manipulator.lidar_elevation_angle_num
        min_elevation_index = elevation_index - mask_window_size
        max_elevation_index = elevation_index + mask_window_size
        elevation_index_range = np.array(range(min_elevation_index, max_elevation_index)) % elevation_angle_num

        lidar_mask_buffer[np.ix_(azimuth_index_range, elevation_index_range)] = True

    def add_background_points_to_scene(self, valid_object_indices, background_pcd):
        # first remove points in objects' boxes
        for valid_object_index in valid_object_indices:
            # construct 3d box
            box_info = self.selected_objects[valid_object_index]['object_data']
            bbox = o3d.geometry.OrientedBoundingBox(
                center=box_info.box3d.location,
                R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(box_info.box3d.rotation),
                extent=box_info.box3d.dimension,
            )

            # crop the object points
            object_points = background_pcd.crop(bbox)

            if np.asarray(object_points.points).shape[0] == 0:
                # no points to remove
                continue

            # remove the object with distance threshold
            dist_threshold = 1e-3  # m, it should be very small, since ideally the min distance is 0
            non_ground_to_object_distance = np.asarray(background_pcd.compute_point_cloud_distance(object_points))
            remain_points_mask = non_ground_to_object_distance > dist_threshold
            remain_points_indices = np.nonzero(remain_points_mask)
            background_pcd = background_pcd.select_down_sample(remain_points_indices[0])

        # then transform to lidar frame
        background_pcd.transform(self.object_manipulator.transform_current_to_origin_lidar)

        # then handle the occlusion when add it to buffer using lidar mask created by objects
        background_points = np.asarray(background_pcd.points)
        valid_background_points_num = \
            self.handle_occlusion(background_points, use_lidar_mask=True, update_lidar_mask=False)

        logging.info('All background points: {}'.format(background_points.shape[0]))
        logging.info('Valid background points in the scene: {}'.format(valid_background_points_num))

    def generate_scene(self, label_data_dict, objects_data, with_ground=True):

        self.label_data_dict = label_data_dict.copy()

        # set output file
        self.output_file_name = Path(label_data_dict['path']).name

        # load cloud data
        cloud_path = Path(self.cloud_data_folder).joinpath(self.label_data_dict['path']).as_posix()
        cloud = np.fromfile(cloud_path, dtype=np.float32)
        cloud = cloud.reshape((-1, 4))

        # remove points far under or over ground
        z_filt = np.logical_and(cloud[:, 2] > -10.0, cloud[:, 2] < 30.0)
        self.cloud = cloud[z_filt, :]

        # remove objects from the original cloud
        self.remove_original_objects()

        # randomly select objects
        self.random_select_candidate_objects(objects_data)

        # sort objects by radial distance, in order to add object from close to far
        self.sort_by_distance()

        # combine all objects
        # -- construct a 2D polar buffer for occlusion handling
        # -- azimuth angle range: -pi ~ pi
        azimuth_angle_start = -np.pi
        azimuth_angle_num = self.object_manipulator.lidar_azimuth_angle_num
        elevation_angle_num = self.object_manipulator.lidar_elevation_angle_num

        # -- create XYZ-range distance buffer for each ray
        self.point_distance_buffer = np.full((azimuth_angle_num, elevation_angle_num), -1.0)
        # -- create lidar mask for handling occlusion of objects. True means occupied by objects
        self.lidar_mask_buffer = np.full((azimuth_angle_num, elevation_angle_num), False)

        logging.info('Adding objects...')
        valid_object_indices = list()
        for i in range(len(self.selected_objects)):
            object = self.selected_objects[i]
            valid_points_num_threshold = 50
            valid_points_num = self.add_object_to_scene(object, points_num_threshold=valid_points_num_threshold)
            if valid_points_num > valid_points_num_threshold:
                valid_label = self.labels_of_objects[i]
                valid_label['num_points'] = valid_points_num
                self.labels_of_valid_objects.append(valid_label)
                # logging.info("Valid object #{}, points: {}".format(i, valid_points_num))

        logging.info('Valid objects in the scene: {}'.format(len(self.labels_of_valid_objects)))
        logging.info('Objects points in the scene: {}'.format(np.count_nonzero(self.point_distance_buffer > 0)))

        if not with_ground:
            # split ground and non-ground points
            rgf = RayGroundFilter(refinement_mode='nearest_neighbor')
            ground_points, non_ground_points = rgf.filter(np.asarray(self.pcd.points))

            non_ground_pcd = o3d.geometry.PointCloud()
            non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points[:, :3])

            # show results for debug
            # o3d.visualization.draw_geometries([non_ground_pcd])

            # add the non-ground points
            self.add_background_points_to_scene(valid_object_indices, non_ground_pcd)

        else:
            # add all background points
            self.add_background_points_to_scene(valid_object_indices, self.pcd)

        # convert points from polar coordinates to XYZ
        scene_points_list = list()
        for azimuth_index, elevation_index in np.ndindex(self.point_distance_buffer.shape):

            # ignore empty buffer
            xyz_range_distance = self.point_distance_buffer[azimuth_index, elevation_index]
            if xyz_range_distance < 0:
                continue

            # compute point coordinates
            azimuth_angle = azimuth_angle_start + \
                            azimuth_index * np.radians(self.object_manipulator.lidar_azimuth_angle_increment)
            elevation_angle = np.radians(self.object_manipulator.lidar_elevation_angle[elevation_index])

            x = xyz_range_distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
            y = xyz_range_distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
            z = xyz_range_distance * np.sin(elevation_angle)

            # add to buffer
            scene_points_list.append([x, y, z])

        self.scene_points = np.array(scene_points_list)

        # transform back to current frame
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(self.scene_points[:, :3])
        scene_pcd.transform(self.object_manipulator.transform_origin_lidar_to_current)
        self.scene_points = np.asarray(scene_pcd.points)

    def save_scene_cloud_to_file(self):
        self.output_cloud_file = Path(self.output_folder).joinpath(self.output_file_name)

        # add intensity
        scene_points_with_intensity = np.zeros((self.scene_points.shape[0], 4))
        scene_points_with_intensity[:, :3] = self.scene_points.copy()

        # write
        scene_points_with_intensity.astype(np.float32).tofile(self.output_cloud_file.as_posix())
        logging.info('Scene cloud saved to: ' + self.output_cloud_file.as_posix())

    def get_scene_labels(self):

        self.scene_labels = self.label_data_dict.copy()

        # update data
        self.scene_labels['md5'] = hashlib.md5(open(self.output_cloud_file.as_posix(), 'rb').read()).hexdigest()
        self.scene_labels['gts'] = list()
        for label in self.labels_of_valid_objects:
            self.scene_labels['gts'].append(label)

        return self.scene_labels

    def save_scene_labels_to_file(self):

        scene_labels = self.get_scene_labels()

        # write
        self.output_label_file = Path(self.output_folder).joinpath(self.output_cloud_file.stem + '.txt')
        with open(self.output_label_file.as_posix(), 'w') as fout:
            json.dump(scene_labels, fout)

        logging.info('Scene labels saved to: ' + self.output_label_file.as_posix())

    def convert_scene_to_geometries(self):
        geometries = list()

        # add scene points
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(self.scene_points[:, :3])

        geometries.append(scene_pcd)

        for label in self.labels_of_valid_objects:

            # set color
            color = self.object_manipulator.box_colors[label['class_name']]

            # add box
            box = o3d.geometry.OrientedBoundingBox(
                center=label['location'],
                R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(label['rotation']),
                extent=label['dimension'],
            )
            box.color = color
            geometries.append(box)

            # add orientation
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.1, cone_radius=0.2, cylinder_height=label['dimension'][0] * 0.6,
                cone_height=0.5)
            arrow.paint_uniform_color(color)
            transformation = np.eye(4)
            transformation[:3, 3] = label['location']
            transformation[:3, :3] = o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz(
                [np.pi / 2, label['rotation'][2] + np.pi / 2, 0])
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
            lidar_origin = self.object_manipulator.transform_origin_lidar_to_current[:3, 3]
            transformation = np.eye(4)
            transformation[:3, 3] = lidar_origin
            lidar_sensor.transform(transformation)
            geometries.append(lidar_sensor)

            # add lidar sensor frame
            lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            lidar_frame.transform(self.object_manipulator.transform_origin_lidar_to_current)
            geometries.append(lidar_frame)

        return geometries

    def reset(self):
        self.output_cloud_file = None  # path to save output cloud .bin file
        self.label_data_dict = None  # label data dict of the original scene

        self.output_file_name = None

        self.cloud = None  # cloud as numpy ndarray type
        self.pcd = None  # cloud as Open3d type
        self.scene_points = None  # generated scene cloud as numpy ndarray type

        self.point_distance_buffer = None
        self.lidar_mask_buffer = None

        self.selected_objects = list()
        self.labels_of_objects = list()
        self.labels_of_valid_objects = list()

        self.create_object_manipulator()


if __name__ == '__main__':

    data_folder = '/home/data/deecamp/DeepCamp_Lidar'
    objects_folder = '/home/data/deecamp/DeepCamp_Lidar/objects'
    label_file = Path(data_folder).joinpath('labels_filer/train_filter.txt')

    output_scenes_folder = Path(data_folder).joinpath('sim_scenes')
    output_scenes_folder.mkdir(exist_ok=True)

    # load objects
    object_files = Path(objects_folder).glob('*.bin')
    objects_data = list()
    for file in object_files:
        with open(file.as_posix(), 'rb') as f:
            objects_data.append(pickle.load(f))

    # load labels
    labels = []
    with open(label_file) as rf:
        for line in rf:
            data_dict = json.loads(line.strip())
            labels.append(data_dict)

    logging.info('Total labeled scenes: {}'.format(len(labels)))

    # create scene generator
    sg = SceneGenerator(data_folder, output_scenes_folder)

    # create visualizer
    vis = VisualizerSequence()

    # iterate for each original cloud
    scene_num = 0
    for data_dict in labels:

        logging.info("Scene #{}".format(scene_num))

        # select a subset of all objects
        objects_data_subset = list()
        subset_size = 50
        for i in range(len(objects_data)):
            class_name = objects_data[i]['class_name']
            samples = random.sample(objects_data[i]['objects'], subset_size)
            objects_data_subset.append({'class_name': class_name, 'objects': samples})

        # generate a scene
        sg.generate_scene(data_dict, objects_data_subset)

        # save scene cloud to .bin file
        sg.save_scene_cloud_to_file()

        # save scene labels to .txt file
        sg.save_scene_labels_to_file()

        # visualization
        geometries = sg.convert_scene_to_geometries()
        vis.show(geometries)

        # reset scene generator
        sg.reset()

        scene_num += 1
