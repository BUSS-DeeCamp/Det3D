import numpy as np
from det3d.ops.point_cloud.point_cloud_ops import points_to_voxel


def numpy_reduce_op(list_of_arr, op=np.logical_and):
  """ doing logical on more than 1 elements
  :param list_of_arr: list of numpy array
  :param op: operation on list_of_arr, currently only support np.logical_and,
  np.logical_or
  """

  assert isinstance(list_of_arr, (list, tuple)), \
    "list_of_arr only support list or tuple of numpy array"

  assert len(list_of_arr) >= 2, "only support more than 1 array"

  assert op in [np.logical_and, np.logical_or], \
    "not supported op {}".format(op)

  ret = op(list_of_arr[0], list_of_arr[1])
  for i in range(2, len(list_of_arr)):
    ret = op(ret, list_of_arr[i])

  return ret

class VoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels=20000):
        conds = []
        for i in range(3):
            cond = np.logical_and(
            points[:, i] >= self._point_cloud_range[i],
            points[:, i] <  self._point_cloud_range[i + 3]
            )
            conds.append(cond)

        final_cond = numpy_reduce_op(conds, np.logical_and)
        points = points[final_cond]
        return points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            self._max_voxels,
        )

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
