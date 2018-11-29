# Copyright 2004-present Facebook. All Rights Reserved.
"""Point Cloud in 3D Space."""

import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt

from sumo.base.vector cimport array3, vector3, array_of_vector3s, vector3s_of_array
from sumo.base.vector import Vector2
from sumo.base.vector import Vector3
from sumo.threedee.point_cloud_ply import read_points
from sumo.threedee.point_cloud_ply import write_points, write_points_and_colors


# Convert single color from C++ to Python
cdef np.ndarray numpy_of_color(CColor color):
  return np.array([color.r, color.g, color.b], dtype=np.uint8)

# Convert single color from Python to C++
cdef CColor color_of_numpy(np.ndarray color):
  cdef CColor c
  c.r = color[0]
  c.g = color[1]
  c.b = color[2]
  return c

# Convert colors from C++ to Python
cdef np.ndarray array_of_colors(vector[CColor] colors):
  cdef size_t n = colors.size()
  result = np.empty((3,n), dtype=np.uint8)
  for i in range(n):
    result[:,i] = numpy_of_color(colors[i])
  return result

# Convert colors from Python to C++
cdef vector[CColor] colors_of_array(np.ndarray colors):
  cdef vector[CColor] cpp_colors
  cdef CColor cpp_color
  for i in range(colors.shape[1]):
      cpp_color = color_of_numpy(colors[:,i])
      cpp_colors.push_back(cpp_color)
  return cpp_colors

# This is the python wrapper class which dispatches to the C++ class
cdef class PointCloud:
  def __init__(self,
                points = None,
                colors = None):
    cdef vector[CVector3] cpp_points
    cdef vector[CColor] cpp_colors
    if points is not None:
        assert points.ndim == 2
        assert points.shape[0] == 3
        cpp_points = vector3s_of_array(points)
    if colors is not None:
        assert points.shape == colors.shape
        cpp_colors = colors_of_array(colors)
        self._c_ptr = new CPointCloud(cpp_points, cpp_colors)
    else:
        self._c_ptr = new CPointCloud(cpp_points)

  def __dealloc__(self):
      del self._c_ptr

  @classmethod
  def load_ply(cls, filename):
      """ Load a ply file with the given filename """
      points, num_of_points = read_points(filename)
      assert num_of_points != 0, "Reading Failed !! no points"
      return cls(points)

  @classmethod
  def merge(cls, first, second):
      """ Return a new instance that merges the two point clouds."""
      if first.colored() and not second.colored():
          raise ValueError("Can't merge colored point cloud with non-colored point cloud.")
      result = cls()
      result.append(first)
      result.append(second)
      return result

  @classmethod
  def register(cls, iterator):
      """ Given an iterator that (Pose3, PointCloud) pairs, where each Pose3 is a
          transform from cloud to the world frame, create a merged point cloud in
          the world coordinate frame.
      """
      cdef PointCloud w_point_cloud
      cdef const CPointCloud* cloud_ptr
      cdef vector[const CPointCloud*] clouds
      point_clouds = [] # to keep python objects from being garbage-collected
      for wTc, c_point_cloud in iterator:
          w_point_cloud = c_point_cloud.transform_from(wTc)
          cloud_ptr = w_point_cloud._c_ptr
          clouds.push_back(cloud_ptr)
          point_clouds.append(w_point_cloud)
      wrapper = <PointCloud>PointCloud.__new__(PointCloud)
      wrapper._c_ptr = new CPointCloud(clouds)
      return wrapper # python objects will now be destroyed

  def __add__(self, other):
      """ Return a new instance that merges the two point clouds."""
      if isinstance(other, int):
          return self # this only happens when we call sum
      elif isinstance(self, int):
          return other
      else:
          return PointCloud.merge(self, other)

  def __radd__(self, other):
      # to make 'sum' work, see https://stackoverflow.com/questions/1218710
      return self.__add__(other)

  def append(self, PointCloud other):
      """ Append the points (and colors) of the other instance. Imperative."""
      self._c_ptr.append(other._c_ptr[0])

  def point(self, size_t i):
    return array3(self._c_ptr.point(i))

  def points(self):
      return array_of_vector3s(self._c_ptr.points())

  def color(self, size_t i):
    return numpy_of_color(self._c_ptr.color(i))

  def colors(self):
      return array_of_colors(self._c_ptr.colors())

  def num_points(self):
    return self._c_ptr.numPoints()

  def colored(self):
    return self._c_ptr.colored()

  def thin(self, num_to_keep):
      """Randomly select <num_to_keep> points to keep."""
      idx = np.random.randint(self.num_points(), size=num_to_keep)
      points = self.points()[:, idx]
      colors = self.colors()[:, idx] if self.colored() else None
      return PointCloud(points, colors)

  def transform_from(self, aTb):
      """ Transforms all points from B to A frame.
          Keyword arguments:
              aTb -- pose of frame B in frame A
          Returns new point cloud.
      """
      if self.colored():
          return PointCloud(aTb.transform_all_from(self.points()), self.colors())
      else:
          return PointCloud(aTb.transform_all_from(self.points()))

  def transform_to(self, aTb):
      """ Transforms all points from A to B frame.
          Keyword arguments:
              aTb -- pose of frame B in frame A
          Returns new point cloud.
      """
      if self.colored():
          return PointCloud(aTb.transform_all_to(self.points()), self.colors())
      else:
          return PointCloud(aTb.transform_all_to(self.points()))

  def show(self, figsize=(12, 6), colored=False, **options):
      """Show using matplotlib; does not call plt.show()."""
      fig = plt.figure(figsize=figsize)
      P = self.points()

      if colored and self.colors is not None:
          options['facecolors'] = np.transpose(self.colors).astype(np.float)/255.0
          options['edgecolors'] = 'none'

      if 'marker' not in options:
          options['marker'] = '.'

      if 's' not in options:
          options['s'] = 1

      ax = fig.add_subplot(311)
      ax.scatter(P[0,:], P[1,:], **options)
      ax.set_aspect("equal")
      ax.set_title('XY')

      ax = fig.add_subplot(312)
      ax.scatter(P[0,:], P[2,:], **options)
      ax.set_aspect("equal")
      ax.set_title('XZ')

      ax = fig.add_subplot(313)
      ax.scatter(P[1,:], P[2,:], **options)
      ax.set_aspect("equal")
      ax.set_title('YZ')

      return fig

  # TODO: Write faceTexcoords, norms
  def write_ply(self, filename):
      """ Write object info into a ply file with the given filename. """
      if self.colored():
          write_points_and_colors(self.points(), self.colors(), filename)
      else:
          write_points(self.points(), filename)

  def save(self, path, color=np.array([255, 255, 255]), keep=None):
      """ Save to a text file <path>, with given <color>."""
      points_as_rows = np.transpose(self.points())
      colors_as_rows = np.matlib.repmat(color, points_as_rows.shape[0], 1)
      stacked_rows = np.hstack([points_as_rows, colors_as_rows])
      # TODO: don't use savetxt
      np.savetxt(path, stacked_rows)
