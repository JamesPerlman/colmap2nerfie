# @title Define Scene Manager.
from collections import namedtuple
import cv2
import imageio
import numpy as np
from nerfies.camera import Camera
import pandas as pd
from pathlib import Path
from typing import Dict
from utils import pycolmap
from utils.pycolmap import Quaternion
from utils.image_utils import variance_of_laplacian

def convert_colmap_camera(colmap_camera, colmap_image):
  """Converts a pycolmap `image` to an SFM camera."""
  camera_rotation = colmap_image.R()
  camera_position = -(colmap_image.t @ camera_rotation)
  new_camera = Camera(
      orientation=camera_rotation,
      position=camera_position,
      focal_length=colmap_camera.fx,
      pixel_aspect_ratio=colmap_camera.fx / colmap_camera.fx,
      principal_point=np.array([colmap_camera.cx, colmap_camera.cy]),
      radial_distortion=np.array([colmap_camera.k1, colmap_camera.k2, 0.0]),
      tangential_distortion=np.array([colmap_camera.p1, colmap_camera.p2]),
      skew=0.0,
      image_size=np.array([colmap_camera.width, colmap_camera.height])
  )
  return new_camera


def filter_outlier_points(points, inner_percentile):
  """Filters outlier points."""
  outer = 1.0 - inner_percentile
  lower = outer / 2.0
  upper = 1.0 - lower
  centers_min = np.quantile(points, lower, axis=0)
  centers_max = np.quantile(points, upper, axis=0)
  result = points.copy()

  too_near = np.any(result < centers_min[None, :], axis=1)
  too_far = np.any(result > centers_max[None, :], axis=1)

  return result[~(too_near | too_far)]


def average_reprojection_errors(points, pixels, cameras):
  """Computes the average reprojection errors of the points."""
  cam_errors = []
  for i, camera in enumerate(cameras):
    cam_error = reprojection_error(points, pixels[:, i], camera)
    cam_errors.append(cam_error)
  cam_error = np.stack(cam_errors)

  return cam_error.mean(axis=1)


def _get_camera_translation(camera):
  """Computes the extrinsic translation of the camera."""
  rot_mat = camera.orientation
  return -camera.position.dot(rot_mat.T)


def _transform_camera(camera, transform_mat):
  """Transforms the camera using the given transformation matrix."""
  # The determinant gives us volumetric scaling factor.
  # Take the cube root to get the linear scaling factor.
  scale = np.cbrt(linalg.det(transform_mat[:, :3]))
  quat_transform = ~Quaternion.FromR(transform_mat[:, :3] / scale)

  translation = _get_camera_translation(camera)
  rot_quat = Quaternion.FromR(camera.orientation)
  rot_quat *= quat_transform
  translation = scale * translation - rot_quat.ToR().dot(transform_mat[:, 3])
  new_transform = np.eye(4)
  new_transform[:3, :3] = rot_quat.ToR()
  new_transform[:3, 3] = translation

  rotation = rot_quat.ToR()
  new_camera = camera.copy()
  new_camera.orientation = rotation
  new_camera.position = -(translation @ rotation)
  return new_camera


def _pycolmap_to_sfm_cameras(manager: pycolmap.SceneManager) -> Dict[int, Camera]:
  """Creates SFM cameras."""
  # Use the original filenames as indices.
  # This mapping necessary since COLMAP uses arbitrary numbers for the
  # image_id.
  image_id_to_colmap_id = {
      image.name.split('.')[0]: image_id
      for image_id, image in manager.images.items()
  }

  sfm_cameras = {}
  for image_id in image_id_to_colmap_id:
    colmap_id = image_id_to_colmap_id[image_id]
    image = manager.images[colmap_id]
    camera = manager.cameras[image.camera_id]
    sfm_cameras[image_id] = convert_colmap_camera(camera, image)

  return sfm_cameras

SceneTransform = namedtuple("SceneTransform", "bounding_box, scale, center")

class SceneManager:
  """A thin wrapper around pycolmap."""

  @classmethod
  def from_pycolmap(cls, colmap_path, image_path, min_track_length=10):
    """Create a scene manager using pycolmap."""
    manager = pycolmap.SceneManager(str(colmap_path))
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    manager.filter_points3D(min_track_len=min_track_length)
    sfm_cameras = _pycolmap_to_sfm_cameras(manager)
    return cls(sfm_cameras, manager.get_filtered_points3D(), image_path)

  def __init__(self, cameras, points, image_path):
    self.image_path = Path(image_path)
    self.camera_dict = cameras
    self.points = points
    print(f"Created scene manager with {len(self)} cameras")

  def __len__(self):
    return len(self.camera_dict)

  @property
  def image_ids(self):
    return sorted(self.camera_dict.keys())

  @property
  def camera_list(self):
    return [self.camera_dict[i] for i in self.image_ids]

  @property
  def camera_positions(self):
    """Returns an array of camera positions."""
    return np.stack([camera.position for camera in self.camera_list])

  def path_to_image(self, image_id) -> Path:
    return self.image_path / f'{image_id}.png'

  def load_image(self, image_id):
    """Loads the image with the specified image_id."""
    path = self.path_to_image(image_id)
    with path.open('rb') as f:
      return imageio.imread(f)

  def triangulate_pixels(self, pixels):
    """Triangulates the pixels across all cameras in the scene.

    Args:
      pixels: the pixels to triangulate. There must be the same number of pixels
        as cameras in the scene.

    Returns:
      The 3D points triangulated from the pixels.
    """
    if pixels.shape != (len(self), 2):
      raise ValueError(
          f'The number of pixels ({len(pixels)}) must be equal to the number '
          f'of cameras ({len(self)}).')

    return triangulate_pixels(pixels, self.camera_list)

  def change_basis(self, axes, center):
    """Change the basis of the scene.

    Args:
      axes: the axes of the new coordinate frame.
      center: the center of the new coordinate frame.

    Returns:
      A new SceneManager with transformed points and cameras.
    """
    transform_mat = np.zeros((3, 4))
    transform_mat[:3, :3] = axes.T
    transform_mat[:, 3] = -(center @ axes)
    return self.transform(transform_mat)

  def transform(self, transform_mat):
    """Transform the scene using a transformation matrix.

    Args:
      transform_mat: a 3x4 transformation matrix representation a
        transformation.

    Returns:
      A new SceneManager with transformed points and cameras.
    """
    if transform_mat.shape != (3, 4):
      raise ValueError('transform_mat should be a 3x4 transformation matrix.')

    points = None
    if self.points is not None:
      points = self.points.copy()
      points = points @ transform_mat[:, :3].T + transform_mat[:, 3]

    new_cameras = {}
    for image_id, camera in self.camera_dict.items():
      new_cameras[image_id] = _transform_camera(camera, transform_mat)

    return SceneManager(new_cameras, points, self.image_path)

  def filter_images(self, image_ids):
    num_filtered = 0
    for image_id in image_ids:
      if self.camera_dict.pop(image_id, None) is not None:
        num_filtered += 1

    return num_filtered

  def filter_out_blurry_images(self, blur_filter_percent: float = 95):
    if blur_filter_percent > 0.0:
      image_paths = sorted(self.image_path.iterdir())
      print('Loading images.')
      images = list(map(self.load_image, self.image_ids))
      print('Computing blur scores.')
      blur_scores = np.array([variance_of_laplacian(im) for im in images])
      blur_thres = np.percentile(blur_scores, blur_filter_percent)
      blur_filter_inds = np.where(blur_scores >= blur_thres)[0]
      blur_filter_scores = [blur_scores[i] for i in blur_filter_inds]
      blur_filter_inds = blur_filter_inds[np.argsort(blur_filter_scores)]
      blur_filter_scores = np.sort(blur_filter_scores)
      blur_filter_image_ids = [self.image_ids[i] for i in blur_filter_inds]
      print(f'Filtering {len(blur_filter_image_ids)} IDs: {blur_filter_image_ids}')
      num_filtered = self.filter_images(blur_filter_image_ids)
      print(f'Filtered {num_filtered} images')

  def estimate_near_far_for_image(self, image_id: int):
    """Estimate near/far plane for a single image based via point cloud."""
    points = filter_outlier_points(self.points, 0.95)
    points = np.concatenate([
        points,
        self.camera_positions,
    ], axis=0)
    camera = self.camera_dict[image_id]
    pixels = camera.project(points)
    depths = camera.points_to_local_points(points)[..., 2]

    # in_frustum = camera.ArePixelsInFrustum(pixels)
    in_frustum = (
        (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] <= camera.image_size_x)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] <= camera.image_size_y))
    depths = depths[in_frustum]

    in_front_of_camera = depths > 0
    depths = depths[in_front_of_camera]

    near = np.quantile(depths, 0.001)
    far = np.quantile(depths, 0.999)

    return near, far


  def estimate_near_far(self):
    """Estimate near/far plane for a set of randomly-chosen images."""
    # image_ids = sorted(self.images.keys())
    image_ids = self.image_ids
    rng = np.random.RandomState(0)
    image_ids = rng.choice(
        image_ids, size=len(self.camera_list), replace=False)
    
    result = []
    for image_id in image_ids:
      near, far = self.estimate_near_far_for_image(image_id)
      result.append({'image_id': image_id, 'near': near, 'far': far})
    result = pd.DataFrame.from_records(result)
    return result
  
  def get_bounding_box(self):
    filtered_points = filter_outlier_points(self.points, 0.95)
    points = np.concatenate([filtered_points, self.camera_positions], axis=0)  
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return np.stack([lower, upper])

  def get_scene_transform(self) -> SceneTransform:
    points = filter_outlier_points(self.points, 0.95)
    bounding_box = self.get_bounding_box()
    center = np.mean(bounding_box, axis=0)
    scale = 1.0 / np.sqrt(np.sum((bounding_box[1] - bounding_box[0]) ** 2))

    return SceneTransform(bounding_box, scale, center)

