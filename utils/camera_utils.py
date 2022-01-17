import numpy as np
import math
from tensorflow_graphics.geometry.representation.ray import triangulate as ray_triangulate

from utils.scene_utils import SceneManager

_EPSILON = 1e-5

def points_bound(points):
  """Computes the min and max dims of the points."""
  min_dim = np.min(points, axis=0)
  max_dim = np.max(points, axis=0)
  return np.stack((min_dim, max_dim), axis=1)


def points_centroid(points):
  """Computes the centroid of the points from the bounding box."""
  return points_bound(points).mean(axis=1)


def points_bounding_size(points):
  """Computes the bounding size of the points from the bounding box."""
  bounds = points_bound(points)
  return np.linalg.norm(bounds[:, 1] - bounds[:, 0])


def look_at(camera,
            camera_position: np.ndarray,
            look_at_position: np.ndarray,
            up_vector: np.ndarray):
  look_at_camera = camera.copy()
  optical_axis = look_at_position - camera_position
  norm = np.linalg.norm(optical_axis)
  if norm < _EPSILON:
    raise ValueError('The camera center and look at position are too close.')
  optical_axis /= norm

  right_vector = np.cross(optical_axis, up_vector)
  norm = np.linalg.norm(right_vector)
  if norm < _EPSILON:
    raise ValueError('The up-vector is parallel to the optical axis.')
  right_vector /= norm

  # The three directions here are orthogonal to each other and form a right
  # handed coordinate system.
  camera_rotation = np.identity(3)
  camera_rotation[0, :] = right_vector
  camera_rotation[1, :] = np.cross(optical_axis, right_vector)
  camera_rotation[2, :] = optical_axis

  look_at_camera.position = camera_position
  look_at_camera.orientation = camera_rotation
  return look_at_camera

def compute_camera_rays(points, camera):
  origins = np.broadcast_to(camera.position[None, :], (points.shape[0], 3))
  directions = camera.pixels_to_rays(points.astype(jnp.float32))
  endpoints = origins + directions
  return origins, endpoints

def triangulate_rays(origins, directions):
  origins = origins[np.newaxis, ...].astype('float32')
  directions = directions[np.newaxis, ...].astype('float32')
  weights = np.ones(origins.shape[:2], dtype=np.float32)
  points = np.array(ray_triangulate(origins, origins + directions, weights))
  return points.squeeze()

def generate_camera_paths(scene_manager: SceneManager):
    print("Generating camera paths...")

    ref_cameras = [c for c in scene_manager.camera_list]
    origins = np.array([c.position for c in ref_cameras])
    directions = np.array([c.optical_axis for c in ref_cameras])
    look_at = triangulate_rays(origins, directions)

    print('look_at', look_at)

    avg_position = np.mean(origins, axis=0)

    print('avg_position', avg_position)

    up = -np.mean([c.orientation[..., 1] for c in ref_cameras], axis=0)

    print('up', up)

    bounding_size = points_bounding_size(origins) / 2
    x_scale =   0.75# @param {type: 'number'}
    y_scale = 0.75  # @param {type: 'number'}
    xs = x_scale * bounding_size
    ys = y_scale * bounding_size
    radius = 0.75  # @param {type: 'number'}
    num_frames = 100  # @param {type: 'number'}

    origin = np.zeros(3)

    ref_camera = ref_cameras[0]
    print(ref_camera.position)
    z_offset = -0.1

    angles = np.linspace(0, 2*math.pi, num=num_frames)
    positions = []
    for angle in angles:
        x = np.cos(angle) * radius * xs
        y = np.sin(angle) * radius * ys
        # x = xs * radius * np.cos(angle) / (1 + np.sin(angle) ** 2)
        # y = ys * radius * np.sin(angle) * np.cos(angle) / (1 + np.sin(angle) ** 2)

        position = np.array([x, y, z_offset])
        # Make distance to reference point constant.
        position = avg_position + position
        positions.append(position)

    positions = np.stack(positions)

    orbit_cameras = []
    for position in positions:
        camera = ref_camera.look_at(position, look_at, up)
        orbit_cameras.append(camera)

    camera_paths = {'orbit-mild': orbit_cameras}
    return camera_paths