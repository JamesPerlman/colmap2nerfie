#
# Most of this code is from https://github.com/google/nerfies
# It was converted to a command-line script by James Perlman in 2022
#

import argparse
import bisect
import json
import numpy as np
from pathlib import Path
import shutil
from utils.scene_utils import *
from utils.camera_utils import generate_camera_paths

parser = argparse.ArgumentParser(description="colmap2nerf")
parser.add_argument("-i", "--input", required=True, help="Input folder of colmap project")
parser.add_argument("-o", "--output", required=True, help="Output folder for Nerfies dataset")
args = parser.parse_args()

# Setup paths

colmap_dir = Path(args.input)
images_dir = colmap_dir / "images"
cameras_dir = colmap_dir / "sparse/0"
output_dir = Path(args.output)

# Interpret scene

scene_manager = SceneManager.from_pycolmap(cameras_dir, images_dir, min_track_length=5)
scene_manager.filter_out_blurry_images()

# Compute scene and camera properties

scene_transform = scene_manager.get_scene_transform()
print(f'Scene Center: {scene_transform.center}')
print(f'Scene Scale: {scene_transform.scale}')

print('Computing near & far of scene...')
near_far = scene_manager.estimate_near_far()
near = near_far['near'].quantile(0.001) / 0.8
far = near_far['far'].quantile(0.999) * 1.2

print('Statistics for near/far computation:')
print(near_far.describe())

print('Selected near/far values:')
print(f'Near = {near:.04f}')
print(f'Far = {far:.04f}')

camera_paths = generate_camera_paths(scene_manager)

# Save scene data

shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(exist_ok=True)

scene_json_path = output_dir /  'scene.json'

with scene_json_path.open('w') as f:
  json.dump({
      'scale': scene_transform.scale,
      'center': scene_transform.center.tolist(),
      'bbox': scene_transform.bounding_box.tolist(),
      'near': near * scene_transform.scale,
      'far': far * scene_transform.scale,
  }, f, indent=2)

print(f'Saved scene information to {scene_json_path}')

# Save dataset split

all_ids = scene_manager.image_ids
val_ids = all_ids[::20]
train_ids = sorted(set(all_ids) - set(val_ids))
dataset_json = {
    'count': len(scene_manager),
    'num_exemplars': len(train_ids),
    'ids': scene_manager.image_ids,
    'train_ids': train_ids,
    'val_ids': val_ids,
}

dataset_json_path = output_dir / 'dataset.json'
with dataset_json_path.open('w') as f:
    json.dump(dataset_json, f, indent=2)

print(f'Saved dataset information to {dataset_json_path}')

# Save metadata information

metadata_json = {}

for i, image_id in enumerate(train_ids):
  metadata_json[image_id] = {
      'warp_id': i,
      'appearance_id': i,
      'camera_id': 0,
  }

for i, image_id in enumerate(val_ids):
  i = bisect.bisect_left(train_ids, image_id)
  metadata_json[image_id] = {
      'warp_id': i,
      'appearance_id': i,
      'camera_id': 0,
  }

metadata_json_path = output_dir / 'metadata.json'
with metadata_json_path.open('w') as f:
    json.dump(metadata_json, f, indent=2)

print(f'Saved metadata information to {metadata_json_path}')

# Save cameras

camera_dir = output_dir / 'camera'
camera_dir.mkdir(exist_ok=True, parents=True)

for item_id, camera in scene_manager.camera_dict.items():
    camera_path = camera_dir / f'{item_id}.json'
    with camera_path.open('w') as f:
        json.dump(camera.to_json(), f, indent=2)

print(f'Saved cameras to {camera_dir}')

# Save test cameras

test_camera_dir = output_dir / 'camera-paths'

for test_path_name, test_cameras in camera_paths.items():
    out_dir = test_camera_dir / test_path_name
    out_dir.mkdir(exist_ok=True, parents=True)
    for i, camera in enumerate(test_cameras):
        camera_path = out_dir / f'{i:06d}.json'
        with camera_path.open('w') as f:
            json.dump(camera.to_json(), f, indent=2)

print(f'Saved camera paths to {test_camera_dir}')

# Copy images to nerfies project

output_image_dir = output_dir / 'rgb/1x'
output_image_dir.mkdir(exist_ok=True, parents=True)

for image_id in scene_manager.image_ids:
    img_path = scene_manager.path_to_image(image_id)
    output_image_path = output_image_dir / img_path.name
    if not output_image_path.exists():
        shutil.copy(img_path, output_image_dir)
        print(f"Saved {output_image_path}")

print(f'Saved images to {output_image_dir}')

# Visualize scene (optional)


def scatter_points(points, size=2):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=size),
    )

display_scene = False
if display_scene:
    import jax
    from jax import numpy as jnp
    camera = scene_manager.camera_list[0]
    near_points = camera.pixels_to_points(camera.get_pixel_centers()[::8, ::8], jnp.array(near)).reshape((-1, 3))
    far_points = camera.pixels_to_points(camera.get_pixel_centers()[::8, ::8], jnp.array(far)).reshape((-1, 3))

    # broken: need positions and origins from generate_orbit_camera_paths
    traces = [
        scatter_points(scene_manager.points),
        scatter_points(scene_manager.camera_positions),
        scatter_points(scene_transform.bounding_box),
        scatter_points(near_points),
        scatter_points(far_points),
        scatter_points(positions),
        scatter_points(origins),
    ]

    fig = go.Figure(traces)
    fig.update_layout(scene_dragmode='orbit')
    fig.show()
