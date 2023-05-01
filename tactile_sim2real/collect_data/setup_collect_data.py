import os
import numpy as np

from tactile_data.utils import save_json_obj


def setup_sensor_image_params(robot, sensor, save_dir=None):

    bbox_dict = {
        'mini': (320-160,    240-160+25, 320+160,    240+160+25),
        'midi': (320-220+10, 240-220-20, 320+220+10, 240+220-20)
    }
    sensor_type = 'midi'  # TODO: Fix hardcoded sensor type

    if 'sim' in robot:
        sensor_image_params = {
            "type": "standard_tactip",
            "image_size": (256, 256),
            "show_tactile": True
        }

    else:
        sensor_image_params = {
            'type': sensor_type,
            'source': 0,
            'exposure': -7,
            'gray': True,
            'bbox': bbox_dict[sensor_type]
        }

    if save_dir:
        save_json_obj(sensor_image_params, os.path.join(save_dir, 'sensor_image_params'))

    return sensor_image_params


def setup_collect_params(robot, task, save_dir=None):

    if robot.split('_')[0] == 'sim':
        robot = 'sim'

    pose_lims_dict = {
        'surface_3d': [(0, 0, 2.5, -15, -15, 0), (0, 0, 5.5, 15, 15, 0)],
        'edge_2d':    [(0, -6, 3, 0, 0, -180),   (0, 6, 5, 0, 0, 180)],
        'spheres_2d': [(-12.5, -12.5, 4, 0, 0, 0), (12.5, 12.5, 5, 0, 0, 0)],
        'mixed_2d':   [(-5, -5, 4, 0, 0, 0),       (5, 5, 5, 0, 0, 0)],
    }

    shear_lims_dict = {
        'surface_3d': [(-5, -5, 0, -5, -5, -5), (5, 5, 0, 5, 5, 5)],
        # 'surface_3d': [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)], # tap data
        'edge_2d':    [(-5, -5, 0, -5, -5, -5), (5, 5, 0, 5, 5, 5)],
        # 'edge_2d':   [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)], # tap data
        'spheres_2d': [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)],
        'mixed_2d':   [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)],
    }
    
    sphere_names = [
        ['2mm', '3mm', '4mm'],  ['5mm', '6mm', '7mm'], ['8mm', '9mm', '10mm'],
    ]

    mixed_names = [
        ['cone', 'cross_lines', 'curved_surface', 'cylinder', 'cylinder_shell', 'cylinder_side', 'dot_in'], 
        ['dots', 'flat_slab', 'hexagon', 'line', 'moon', 'pacman', 'parallel_lines'],
        ['prism', 'random', 'sphere', 'sphere2', 'torus', 'triangle', 'wave1']
    ]

    object_poses_dict = {
        "surface_3d": {'surface': (0, 0, 0, 0, 0, 0)},
        "edge_2d":    {'edge':    (0, 0, 0, 0, 0, 0)},
        "spheres_2d": {
            sphere_names[i][j]: (60*(1-j), 60*(1-i), 0, 0, 0, -48) 
            for i, j in np.ndindex(3, 3)
        },
        "mixed_2d":  {
            mixed_names[i][j]: (25*(i-1), 25*(3-j), 0, 0, 0, 0) 
            for i, j in np.ndindex(3, 7)
        }
    }

    collect_params = {
        'pose_llims': pose_lims_dict[task][0],
        'pose_ulims': pose_lims_dict[task][1],
        'shear_llims': shear_lims_dict[task][0],
        'shear_ulims': shear_lims_dict[task][1],
        'object_poses': object_poses_dict[task],
        'sample_disk': False,
        'sort': False,
        'seed': 0
    }

    if robot == 'sim':
        collect_params['sort'] = True

    if save_dir:
        save_json_obj(collect_params, os.path.join(save_dir, 'collect_params'))

    return collect_params


def setup_env_params(robot, task, save_dir=None):

    if robot.split('_')[0] == 'sim':
        robot = 'sim'

    # pick the correct stimuli dependent on the task
    if 'surface' in task:
        stim_name = 'square'
        stim_pose = (650, 0, 12.5, 0, 0, 0) 
        workframe_dict = {
            'sim': (650, 0,  50, -180, 0, 90),
            'ur':  (0, -500, 54, -180, 0, 0)
        }  
        tcp_pose_dict = {
            'sim': (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 101, 0, 0, 0)
        }  
    if 'edge' in task:
        stim_name = 'square'
        stim_pose = (600, 0, 12.5, 0, 0, 0) 
        workframe_dict = {
            'sim': (650, 0,  50, -180, 0, 90),
            'ur':  (0, -451, 54, -180, 0, 0)
        } 
        tcp_pose_dict = {
            'sim': (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 101, 0, 0, 0)
        }  
    if 'spheres' in task:
        stim_name = 'spherical_probes'
        stim_pose = (650, 0, 0, 0, 0, 0)
        workframe_dict = {
            'sim': (650, 0, 42.5, -180, 0, 90),
            'ur': (-15.75, -462, 47.0, -180, 0, 0)
        }
        tcp_pose_dict = {
            'sim': (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 88.5, 0, 0, 0)
        }  
    if 'mixed' in task:
        stim_name = 'mixed_probes'
        stim_pose = (650, 0, 0, 0, 0, 0)
        workframe_dict = {
            'sim': (650, 0, 20, -180, 0, 0)
        }
        tcp_pose_dict = {
            'sim':   (0, 0, -85, 0, 0, 0),
        }  

    env_params = {
        'robot': robot,
        'stim_name': stim_name,
        'speed': 50,
        'work_frame': workframe_dict[robot],
        'tcp_pose': tcp_pose_dict[robot],
    }

    if 'sim' in robot:
        env_params['speed'] = float('inf')
        env_params['stim_pose'] = stim_pose

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def setup_collect_data(robot, sensor, task, save_dir=None):
    collect_params = setup_collect_params(robot, task, save_dir)
    sensor_image_params = setup_sensor_image_params(robot, sensor, save_dir)
    env_params = setup_env_params(robot, task, save_dir)

    return collect_params, env_params, sensor_image_params


if __name__ == '__main__':
    pass
