import os
import argparse
import itertools
from tactile_learning.pix2pix.image_generator import demo_image_generation

from tactile_sim2real.learning.utils_learning import make_save_dir_str
from tactile_sim2real import BASE_DATA_PATH


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['edge_2d', 'surface_3d', 'spherical_probe'].",
        default=['surface_3d']
    )
    parser.add_argument(
        '-i', '--input_dir',
        nargs=1,
        help="Choose input directory from ['tactip_331', 'sim_tactip'].",
        default=['tactip_331']
    )
    parser.add_argument(
        '-o', '--target_dir',
        nargs=1,
        help="Choose target directory from ['tactip_331', 'sim_tactip'].",
        default=['sim_tactip']
    )
    parser.add_argument(
        '-c', '--collection_modes',
        nargs='+',
        help="Choose task from ['tap', 'shear'].",
        default=['tap']
    )

    args = parser.parse_args()
    tasks = args.tasks
    input_dir = args.input_dir
    target_dir = args.target_dir
    collection_modes = args.collection_modes

    save_dir_str = make_save_dir_str(tasks, input_dir, target_dir, collection_modes)

    learning_params = {
        'batch_size':  8,
        'shuffle': True,
        'n_cpu': 1,
    }

    image_processing_params = {
        'dims': (128, 128),
        'bbox': None,
        'thresh': None,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift':  None,  # (0.015, 0.015),
        'rzoom':   None,
        'brightlims': None,
        'noise_var': None,
        "joint_aug": True,
    }

    # combine the data directories
    input_combined_dirs = list(itertools.product(tasks, input_dir, collection_modes))
    input_combined_paths = [os.path.join(*i) for i in input_combined_dirs]

    target_combined_dirs = list(itertools.product(tasks, target_dir, collection_modes))
    target_combined_paths = [os.path.join(*i) for i in target_combined_dirs]

    input_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "train")
        for data_path in input_combined_paths
    ]

    target_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "train")
        for data_path in target_combined_paths
    ]

    demo_image_generation(
        input_data_dirs,
        target_data_dirs,
        learning_params,
        image_processing_params,
        augmentation_params
    )
