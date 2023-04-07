"""
python demo_image_generation.py -i sim_tactip -o ur_tactip -t edge_2d -v tap
"""
import os
import itertools

from tactile_data.tactile_sim2real import BASE_DATA_PATH
from tactile_learning.pix2pix.image_generator import demo_image_generation
from tactile_sim2real.learning.setup_learning import setup_parse_args, setup_learning


if __name__ == '__main__':

    tasks, input_dir, target_dir, version, _, _ = setup_parse_args(
        tasks=['edge_2d'],
        input_dir=['ur_tactip'],
        target_dir=['sim_tactip'],
        version='tap'
    )

    learning_params, preproc_params = setup_learning()

    # combine the data directories
    input_combined_dirs = list(itertools.product(tasks, input_dir))
    input_combined_paths = [os.path.join(*i) for i in input_combined_dirs]

    target_combined_dirs = list(itertools.product(tasks, target_dir))
    target_combined_paths = [os.path.join(*i) for i in target_combined_dirs]

    input_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "train"+version) for data_path in input_combined_paths
    ]

    target_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "train"+version) for data_path in target_combined_paths
    ]

    demo_image_generation(
        input_data_dirs,
        target_data_dirs,
        learning_params,
        preproc_params['image_processing'],
        {} # preproc_params['augmentation']
    )
