"""
python demo_image_generation.py -i sim_tactip -o cr_tactip -t edge_2d -iv tap -tv data
"""
import os
import itertools as it

from tactile_data.tactile_sim2real import BASE_DATA_PATH
from tactile_learning.pix2pix.image_generator import demo_image_generation

from tactile_sim2real.learning.setup_training import setup_learning
from tactile_sim2real.utils.parse_args import parse_args


if __name__ == '__main__':

    args = parse_args(
        tasks=['edge_2d'],
        input_dirs=['ur_tactip'],
        target_dirs=['sim_tactip'],
        input_version=['data'],
        target_version=['data_temp']
    )

    learning_params, preproc_params = setup_learning()

    # combine the data directories
    input_paths = [os.path.join(*i) for i in it.product(args.input_dirs, args.tasks)]
    target_paths = [os.path.join(*i) for i in it.product(args.target_dirs, args.tasks)]

    input_train_dir_name = '_'.join(["train", *args.input_version])
    target_train_dir_name = '_'.join(["train", *args.target_version])
    input_val_dir_name = '_'.join(["val", *args.input_version])
    target_val_dir_name = '_'.join(["val", *args.target_version])

    input_data_dirs = [
        *[os.path.join(BASE_DATA_PATH, path, input_train_dir_name) for path in input_paths],
        *[os.path.join(BASE_DATA_PATH, path, input_val_dir_name) for path in input_paths]
    ]

    target_data_dirs = [
        *[os.path.join(BASE_DATA_PATH, path, target_train_dir_name) for path in target_paths],
        *[os.path.join(BASE_DATA_PATH, path, target_val_dir_name) for path in target_paths]
    ]

    demo_image_generation(
        input_data_dirs,
        target_data_dirs,
        learning_params,
        preproc_params['image_processing'],
        preproc_params['augmentation']
    )
