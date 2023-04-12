"""
python demo_image_generation.py -i sim_tactip -o ur_tactip -t edge_2d -v tap
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
        version=['']
    )

    learning_params, preproc_params = setup_learning()

    # combine the data directories
    input_paths = [os.path.join(*i) for i in it.product(args.input_dirs, args.tasks)]
    target_paths = [os.path.join(*i) for i in it.product(args.target_dirs, args.tasks)]

    train_dir_names = ['_'.join(filter(None, ["train", i])) for i in args.version]
    val_dir_names = ['_'.join(filter(None, ["val", i])) for i in args.version]

    input_data_dirs = [
        *[os.path.join(BASE_DATA_PATH, path, train_dir_name) for path in input_paths for train_dir_name in train_dir_names],
        *[os.path.join(BASE_DATA_PATH, path, val_dir_name) for path in input_paths for val_dir_name in val_dir_names]
    ]

    target_data_dirs = [
        *[os.path.join(BASE_DATA_PATH, path, train_dir_name) for path in target_paths for train_dir_name in train_dir_names],
        *[os.path.join(BASE_DATA_PATH, path, val_dir_name) for path in target_paths for val_dir_name in val_dir_names]
    ]

    demo_image_generation(
        input_data_dirs,
        target_data_dirs,
        learning_params,
        preproc_params['image_processing'],
        preproc_params['augmentation']
    )
