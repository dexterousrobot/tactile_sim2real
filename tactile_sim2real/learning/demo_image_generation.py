"""
python demo_image_generation.py -i sim_tactip -o cr_tactip -t edge_2d -iv tap -tv data
"""
import os
import itertools as it

from tactile_data.tactile_servo_control import BASE_DATA_PATH as INPUT_DATA_PATH
from tactile_data.tactile_sim2real import BASE_DATA_PATH as TARGET_DATA_PATH
from tactile_learning.pix2pix.image_generator import demo_image_generation

from tactile_sim2real.learning.setup_training import setup_learning
from tactile_sim2real.utils.parse_args import parse_args


if __name__ == '__main__':

    args = parse_args(
        inputs=['ur_tactip'],
        targets=['sim_ur_tactip'],
        tasks=['edge_2d'],
        data_dirs=['train_shear', 'val_shear']
    )

    learning_params, preproc_params = setup_learning()

    # combine the data directories
    input_data_dirs = [
        os.path.join(INPUT_DATA_PATH, *i) for i in it.product(args.inputs, args.tasks, args.data_dirs)
    ]
    target_data_dirs = [
        os.path.join(TARGET_DATA_PATH, *i) for i in it.product(args.targets, args.tasks, args.data_dirs)
    ]

    demo_image_generation(
        input_data_dirs,
        target_data_dirs,
        learning_params,
        preproc_params['image_processing'],
        preproc_params['augmentation']
    )
