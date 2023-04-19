"""
python launch_training.py -i ur_tactip -o sim_tactip -t edge_2d -m pix2pix_128 -v tap
"""
import os
import itertools as it

from tactile_data.tactile_sim2real import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_learning.pix2pix.image_generator import Pix2PixImageGenerator
from tactile_learning.pix2pix.models import create_model
from tactile_learning.pix2pix.train_pix2pix import train_pix2pix
from tactile_learning.utils.utils_learning import seed_everything, make_dir

from tactile_sim2real.learning.setup_training import setup_training
from tactile_sim2real.utils.parse_args import parse_args


def launch(args):

    input_paths = [os.path.join(*i) for i in it.product(args.input_dirs, args.tasks)]
    target_paths = [os.path.join(*i) for i in it.product(args.target_dirs, args.tasks)]

    input_train_dir_name = '_'.join(["train", *args.input_version])
    target_train_dir_name = '_'.join(["train", *args.target_version])
    input_val_dir_name = '_'.join(["val", *args.input_version])
    target_val_dir_name = '_'.join(["val", *args.target_version])

    input_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, path, input_train_dir_name) for path in input_paths
    ]
    target_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, path, target_train_dir_name) for path in target_paths
    ]
    input_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, path, input_val_dir_name) for path in input_paths
    ]
    target_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, path, target_val_dir_name) for path in target_paths
    ]

    for args.model in args.models:

        output_dir = "_to_".join([*args.input_dirs, *args.target_dirs])
        task_dir = "_".join(args.tasks)

        # setup save dir
        save_dir = os.path.join(BASE_MODEL_PATH, output_dir, task_dir, args.model)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, preproc_params = setup_training(
            args.model, 
            input_train_data_dirs, 
            save_dir
        )

        # Configure dataloaders
        train_generator = Pix2PixImageGenerator(
            input_data_dirs=input_train_data_dirs,
            target_data_dirs=target_train_data_dirs,
            **{**preproc_params['image_processing'], **preproc_params['augmentation']}
        )
        val_generator = Pix2PixImageGenerator(
            input_data_dirs=input_val_data_dirs,
            target_data_dirs=target_val_data_dirs,
            **preproc_params['image_processing']
        )

        # create the model
        seed_everything(learning_params['seed'])
        generator, discriminator = create_model(
            preproc_params['image_processing']['dims'],
            model_params,
            device=args.device
        )

        # run training
        train_pix2pix(
            generator,
            discriminator,
            train_generator,
            val_generator,
            learning_params,
            preproc_params['image_processing'],
            save_dir,
            device=args.device
        )


if __name__ == "__main__":

    args = parse_args(
        tasks=['edge_2d'],
        input_dirs=['cr_tactip'],
        target_dirs=['sim_tactip'],
        models=['pix2pix_128_temp'],
        input_version=['data'],
        target_version=['data_temp']
    )

    launch(args)
