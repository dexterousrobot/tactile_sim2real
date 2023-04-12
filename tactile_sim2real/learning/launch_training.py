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


def launch():

    args = parse_args(
        tasks=['edge_5d'],
        input_dirs=['ur_tactip'],
        target_dirs=['sim_tactip'],
        models=['pix2pix_128'],
        version=['tap']
    )

    # combined data directories
    input_paths = [os.path.join(*i) for i in it.product(args.input_dirs, args.tasks)]
    target_paths = [os.path.join(*i) for i in it.product(args.target_dirs, args.tasks)]

    train_dir_names = ['_'.join(filter(None, ["train", i])) for i in args.version]
    val_dir_names = ['_'.join(filter(None, ["val", i])) for i in args.version]

    input_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, path, train_dir_name) for path in input_paths for train_dir_name in train_dir_names
    ]
    target_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, path, train_dir_name) for path in target_paths for train_dir_name in train_dir_names
    ]
    input_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, path, val_dir_name) for path in input_paths for val_dir_name in val_dir_names
    ]
    target_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, path, val_dir_name) for path in target_paths for val_dir_name in val_dir_names
    ]

    for args.model in args.models:

        output_dir = "_to_".join([*args.input_dirs, *args.target_dirs])
        task_dir = "_".join(args.tasks)
        model_dir_name = "_".join([args.model, *args.version])

        # setup save dir
        save_dir = os.path.join(BASE_MODEL_PATH, output_dir, task_dir, model_dir_name)
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
    launch()
