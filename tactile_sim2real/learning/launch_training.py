"""
python launch_training.py -i ur_tactip -o sim_tactip -t edge_2d -m pix2pix_128 -v tap
"""
import os
import itertools

from tactile_data.tactile_sim2real import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_learning.pix2pix.image_generator import Pix2PixImageGenerator
from tactile_learning.pix2pix.models import create_model
from tactile_learning.pix2pix.train_pix2pix import train_pix2pix
from tactile_learning.utils.utils_learning import seed_everything, make_dir
from tactile_sim2real.learning.setup_learning import setup_parse_args, setup_model, setup_learning


def launch():

    tasks, input_dir, target_dir, version, models, device = setup_parse_args(
        tasks=['edge_2d'],
        input_dir=['ur_tactip'],
        target_dir=['sim_tactip'],
        version='tap',
        models=['pix2pix_128'],
    )

    # combined data directories
    input_combined_dirs = list(itertools.product(tasks, input_dir))
    target_combined_dirs = list(itertools.product(tasks, target_dir))

    input_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "_".join(["train", version])) 
                for data_path in [os.path.join(*i) for i in input_combined_dirs]
    ]
    target_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "_".join(["train", version]))
                for data_path in [os.path.join(*i) for i in target_combined_dirs]
    ]
    input_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "_".join(["val", version]))
                for data_path in [os.path.join(*i) for i in input_combined_dirs]
    ]
    target_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "_".join(["val", version]))
                for data_path in [os.path.join(*i) for i in target_combined_dirs]
    ]

    for model_str in models:

        # setup save dir
        save_dir = os.path.join(
            BASE_MODEL_PATH, 
            "_to_".join([*input_dir, *target_dir]), 
            "_".join(tasks), 
            "_".join([model_str, version])
        )
        make_dir(save_dir)

        # setup parameters
        learning_params, preproc_params = setup_learning(save_dir)
        network_params = setup_model(model_str, save_dir)

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
            network_params,
            device=device
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
            device=device
        )


if __name__ == "__main__":
    launch()
