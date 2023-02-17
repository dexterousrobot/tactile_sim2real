import os
import itertools

from tactile_sim2real.learning.setup_learning import parse_args
from tactile_sim2real.learning.setup_learning import setup_model
from tactile_sim2real.learning.setup_learning import setup_learning

from tactile_learning.pix2pix.models import create_model
from tactile_learning.pix2pix.train_pix2pix import train_pix2pix
from tactile_learning.utils.utils_learning import seed_everything, make_dir
from tactile_learning.pix2pix.image_generator import Pix2PixImageGenerator

from tactile_sim2real.learning.utils_learning import make_save_dir_str

from tactile_sim2real import BASE_DATA_PATH
from tactile_sim2real import BASE_MODEL_PATH


def launch():

    args = parse_args()
    tasks = args.tasks
    input_dir = args.input_dir
    target_dir = args.target_dir
    collection_modes = args.collection_modes
    models = args.models
    device = args.device
    image_dim = args.image_dim

    # combine the data directories
    input_combined_dirs = list(itertools.product(tasks, input_dir, collection_modes))
    input_combined_paths = [os.path.join(*i) for i in input_combined_dirs]

    target_combined_dirs = list(itertools.product(tasks, target_dir, collection_modes))
    target_combined_paths = [os.path.join(*i) for i in target_combined_dirs]

    input_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "train")
        for data_path in input_combined_paths
    ]
    target_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "train")
        for data_path in target_combined_paths
    ]
    input_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "val")
        for data_path in input_combined_paths
    ]
    target_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, data_path, "val")
        for data_path in target_combined_paths
    ]

    save_dir_str = make_save_dir_str(tasks, input_dir, target_dir, collection_modes)

    for model_type in models:

        model_name = "_".join([model_type, str(image_dim)])

        # setup save dir
        save_dir = os.path.join(
            BASE_MODEL_PATH,
            save_dir_str,
            model_name
        )
        make_dir(save_dir)

        # setup parameters
        learning_params, image_processing_params, augmentation_params = setup_learning(image_dim, save_dir)
        network_params = setup_model(model_name, save_dir)

        # Configure dataloaders
        train_generator = Pix2PixImageGenerator(
            input_data_dirs=input_train_data_dirs,
            target_data_dirs=target_train_data_dirs,
            **{**image_processing_params, **augmentation_params}
        )

        val_generator = Pix2PixImageGenerator(
            input_data_dirs=input_val_data_dirs,
            target_data_dirs=target_val_data_dirs,
            **image_processing_params
        )

        # create the model
        seed_everything(learning_params['seed'])
        generator, discriminator = create_model(
            image_processing_params['dims'],
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
            image_processing_params,
            save_dir,
            device=device
        )


if __name__ == "__main__":
    launch()
