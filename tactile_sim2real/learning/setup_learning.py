import os

from tactile_learning.utils.utils_learning import save_json_obj


def setup_learning(save_dir):

    # Parameters
    learning_params = {
        'seed': 42,
        'batch_size': 64,
        'epochs': 100,
        'lr': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'adam_decay': 1e-6,
        'shuffle': True,
        'n_cpu': 8,
    }

    image_processing_params = {
        'dims': (64, 64),
        'bbox': None,
        'thresh': None,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift': (0.025, 0.025),
        'rzoom': None,
        'brightlims': None,
        'noise_var': None,
    }

    save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))
    save_json_obj(image_processing_params, os.path.join(save_dir, 'image_processing_params'))
    save_json_obj(augmentation_params, os.path.join(save_dir, 'augmentation_params'))

    return learning_params, image_processing_params, augmentation_params


def setup_model(model_type, save_dir):

    model_params = {
        'model_type': model_type
    }

    if model_type == 'pix2pix_64':

        model_params['generator_kwargs'] = {
            'in_channels': 1,
            'out_channels': 1,
            'unet_down': [64, 128, 256, 512, 512, 512],
            'dropout_down': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
            'normalise_down': [False, True, True, True, True, False],
            'unet_up': [0, 512, 512, 256, 128, 64],
            'dropout_up': [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        }
        model_params['discriminator_kwargs'] = {
            'in_channels': 1,
            'disc_block': [64, 128, 256, 512],
        }

    # save parameters
    save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params
