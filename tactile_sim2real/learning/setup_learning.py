import os
import argparse

from tactile_learning.utils.utils_learning import save_json_obj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['edge_2d', 'surface_3d', 'spherical_probe'].",
        default=['surface_3d']
    )
    parser.add_argument(
        '-i', '--input_dir',
        nargs='+',
        help="Choose input directory from ['tactip_331', 'sim_tactip'].",
        default=['tactip_331']
    )
    parser.add_argument(
        '-o', '--target_dir',
        nargs='+',
        help="Choose target directory from ['tactip_331', 'sim_tactip'].",
        default=['sim_tactip']
    )
    parser.add_argument(
        '-c', '--collection_modes',
        nargs='+',
        help="Choose task from ['tap', 'shear'].",
        default=['tap']
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose model from ['pix2pix'].",
        default=['pix2pix']
    )
    parser.add_argument(
        '-r', '--image_dim',
        help="Choose input directory from ['64', '128', '128'].",
        default=128,
        type=int
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default='cuda'
    )

    # parse arguments
    args = parser.parse_args()
    return args


def setup_learning(image_dim, save_dir=None):

    # Parameters
    learning_params = {
        'seed': 42,
        'batch_size': 32,
        'epochs': 100,
        'n_val_batches': 10,
        'lr': 2e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'adam_decay': 0.0,
        'adam_b1': 0.5,
        'adam_b2': 0.999,
        'shuffle': True,
        'n_cpu': 8,
        'sample_interval': 5,
        'lambda_gan': 1.0,
        'lambda_pixel': 100.0,
        'n_save_images': 16,
        'save_every': 5,
    }

    image_processing_params = {
        'dims': (image_dim, image_dim),
        'bbox': None,
        'thresh': None,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift': None,  # (0.025, 0.025),
        'rzoom': None,
        'brightlims': None,
        'noise_var': None,
    }

    if save_dir:
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
            'normalise_disc': [False, True, True, True],
        }

    elif model_type == 'pix2pix_128':

        model_params['generator_kwargs'] = {
            'in_channels': 1,
            'out_channels': 1,
            'unet_down': [64, 128, 256, 512, 512, 512, 512],
            'dropout_down': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            'normalise_down': [False, True, True, True, True, True, False],
            'unet_up': [0, 512, 512, 512, 256, 128, 64],
            'dropout_up': [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        }
        model_params['discriminator_kwargs'] = {
            'in_channels': 1,
            'disc_block': [64, 128, 256, 512],
            'normalise_disc': [False, True, True, True],
        }

    elif model_type == 'pix2pix_256':

        model_params['generator_kwargs'] = {
            'in_channels': 1,
            'out_channels': 1,
            'unet_down': [64, 128, 256, 512, 512, 512, 512, 512],
            'dropout_down': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            'normalise_down': [False, True, True, True, True, True, True, False],
            'unet_up': [0, 512, 512, 512, 512, 256, 128, 64],
            'dropout_up': [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        }
        model_params['discriminator_kwargs'] = {
            'in_channels': 1,
            'disc_block': [64, 128, 256, 512],
            'normalise_disc': [False, True, True, True],
        }
    # save parameters
    save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params
