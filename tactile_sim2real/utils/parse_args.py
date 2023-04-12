import argparse


def parse_args(
    input_dirs=[],
    target_dirs=['sim_tactip'],
    robot='sim',
    sensor='tactip',
    tasks=['edge_2d'],
    models=['pix2pix_128'],
    version=[],
    device='cuda'
):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input_dirs',
        nargs='+',
        help="Choose input directory from ['ur_tactip', 'sim_tactip'].",
        default=input_dirs
    )
    parser.add_argument(
        '-o', '--target_dirs',
        nargs='+',
        help="Choose target directory from ['ur_tactip', 'sim_tactip'].",
        default=target_dirs
    )
    parser.add_argument(
        '-r', '--robot',
        type=str,
        help="Choose robot from ['sim', 'mg400', 'cr']",
        default=robot
    )
    parser.add_argument(
        '-s', '--sensor',
        type=str,
        help="Choose sensor from ['tactip', 'tactip_127']",
        default=sensor
    )
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose tasks from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d']",
        default=tasks
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose model from ['pix2pix'].",
        default=models
    )
    parser.add_argument(
        '-v', '--version',
        type=str,
        help="Choose version from ['tap', 'shear].",
        default=version
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda']",
        default=device
    )

    return parser.parse_args()
