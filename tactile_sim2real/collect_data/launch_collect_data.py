"""
python launch_collect_data.py -i cr_tactip -r sim_cr -s tactip -t edge_5d
"""
import os
import itertools as it
import pandas as pd

from tactile_data.tactile_servo_control import BASE_DATA_PATH as INPUT_DATA_PATH
from tactile_data.tactile_sim2real import BASE_DATA_PATH as TARGET_DATA_PATH
from tactile_data.collect_data.collect_data import collect_data
from tactile_data.collect_data.process_image_data import process_image_data, partition_data
from tactile_data.utils import make_dir
from tactile_servo_control.collect_data.setup_collect_data import setup_collect_data
from tactile_servo_control.utils.setup_embodiment import setup_embodiment

from tactile_sim2real.utils.parse_args import parse_args


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task, args.input in it.product(args.tasks, args.inputs):
        for args.data_dir in args.data_dirs:

            # setup save dir
            save_dir = os.path.join(TARGET_DATA_PATH, output_dir, args.task, args.data_dir)
            image_dir = os.path.join(save_dir, "sensor_images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            collect_params, env_params, sensor_params = setup_collect_data(
                args.robot,
                args.sensor,
                args.task,
                save_dir
            )

            # setup embodiment
            robot, sensor = setup_embodiment(
                env_params,
                sensor_params
            )

            # load targets to collect (select one of available)
            load_dir = os.path.join(INPUT_DATA_PATH, args.input, args.task, args.data_dir)
            target_df = pd.read_csv(os.path.join(load_dir, 'targets_images.csv'))
            target_df.to_csv(os.path.join(save_dir, "targets.csv"), index=False)

            # collect
            collect_data(
                robot,
                sensor,
                target_df,
                image_dir,
                collect_params
            )


def process_images(args, image_params, split=None):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task in args.tasks:
        path = os.path.join(TARGET_DATA_PATH, output_dir, args.task)

        dir_names = partition_data(path, args.data_dirs, split)
        process_image_data(path, dir_names, image_params)


if __name__ == "__main__":

    args = parse_args(
        inputs=['cr_tactip'],
        robot='sim_cr',
        sensor='tactip',
        tasks=['surface_3d'],
        data_dirs=['train_data', 'val_data']
    )

    image_params = {
        "bbox": (12, 12, 240, 240)  # sim (12, 12, 240, 240)
    }

    launch(args)
    process_images(args, image_params)#, split=0.8)
