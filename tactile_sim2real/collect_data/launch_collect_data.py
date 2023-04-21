"""
python launch_collect_data.py -i cr_tactip -r sim -s tactip -t edge_5d
"""
import os
import itertools as it
import pandas as pd

from tactile_data.tactile_sim2real import BASE_DATA_PATH
from tactile_data.collect_data.process_data import process_data, split_data
from tactile_data.collect_data.setup_embodiment import setup_embodiment
from tactile_data.collect_data.setup_targets import setup_targets
from tactile_data.utils import make_dir
from tactile_servo_control.collect_data.launch_collect_data import collect_data
from tactile_servo_control.collect_data.setup_collect_data import setup_collect_data

from tactile_sim2real.utils.parse_args import parse_args


def launch(args, data_params):

    for task, input_dir, version in it.product(args.tasks, args.input_dirs, args.version):
        for data_dir_name, num_samples in data_params.items():

            input_dir_name = '_'.join([data_dir_name, *args.input_version])
            data_dir_name = '_'.join([data_dir_name, *args.data_version])
            output_dir = '_'.join([args.robot, args.sensor])

            # setup save dir
            save_dir = os.path.join(BASE_DATA_PATH, output_dir, task, data_dir_name)
            image_dir = os.path.join(save_dir, "images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            collect_params, env_params, sensor_params = setup_collect_data(
                args.robot,
                args.sensor,
                task,
                save_dir
            )

            # setup embodiment
            robot, sensor = setup_embodiment(
                env_params,
                sensor_params
            )

            # setup targets to collect
            if args.input_dir:
                load_dir = os.path.join(BASE_DATA_PATH, args.input_dir, args.task, input_dir_name)
                target_df = pd.read_csv(os.path.join(load_dir, 'targets.csv'))
                target_df.to_csv(os.path.join(save_dir, "targets.csv"), index=False)

            else:
                target_df = setup_targets(
                    collect_params,
                    num_samples,
                    save_dir
                )

            # collect
            collect_data(
                robot,
                sensor,
                target_df,
                image_dir,
                collect_params
            )


def process(args, data_params, process_params, split=None):

    output_dir = '_'.join([args.robot, args.sensor])
    dir_names = ['_'.join(filter(None, [dir, *args.data_version])) for dir in data_params]

    for args.task in args.tasks:
        path = os.path.join(BASE_DATA_PATH, output_dir, args.task)

        dir_names = split_data(path, dir_names, split)
        process_data(path, dir_names, process_params)


if __name__ == "__main__":

    args = parse_args(
        input_dirs=['cr_tactip'],
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        input_version=['data'],
        data_version=['data_temp']
    )

    data_params = {
        'train': 0,
        'val': 0
    }

    process_params = {
        "bbox": (12, 12, 240, 240)  # sim (12, 12, 240, 240)
    }

    # launch(args, data_params)
    process(args, data_params, process_params)  # , split=0.8)
