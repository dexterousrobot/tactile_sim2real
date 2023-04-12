"""
python launch_collect_data.py -i cr_tactip -r sim -s tactip -t edge_5d
"""
import os
import itertools as it
import pandas as pd

from tactile_data.tactile_sim2real import BASE_DATA_PATH
from tactile_data.utils_data import make_dir
from tactile_servo_control.collect_data.launch_collect_data import collect_data
from tactile_servo_control.collect_data.setup_collect_data import setup_collect_data
from tactile_servo_control.collect_data.utils_collect_data import setup_target_df
from tactile_servo_control.utils.setup_embodiment import setup_embodiment

from tactile_sim2real.utils.parse_args import parse_args


def launch():

    args = parse_args(
        input_dirs=[''],
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        version=['']
    )

    data_params = {
        'train': 10,
        'val': 10
    }

    for task, input_dir, version in it.product(args.tasks, args.input_dirs, args.version):
        for data_dir_name, num_samples in data_params.items():

            data_dir_name = '_'.join(filter(None, [data_dir_name, version]))
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
            if input_dir:
                load_dir = os.path.join(BASE_DATA_PATH, input_dir, task, data_dir_name)
                target_df = pd.read_csv(os.path.join(load_dir, 'targets.csv'))
                target_df.to_csv(os.path.join(save_dir, "targets.csv"), index=False)

            else:
                target_df = setup_target_df(
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


if __name__ == "__main__":
    launch()
