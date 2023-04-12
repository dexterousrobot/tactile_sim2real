"""
python process_collect_data.py -r cr -s tactip_331 -t edge_5d
"""
import os

from tactile_data.tactile_sim2real import BASE_DATA_PATH
from tactile_servo_control.collect_data.process_collect_data import split_data, process_data

from tactile_sim2real.utils.parse_args import parse_args


def main():

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['edge_2d'],
        version=['test']
    )

    dir_in = "data"
    dirs_out = ["train", "val"]
    frac = 0.8

    process_params = {
        # 'thresh': True,
        'dims': (128, 128),
        # "circle_mask_radius": 220,
        "bbox": (12, 12, 240, 240)  # sim (12, 12, 240, 240) # midi (10, 10, 430, 430) # mini (10, 10, 310, 310)
    }

    for args.task in args.tasks:

        output_dir = '_'.join([args.robot, args.sensor])
        dir_in = '_'.join([dir_in, *args.version])
        dirs_out = ['_'.join([dir, *args.version]) for dir in dirs_out]
        path = os.path.join(BASE_DATA_PATH, output_dir, args.task)

        # split_data(path, dir_in, dirs_out, frac)
        process_data(path, dirs_out, process_params)


if __name__ == "__main__":
    main()
