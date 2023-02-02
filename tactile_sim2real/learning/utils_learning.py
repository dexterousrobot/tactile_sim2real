import os


def make_save_dir_str(tasks, input_dir, target_dir, collection_modes):
    """
    Combines tasks, input/target dirs and collection modes into single path
    """

    return os.path.join(
        "_".join(tasks),
        "_to_".join([*input_dir, *target_dir]),
        "_".join(collection_modes),
    )
