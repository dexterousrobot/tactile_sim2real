import os


def make_save_dir_str(tasks, input_dir, target_dir, collection_modes):
    """
    Combines tasks, input/target dirs and collection modes into single path
    """

    return os.path.join(
        "_to_".join([*input_dir, *target_dir]),
        "_".join(tasks),
        "_".join(collection_modes),
    )
