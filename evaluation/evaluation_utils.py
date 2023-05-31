import pathlib
import sys
import inspect
from typeguard import check_type
import h5py

root_path = str(pathlib.Path(__file__).parent.parent)

def extend_path():
    sys.path.append(root_path)
    sys.path.append(str(pathlib.Path(__file__).parent.parent / 'cycleGAN'))

def get_path(path):
    """
    Converts a relative path to an absolute path relative to the root directory of this project.
    """
    return pathlib.Path(root_path) / path

def verify_config(config):
    assert len(config.__class__.__bases__) == 1
    base_class = config.__class__.__bases__[0]

    for attr, attr_type in inspect.get_annotations(base_class).items():
        check_type(getattr(config, attr), attr_type)

    config_attrs = set([x for x in vars(config.__class__).keys() if not x.startswith('__')])
    attrs = set(inspect.get_annotations(base_class).keys())

    if len(config_attrs - attrs) != 0:
        raise ValueError(f'Attributes {config_attrs - attrs} should not be defined for {base_class.__name__}')

def save_images(file_name, images, datasets):
    """
    Saves the list of images into the given file. Each image is saved to a dataset whose name is determined by the
    datasets list. The image at index i is saved to the dataset name at index i.
    """
    assert len(images) == len(datasets)

    output_path = get_path(file_name)
    if output_path.is_file():
        with h5py.File(output_path, 'a') as f:
            for dataset, output in zip(datasets, images):
                if dataset in f:
                    raise ValueError(f'The dataset {dataset} already exists in the output file')
                f.create_dataset(dataset, data=output)
    else:
        with h5py.File(output_path, 'w-') as f:
            for dataset, output in zip(datasets, images):
                f.create_dataset(dataset, data=output)
