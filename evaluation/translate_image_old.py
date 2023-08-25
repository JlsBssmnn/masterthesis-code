'''
This is an earlier version of the translate_image.py script. It has worse perfromance, but it works on any input shape
with any stride. The current translate_image.py version only work on images whoose shape is a multiple of the patch
size.
'''
import numpy as np
import torch
from evaluation.config.template import TranslateImageConfig
from image_synthesis.logging_config import logging
import h5py
import numpy as np

from cycleGAN.models.networks_3d import define_G
from cycleGAN.util.my_utils import object_to_dict
from image_synthesis.logging_config import logging

from .evaluation_utils import get_path, save_images

def compute_patch_locations(input_shape, patch_size, stride):
    patches_per_axis = (input_shape[-3:] - patch_size) / stride
    covers_axis = patches_per_axis % 1 == 0
    patches_per_axis = np.floor(patches_per_axis).astype(int) + 1

    patch_locations = [list(range(0, patches_per_axis[i]*stride[i], stride[i])) for i in range(3)]
    patch_locations = [patch_locations[i] if covers
                       else patch_locations[i] + [input_shape[i - 3] - patch_size[i]]
                       for i, covers in enumerate(covers_axis)]

    z, y, x = np.meshgrid(*patch_locations, indexing='ij')
    z, y, x = z.flatten(), y.flatten(), x.flatten()
    return np.stack((z, y, x), 0).T

@torch.no_grad()
def apply_generator(input, generator, config: TranslateImageConfig):
    """
    This function applies the provided generator to the provided input. This is done by feeding small patches of the
    input into the generator. The output for these patches is then aggregated via averaging to construct the output for
    the entire input.
    """
    device = torch.device('cuda:0') if config.use_gpu else torch.device('cpu')

    patch_size = np.array(config.patch_size)
    stride = np.array(config.stride)
    out_shape = (config.generator_config.output_nc,) + input.shape[-3:]
    batch_size = config.batch_size
    assert (input.shape[-3:] >= patch_size).all(), 'Input must be at least as big as one patch'

    if config.mask is None:
        mask = torch.ones(patch_size)
    else:
        mask = torch.from_numpy(config.mask)

    input_batch = torch.empty((batch_size, input.shape[0]) + tuple(patch_size), device=device)

    n_output_layers = int((np.ceil(patch_size / stride) + 1).prod()) # This only works for stride = patch_size / 2
    outputs = torch.full((n_output_layers,) + out_shape, np.nan)
    patch_locations = compute_patch_locations(input.shape, patch_size, stride)

    i = 0
    slices = np.empty(batch_size, dtype=object)

    def insert_gen_output(output):
        nonlocal outputs
        output = output.detach().cpu()
        for out, s in zip(output, slices):
            free_channels = torch.all(torch.isnan(outputs[(slice(None),) + s]).view(n_output_layers, -1), dim=1)
            free_channels = torch.nonzero(free_channels, as_tuple=True)[0]
            outputs[free_channels[0]][s] = out

    for z, y, x in patch_locations:
        cords = (z, y, x)
        s = tuple([slice(c, c + patch_size[i]) for i, c in enumerate(cords)])
        s = (slice(None),) + s
        if config.scale_with_patch_max:
            gen_input = (input[s] / input[s].max() - 0.5) * 2
        else:
            gen_input = input[s]

        input_batch[i] = gen_input
        slices[i] = s

        i += 1

        if i < batch_size:
            continue

        i = 0

        gen_output = generator(input_batch)
        gen_output[:, ~mask] = float('nan')
        insert_gen_output(gen_output)
    else:
        if i > 0:
            gen_output = generator(input_batch[:i])
            gen_output[:, ~mask] = float('nan')
            insert_gen_output(gen_output)

    outputs = torch.nanmean(outputs, axis=0)
    outputs[torch.isnan(outputs)] = config.masked_value
    return outputs.detach().cpu().numpy()


def translate_image(config: TranslateImageConfig):
    """
    Performs the image translation step.
    """
    config.generator_config.gpu_ids = [] # otherwise state_dict cannot be loaded
    images = []
    s = config.slices
    if s is None:
        s = [':']
    with h5py.File(get_path(config.input_file)) as f:
        image = f[config.input_dataset]
        for slice_string in s:
            images.append(np.asarray(eval(f'image[{slice_string}]')))

    if config.skip_translation:
        outputs = []
        for image in images:
            if image.dtype == np.uint8:
                image = image / 255
            elif image.dtype == np.uint16:
                image = image / image.max()
            else:
                raise NotImplementedError(f"The datatype {image.dtype} is not implemented")
            outputs.append(image)
        return outputs

    generator = define_G(**object_to_dict(config.generator_config))
    sucess = generator.load_state_dict(torch.load(get_path(config.generator_save)))
    logging.info(sucess)
    if config.use_gpu:
        generator.to(0)

    outputs = []

    for image in images:
        if not config.scale_with_patch_max:
            if image.dtype == np.uint8:
                image = (image / 127.5) - 1
            elif image.dtype == np.uint16:
                image = ((image / image.max()) - 0.5) * 2
            else:
                raise NotImplementedError(f"The datatype {image.dtype} is not implemented")
        image = torch.tensor(image.astype(np.float32), dtype=torch.float32)

        output = apply_generator(image, generator, config)
        output = ((output + 1) / 2) # normalize to range(0, 1)
        outputs.append(output)
    return outputs


def one_step(config: TranslateImageConfig):
    """
    This function just performs the image translation and saves the result to a file.
    """
    assert config.output_file is not None
    assert config.output_datasets is not None
    assert (config.slices is None and len(config.output_datasets) == 1) or \
            (config.slices is not None and len(config.slices) == len(config.output_datasets))

    outputs = translate_image(config)
    save_images(config.output_file, outputs, config.output_datasets)
