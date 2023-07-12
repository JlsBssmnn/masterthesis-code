import numpy as np
import torch
from evaluation.config.template import TranslateImageConfig
from image_synthesis.logging_config import logging
import h5py
import numpy as np
import neuroglancer
import webbrowser

from cycleGAN.models.networks_3d import define_G
from cycleGAN.util.my_utils import object_to_dict
from image_synthesis.logging_config import logging

from .evaluation_utils import get_path, save_images
from utils.neuroglancer_viewer.neuroglancer_viewer import show_image

class GeneratorApplier:
    def __init__(self, input_shape, config: TranslateImageConfig):
        self.config = config
        self.device = torch.device('cuda:0') if config.use_gpu else torch.device('cpu')
        self.patch_locations, self.patches_per_axis = self.compute_patch_locations(input_shape, config.patch_size)

    def compute_patch_locations(self, input_shape, patch_size):
        patch_size = np.array(patch_size)
        assert (patch_size % 2 == 0).all(), "patch size must be multiple of 2 in each dimension"
        patch_locations_per_layer = []
        patches_per_axis = np.empty((4, 3), dtype=int)

        for i in range(4):
            shape = np.array(input_shape[-3:])
            if i > 0:
                shape[i - 1] -= patch_size[i - 1] / 2

            patches_per_axis_i = shape / patch_size
            covers_axis = patches_per_axis_i % 1 == 0
            patches_per_axis_i = np.floor(patches_per_axis_i).astype(int)
            patches_per_axis[i] = patches_per_axis_i

            start_locations = [0, 0, 0]
            end_locations = patches_per_axis_i * patch_size
            if i > 0:
                start_locations[i - 1] = int(patch_size[i - 1] / 2)
                end_locations[i - 1] += patch_size[i - 1] / 2
            patch_locations = [list(range(start_locations[i], end_locations[i], patch_size[i])) for i in range(3)]

            if i == 0:
                patch_locations = [patch_locations[i] if covers
                                   else patch_locations[i] + [input_shape[i - 3] - patch_size[i]]
                                   for i, covers in enumerate(covers_axis)]

            z, y, x = np.meshgrid(*patch_locations, indexing='ij')
            z, y, x = z.flatten(), y.flatten(), x.flatten()
            patch_locations_per_layer.append(np.stack((z, y, x), 0).T)
        return patch_locations_per_layer, patches_per_axis

    @torch.no_grad()
    def apply_generator(self, input, generator):
        """
        This function applies the provided generator to the provided input. This is done by feeding small patches of the
        input into the generator. The output for these patches is then aggregated via averaging to construct the output for
        the entire input.

        When using the gpu and a large image, it is highly advised to use pytorch gpu tensor for input. This will avoid
        a lot of cpu() calls and thus will be faster.
        """
        patch_size = np.array(self.config.patch_size)
        c = self.config.generator_config.output_nc
        out_shape = (c,) + input.shape[-3:]
        batch_size = self.config.batch_size
        assert (input.shape[-3:] >= patch_size).all(), 'Input must be at least as big as one patch'

        input_batch = torch.empty((batch_size, input.shape[0]) + tuple(patch_size), device=self.device)

        n_output_layers = 4
        outputs = torch.full((n_output_layers,) + out_shape, np.nan, device=self.device)
        slices = np.empty(batch_size, dtype=object)

        for i in range(4):
            output_stack = torch.empty((len(self.patch_locations[i]), c, *patch_size), device=self.device)
            stack_pointer = 0
            size_of_current_batch = 0
            for z, y, x in self.patch_locations[i]:
                cords = (z, y, x)
                s = tuple([slice(c, c + patch_size[j]) for j, c in enumerate(cords)])
                s = (slice(None),) + s
                if self.config.scale_with_patch_max:
                    gen_input = (input[s] / input[s].max() - 0.5) * 2
                else:
                    gen_input = input[s]

                input_batch[size_of_current_batch] = gen_input
                slices[size_of_current_batch] = s

                size_of_current_batch += 1

                if size_of_current_batch < batch_size:
                    continue

                gen_output = generator(input_batch)
                output_stack[stack_pointer:stack_pointer + size_of_current_batch] = gen_output

                stack_pointer = size_of_current_batch
                size_of_current_batch = 0
            else:
                if size_of_current_batch > 0:
                    gen_output = generator(input_batch[:size_of_current_batch])
                    output_stack[stack_pointer:stack_pointer + size_of_current_batch] = gen_output

            # The following lines are equivalent to this numpy code (thanks pytorch ;^):
            #     net_outputs = net_outputs.reshape(*self.patches_per_axis[i], c, *patch_size)
            #     net_outputs = np.transpose(net_outputs, (3, 0, 4, 1, 5, 2, 6))
            #                     .reshape(c, *(self.patches_per_axis[i] * patch_size))
            output_stack = output_stack.reshape((*self.patches_per_axis[i], c, *patch_size))
            output_stack = torch.transpose(output_stack, 0, 3) # 3, 1, 2, 0, 4, 5, 6
            output_stack = torch.transpose(output_stack, 1, 3) # 3, 0, 2, 1, 4, 5, 6
            output_stack = torch.transpose(output_stack, 2, 5) # 3, 0, 5, 1, 4, 2, 6
            output_stack = torch.transpose(output_stack, 2, 4) # 3, 0, 4, 1, 5, 2, 6
            output_stack = output_stack.reshape(c, *(self.patches_per_axis[i] * patch_size))

            index = [slice(None) for _ in range(4)]
            if i > 0:
                start = int(patch_size[i-1] / 2)
                index[i] = slice(start, start + self.patches_per_axis[i][i - 1] * patch_size[i - 1])
            outputs[i][tuple(index)] = output_stack

        outputs = torch.nanmean(outputs, axis=0)

        # The following assertion should hold. It's commented out, because of performance reasons.
        # assert not torch.any(torch.isnan(outputs)), 'There should be no NaN value left in outputs'
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
        if config.use_gpu:
            image = image.to(0)

        applier = GeneratorApplier(image.shape, config)
        output = applier.apply_generator(image, generator)
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

    if config.save_images:
        save_images(config.output_file, outputs, config.output_datasets)

    if config.show_images:
        viewer = neuroglancer.Viewer()
        for name, image in zip(config.output_datasets, outputs):
            show_image(viewer, image, name=name)
        webbrowser.open(str(viewer), new=0, autoraise=True)
        input("Done?")
