# Source code for the master thesis "Cycle-Consistent Adversarial Learning for Biomedical Image Segmentation"
This repository contains the code for the data generation and evaluation of the trained models. The training code is implemented in another repository, which is a fork of the original CycleGAN repo.

## Generating synthetic data
The files in the directory `image_synthesis` can be used to create the synthetic masks.

### Epithelium
To generate the synthetic epithelium masks, use the script `epithelial_sheet_cli.py`. The script takes 3 positional arguments:

1. The algorithm to use for generation. Available are
    - `v_1_0`
    - `v_1_1` (used for dataset 1 in the thesis)
    - `v_1_2` (used for dataset 2 in the thesis)
    - `v_1_3` (used for dataset 3 in the thesis)
2. The config, which specifies the parameters for the algorithm. Note that the first two version numbers of the config should coincide with the used algorithm. Available are:
    - `v_1_0_0`
    - `v_1_1_0`
    - `v_1_1_1` (used for dataset 1 in the thesis)
    - `v_1_2_0` (used for dataset 2 in the thesis)
    - `v_1_3_0` (used for dataset 3 in the thesis)
3. The output file where the synthetic image will be written to. This should be an h5 file.

Additionally, the optional parameter `--name` can be used to specify the name of the dataset in the h5 file. If not set, the name of the config will be used.

Example to create dataset 1:

```sh
python epithelial_sheet_cli.py v_1_1 v_1_1_1 output_file.h5
```

### Brainbows
To generate the synthetic Brainbow masks, use the script `brainbows_cli.py`. The script takes 3 positional arguments:

1. The algorithm to use for generation. Available are
    - `v_1_0`
    - `v_1_1` 
    - `v_1_2`
    - `v_1_3` (used for the mask in the thesis)
2. The config, which specifies the parameters for the algorithm. Note that the first two version numbers of the config should coincide with the used algorithm. Available are:
    - `v_1_0_0`
    - `v_1_0_1`
    - `v_1_1_0`
    - `v_1_1_1`
    - `v_1_2_0`
    - `v_1_3_0`
    - `v_1_3_1` (used for the mask in the thesis)
3. The output file where the synthetic image will be written to. This should be an h5 file.

Additionally, the optional parameter `--name` can be used to specify the name of the dataset in the h5 file. If not set, the name of the config will be used.

Example to create the dataset in the thesis:

```sh
python brainbows_cli.py v_1_3 v_1_3_1 output_file.h5
```

The resulting file contains the color representation of the Brainbow mask. To use the affinity representation, the `datasetB_creation_func` configuration option for the training can be used, which convert the color representation to an affinity representation before training.

## Evaluation
To evaluate trained models, use the code provided in the `evaluation` directory.
How evaluation is performed is determined by a config file. The file `evaluation/config/template.py` is a template for such a configuration, which documents all the options.
Important for a config file is that it contains an object that is named `config`, and which conforms to the class `Config`.
The evaluation works by loading the state dict of a saved trained model and using this model for inference. The output of the model is then post-processed like described in the thesis.

For evaluating the epithelium, the `segmentation_config` of the `config` object should be an instance of the `EpithelialSegmentationConfig` class. For the Brainbows, it should be an instance of the `BrainbowSegmentationConfig` class.

Use the `evaluation/main.py` file to start an evaluation by providing it with the config file:

```sh
python evaluation.py evaluate {config-file-name}
```

To check, whether a config file has the correct format, you can verify it using:

```sh
python evaluation.py verify {config-file-name}
```

Alternatively, you can only execute one step in the evaluation process, which are to translate the image and to compute a segmentation and evaluate it using the translation. To see how to use these, use the `-h` option for the `main.py` script. If only one step is executed, more parameters need to be specified in the config file. These options are summarized at the end of the respective configs in the template file.

The evaluation will either be printed on the screen or saved to a file, depending on the option `segmentation_config.save_results`.