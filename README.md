# CNAPs: Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes
This repository contains the code to reproduce the few-shot classification experiments carried out in
[Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes](https://arxiv.org/abs/1906.07697).

The code has been authored by: John Bronskill, Jonathan Gordon, and James Reqeima.

## Dependencies
This code requires the following:
* Python 3.5 or greater
* PyTorch 1.0 or greater
* TensorFlow 1.14 or greater

## GPU Requirements
* To train or test a full CNAPs model with auto-regressive FiLM adaptation, 2 GPUs with 16GB or more memory is required.
* To train or test a CNAPs model with FiLM only adaptation, only 1 GPU with 16GB or more memory is required.

## Installation
1. Clone or download this repository.
2. Configure Meta-Dataset:
    * Follow the the "User instructions" in the Meta-Dataset repository (https://github.com/google-research/meta-dataset)
    for "Installation" and "Downloading and converting datasets". This will take some time.
3. Install additional test datasets (MNIST, CIFAR10, CIFAR100):
    * Change to the $DATASRC directory: ```cd $DATASRC```
    * Download the MNIST test images: ```wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz```
    * Download the MNIST test labels: ```wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz```
    * Download the CIFAR10 dataset: ```wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz```
    * Extract the CIFAR10 dataset: ```tar -zxvf cifar-10-python.tar.gz```
    * Download the CIFAR100 dataset: ```wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz```
    * Extract the CIFAR10 dataset: ```tar -zxvf cifar-100-python.tar.gz```
    * Change to the ```cnaps/src``` directory in the repository.
    * Run: ```python prepare_extra_datasets.py```

## Usage
To train and test CNAPs on Meta-Dataset:

1. First run the following two commands.
    
    ```ulimit -n 50000```

    ```export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>```
    
    Note the above commands need to be run every time you open a new command shell.
    
2. Execute the ```run_cnaps.py``` script from the ```src``` directory following the instructions at the beginning
of the file.

## Expected Results
In the [Meta-Dataset paper](https://arxiv.org/abs/1903.03096), the train/validation/test splits were not specified for
most of the datasets. The results originally stated in our [paper](https://arxiv.org/abs/1906.07697) were based on our
version of Meta-Dataset which likely used different splits. However, the authors of the Meta-Dataset paper have recently
specified a version of the dataset with known splits. They have restated their results on the
[Meta-Dataset GitHub site](https://github.com/google-research/meta-dataset). This code-base uses their version of the
dataset and reader and hence the results from this code will differ from what is currently published in our
[paper](https://arxiv.org/abs/1906.07697). The results below are what you should expect by running this code, and they
should be compared to the reproduced results on the
[Meta-Dataset GitHub site](https://github.com/google-research/meta-dataset).

**Models trained on all datasets**

| Dataset       | No Adaptation | FiLM     | FiLM + AR |
| ---           | ---           | ---      | ---       |
| ILSVRC        | 43.8±1.0      | 51.3±1.0 | 52.3±1.0  |
| Omniglot      | 60.1±1.3      | 88.0±0.7 | 88.4±0.7  |
| Aircraft      | 53.0±0.9      | 76.8±0.8 | 80.5±0.6  |
| Birds         | 55.7±1.0      | 71.4±0.9 | 72.2±0.9  |
| Textures      | 60.5±0.8      | 62.5±0.7 | 58.3±0.7  |
| Quick Draw    | 58.1±1.0      | 71.9±0.8 | 72.5±0.8  |
| Fungi         | 28.6±0.9      | 46.0±1.1 | 47.4±1.0  |
| VGG Flower    | 75.3±0.7      | 89.2±0.5 | 86.0±0.5  |
| Traffic Signs | 55.0±0.9      | 60.1±0.9 | 60.2±0.9  |
| MSCOCO        | 41.2±1.0      | 42.0±1.0 | 42.6±1.1  |
| MNIST         | 76.0±0.8      | 88.6±0.5 | 92.7±0.4  |
| CIFAR10       | 61.5±0.7      | 60.0±0.8 | 61.5±0.7  |
| CIFAR100      | 44.8±1.0      | 48.1±1.0 | 50.1±1.0  |

## Contact
To ask questions or report issues, please open an issue on the issues tracker.

## Citation
If you use this code, please cite our [paper](https://arxiv.org/abs/1906.07697):
```
@article{requeima2019fast,
  title={Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes},
  author={Requeima, James and Gordon, Jonathan and Bronskill, John and Nowozin, Sebastian and Turner, Richard E},
  journal={arXiv preprint arXiv:1906.07697},
  year={2019}
}
```
